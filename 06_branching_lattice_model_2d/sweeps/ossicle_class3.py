# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2022-2025 Pranav Vyas
"""Ossicle class of the 2D branching model, imported by the parameter sweeps.

Code release for Vyas et al., "Cellular construction of topologically complex
biomineral lattices in holothurians".

Figure panels : Figs. 6D-G, 6I, 6L-M, S20A, S21-S24, S26 (through the sweep scripts in
                this folder; ../plotting/replot_final_pruned.py imports it to re-render
                the k-D grid images)
What it does  : Defines Ossicle: seed graph, tip growth along
                F_tot = alpha*F_cell + eta*F_pers + beta*F_pair, bifurcation, box-search
                fusion/budding, tip deactivation at tau_min, morphogen field and plotting.
                The replication-interval function is passed in by the calling script as
                params['rep_interval_func'] (logarithmic timing rule, SI Eq. 48:
                rep_interval = max(tau_min, tau_max * ln(c(x)) / kappa)).
                Differences from ../model/growth_sim_v18_main.py: sister tips (the two
                daughters of one bifurcation) may fuse with each other, sisters are left
                out of the pair force F_pair, other seed options (create_seed_pair_angle,
                initiate_graph), and plot styling.
Inputs        : none (library module)
Outputs       : none (the calling scripts save figures and graphs)
Environment   : environment-models.yml (Python 3.13); the paper runs used
                environment-analysis.yml (Python 3.7)
Run           : not run directly; used as `from ossicle_class3 import Ossicle`
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, Point
import copy
import math
import os
import datetime
import itertools
from dataclasses import dataclass
from matplotlib import patheffects as pe
from scipy.special import k0  # For modified Bessel function of the second kind
import matplotlib as mpl

#set font helvetica
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']


#%% Dataclasses for Nodes and Cells (lineage removed)
@dataclass
class NodeInit:
    node_id: int
    pos: list
    pos_hist: list   # History of positions
    active: int      # 1 for active, 0 for inactive
    pair_nodes: list   # Paired node id ([] if none)
    sister_nodes: list # Sister node id from a split ([] if none)
    direction: np.ndarray  # Growth direction vector

@dataclass
class CellInit:
    cell_id: int
    pos: list
    pos_hist: list
    active: int


#%% Ossicle Simulation Class
class Ossicle:
    def __init__(self, start_nodes, start_cells, params):
        """
        Initialize the simulation with starting nodes and cells,
        and set simulation parameters from the provided dictionary.
        """
        # Simulation parameters
        self.alpha       = params.get('alpha', 1e-6)      # Cell repulsion strength
        self.beta        = params.get('beta', 0.01)       # Pair node attraction strength
        self.eta         = params.get('eta', 0.1)         # Persistence force strength
        self.step_length = params.get('step_length', 0.1)   # Growth step length
        self.seed_size   = params.get('seed_size', 4.0)     # Seed size for initial structure
        self.density     = params.get('density', 2.71)      # Material density (not directly used here)
        self.tip_dia     = params.get('tip_dia', 1.0)       # Diameter of the tip
        self.angle       = params.get('angle', 108 * np.pi / 180)  # Branching angle (in radians)
        self.box_size    = 10 * self.step_length             # Collision check box size
        self.pair_thresh_length = params.get('pair_thresh_length', 5)  # Threshold distance for pair check in microns
        
        # Default replication interval (will be overridden by cell contributions)
        self.rep_time    = 20/2//self.step_length       
        # User-provided replication interval function:
        # Now it takes (cell_list, pos_node) and returns a replication interval (> 0).
        self.rep_interval_func = params.get('rep_interval_func', None)

        # Cell force asymmetry function: modifies the cell force based on position.
        self.cell_force_asym_func = params.get('cell_force_modifier', None)

        # Persistence force function: modifies the persistence force based on position.
        self.pers_force_func = params.get('pers_force_modifier', None)

        # Growth rate scaling factor function: modifies the growth step length based on position.
        self.growth_step_scaling_func = params.get('growth_step_scaling_func', None)
        
        # Noise parameters (in radians for angles, in hours for time)
        self.angle_noise = params.get('angle_noise', 0.0)
        self.time_noise = params.get('time_noise', 0.0)
        self.seed_angle_noise = params.get('seed_angle_noise', 0.0)
        
        # Plotting aesthetics parameters
        self.node_size = params.get('node_size', 50)
        self.edge_width = params.get('edge_width', 2)
        self.node_cmap = params.get('node_cmap', plt.cm.viridis)
        self.color_by_topology = params.get('color_by_topology', True)
        
        # Additional parameter for cell translation (step size)
        self.cell_translation_step = params.get('cell_translation_step', 0.1)
        
        # List to optionally store structure positions
        self.structure_pos = []
        self.time = 0  # Initialize simulation time
        self.growth_center = np.array([0, 0])  # Define growth center as (0,0)

        #morphogen parameters
        self.D = params.get('morphogen_diffusion_coefficient', 10)  # Diffusion coefficient
        self.Q = params.get('morphogen_release_rate', 1.0)     # Release rate
        self.lambda0 = params.get('morphogen_decay_rate', 0.1)
        self.tau_min = params.get('min_rep_interval', 1.0)
        self.tau_max = params.get('max_rep_interval', 10.0)
        self.scale = params.get('morphogen_scale', 0.1)

        #plot limit size
        self.plot_lim_size = params.get('plot_limit', 50)
        
        # Create a unique node id generator to avoid conflicts after removals
        self.node_id_counter = itertools.count(0)
        
        # Initialize the seed graph.
        if 'initial_branch_count' in params:
            branch_count = params['initial_branch_count']
            if 'pair_angle' in params:
                # Create the seed graph and then adjust branch 2.
                self.oss_graph = self.create_seed_pair_angle(branch_count, params['pair_angle'])
            else:
                self.oss_graph = self.create_seed_graph(branch_count)
        else:
            self.oss_graph = self.initiate_graph()
        
        # After seed graph creation, update each node’s active flag based on degree and tag as "seed"
        for node in self.oss_graph.nodes():
            if self.oss_graph.degree(node) == 1:
                self.oss_graph.nodes[node]['active'] = True
            else:
                self.oss_graph.nodes[node]['active'] = False
            self.oss_graph.nodes[node]['node_role'] = "seed"
        
        # Maintain a list of active nodes (nodes with degree 1)
        self.active_nodes = [node for node in self.oss_graph.nodes() if self.oss_graph.degree(node) == 1]
        
        # Initialize cells from starting conditions (lineage removed)
        self.n_cells = 0
        self.cell_list = []
        for cell_pos in start_cells:
            cell_obj = CellInit(cell_id=self.n_cells, pos=cell_pos, pos_hist=[cell_pos],
                                active=1)
            self.cell_list.append(cell_obj)
            self.n_cells += 1
        
        # List of active cell indices
        self.active_cells = list(range(self.n_cells))

        # Calculate minimum replication interval for nodes after which growth stops
        self.min_rep_int = self.rep_interval_func(self.cell_list, np.array([10000000,0]))
        print(f"Minimum replication interval: {self.min_rep_int}")
    
    #%% Helper Methods for Active Lists
    def add_active_node(self, node):
        """Add a node to the list of active nodes if not already present."""
        if node not in self.active_nodes:
            self.active_nodes.append(node)
            self.oss_graph.nodes[node]['active'] = True
    
    def remove_active_node(self, node):
        """Remove a node from the active nodes list if present."""
        if node in self.active_nodes:
            self.active_nodes.remove(node)
            self.oss_graph.nodes[node]['active'] = False
    
    def add_active_cell(self, cell_id):
        """Add a cell index to the active cell list."""
        if cell_id not in self.active_cells:
            self.active_cells.append(cell_id)
    
    def remove_active_cell(self, cell_id):
        """Remove a cell index from the active cell list."""
        if cell_id in self.active_cells:
            self.active_cells.remove(cell_id)

    #%% Graph Initialization
    def initiate_graph(self):
        """
        Create the initial graph representing the ossicle structure:
        a horizontal seed with branching at the ends.
        Node IDs are obtained from a unique id generator.
        """
        G = nx.Graph()
        # Create the central seed node at the growth center (not active).
        center = next(self.node_id_counter)
        G.add_node(center, pos=self.growth_center, parent=None,
                   direction=np.array([0, 0]), pair_nodes=[], sister_nodes=[],
                   last_event_time=self.time, create_time=self.time,
                   active=False, node_role="seed")
        self.center = center  # Store the center node id.
        
        # Build the right half of the seed (horizontal line to the right).
        right_nodes = []
        prev_node = center
        num_steps = round(self.seed_size / self.step_length / 2)
        for ii in range(num_steps):
            new_node = next(self.node_id_counter)
            pos = self.growth_center + np.array([(ii+1)*self.step_length, 0])
            G.add_node(new_node, pos=pos, parent=prev_node,
                       direction=np.array([1, 0]), pair_nodes=[], sister_nodes=[],
                       last_event_time=self.time, create_time=self.time,
                       active=False, node_role="seed")
            G.add_edge(prev_node, new_node)
            right_nodes.append(new_node)
            prev_node = new_node

        # Create right-end branches.
        if right_nodes:
            parent_for_upper = right_nodes[-1]
            new_node_upper = next(self.node_id_counter)
            pos_upper = np.array([self.seed_size/2 + self.step_length * np.cos(self.angle/2),
                                  self.step_length * np.sin(self.angle/2)])
            G.add_node(new_node_upper, pos=pos_upper, parent=parent_for_upper,
                       direction=np.array([np.cos(self.angle/2), np.sin(self.angle/2)]),
                       pair_nodes=[], sister_nodes=[], 
                       last_event_time=self.time, create_time=self.time,
                       active=True, node_role="seed")
            G.add_edge(parent_for_upper, new_node_upper)
            
            parent_for_lower = right_nodes[-2] if len(right_nodes) >= 2 else center
            new_node_lower = next(self.node_id_counter)
            pos_lower = np.array([self.seed_size/2 + self.step_length * np.cos(self.angle/2),
                                  -self.step_length * np.sin(self.angle/2)])
            G.add_node(new_node_lower, pos=pos_lower, parent=parent_for_lower,
                       direction=np.array([np.cos(self.angle/2), -np.sin(self.angle/2)]),
                       pair_nodes=[], sister_nodes=[new_node_upper],
                       last_event_time=self.time, create_time=self.time,
                       active=True, node_role="seed")
            G.add_edge(parent_for_lower, new_node_lower)
            G.nodes[new_node_upper]['sister_nodes'] = [new_node_lower]

        # Build the left half (mirror image of the right half).
        left_nodes = []
        prev_node = center
        for ii in range(num_steps):
            new_node = next(self.node_id_counter)
            pos = self.growth_center + np.array([-(ii+1)*self.step_length, 0])
            G.add_node(new_node, pos=pos, parent=prev_node,
                       direction=np.array([-1, 0]), pair_nodes=[], sister_nodes=[],
                       last_event_time=self.time, create_time=self.time,
                       active=False, node_role="seed")
            G.add_edge(prev_node, new_node)
            left_nodes.append(new_node)
            prev_node = new_node

        # Create left-end branches.
        if left_nodes:
            parent_for_upper_left = left_nodes[-1]
            new_node_upper_left = next(self.node_id_counter)
            pos_upper_left = np.array([-self.seed_size/2 - self.step_length * np.cos(self.angle/2),
                                       self.step_length * np.sin(self.angle/2)])
            G.add_node(new_node_upper_left, pos=pos_upper_left, parent=parent_for_upper_left,
                       direction=np.array([-np.cos(self.angle/2), np.sin(self.angle/2)]),
                       pair_nodes=[], sister_nodes=[],
                       last_event_time=self.time, create_time=self.time,
                       active=True, node_role="seed")
            G.add_edge(parent_for_upper_left, new_node_upper_left)
            
            parent_for_lower_left = left_nodes[-2] if len(left_nodes) >= 2 else center
            new_node_lower_left = next(self.node_id_counter)
            pos_lower_left = np.array([-self.seed_size/2 - self.step_length * np.cos(self.angle/2),
                                       -self.step_length * np.sin(self.angle/2)])
            G.add_node(new_node_lower_left, pos=pos_lower_left, parent=parent_for_lower_left,
                       direction=np.array([-np.cos(self.angle/2), -np.sin(self.angle/2)]),
                       pair_nodes=[], sister_nodes=[new_node_upper_left],
                       last_event_time=self.time, create_time=self.time,
                       active=True, node_role="seed")
            G.add_edge(parent_for_lower_left, new_node_lower_left)
            G.nodes[new_node_upper_left]['sister_nodes'] = [new_node_lower_left]

        return G

    #%% Seed Graph Initialization
    def create_seed_graph(self, branch_count):
        """
        Create an initial seed graph with a specified number of branches.
        The branches are evenly distributed around the central node.
        """
        G = nx.Graph()
        central = next(self.node_id_counter)
        G.add_node(central, pos=self.growth_center, parent=None,
                   direction=np.array([0, 0]), pair_nodes=[], sister_nodes=[],
                   last_event_time=self.time, create_time=self.time,
                   active=False, node_role="seed")
        self.center = central
        for i in range(branch_count):
            angle_i = 2 * np.pi * i / branch_count #+ np.pi/branch_count
            noise = np.random.uniform(-self.seed_angle_noise, self.seed_angle_noise)
            angle_i += noise
            direction = np.array([np.cos(angle_i), np.sin(angle_i)])
            pos = self.growth_center + (self.seed_size)*direction
            new_node = next(self.node_id_counter)
            G.add_node(new_node, pos=pos, parent=central,
                       direction=direction, pair_nodes=[], sister_nodes=[],
                       last_event_time=self.time, create_time=self.time,
                       active=True, node_role="seed")
            G.add_edge(central, new_node)

        # Get all degree 1 nodes and update their sister_nodes lists.
        active_nodes = [node for node in G.nodes() if G.degree(node) == 1]
        for node in active_nodes:
            for other_node in active_nodes:
                if node != other_node:
                    G.nodes[node]['sister_nodes'].append(other_node)

        return G

    #%% Seed Graph Initialization with given Pair Angle between first two branches
    def create_seed_pair_angle(self, branch_count, pair_angle):
        """
        Create an initial seed graph with a specified number of branches.
        First, distribute branches evenly as in create_seed_graph.
        Then, adjust the position of the second branch so that the angle
        between the first and second branch equals pair_angle (in degrees).
        The remaining branches are left unchanged.
        """
        # Create the evenly distributed seed graph.
        G = self.create_seed_graph(branch_count)
        central = self.center
        
        # Get branch nodes sorted by their creation order (assuming lower IDs were created earlier).
        branch_nodes = sorted([node for node in G.nodes() if node != central])
        if len(branch_nodes) < 2:
            return G  # Not enough branches to modify.
        
        # Identify the first and second branch nodes.
        first_branch = branch_nodes[0]
        second_branch = branch_nodes[1]
        
        # Get the direction of the first branch.
        base_direction = G.nodes[first_branch]['direction']
        base_angle = np.arctan2(base_direction[1], base_direction[0])
        
        # Convert the desired pair_angle (in degrees) to radians.
        pair_angle_rad = pair_angle * np.pi / 180
        new_angle = base_angle + pair_angle_rad
        
        # Compute the new direction and position for the second branch.
        new_direction = np.array([np.cos(new_angle), np.sin(new_angle)])
        new_pos = self.growth_center + self.seed_size * new_direction
        
        # Update the second branch's attributes.
        G.nodes[second_branch]['direction'] = new_direction
        G.nodes[second_branch]['pos'] = new_pos
        
        # Optionally, update the sister_nodes lists for branch nodes.
        branch_nodes = [node for node in G.nodes() if node != central]
        for node in branch_nodes:
            G.nodes[node]['sister_nodes'] = [other for other in branch_nodes if other != node]
        
        return G



    #%% Mathematical Helper Functions
    def make_unit(self, vector):
        """Return the unit vector of the given vector."""
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def rotate_unit(self, unit_vector, angle):
        """Rotate a unit vector by a given angle (radians) and return the new unit vector."""
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        return np.dot(rot_matrix, unit_vector)

    #%% Growth Functions
    def calc_node_forces(self, nodes, cells):
        """
        Calculate the net force acting on each node.
        Forces include:
        - Pair force: attraction/repulsion between paired nodes.
        - Cell force: repulsion from active cells.
        - Persistence force: tendency to continue in the same direction.
        """
        forces = []
        for node in nodes:
            pos_self = self.oss_graph.nodes[node]['pos']
            
            # Calculate force from the paired node (if any)
            pair_force = np.array([0.0, 0.0])
            pair_ids = self.oss_graph.nodes[node].get('pair_nodes', [])
            for pair_id in pair_ids:
                pos_pair = self.oss_graph.nodes[pair_id]['pos']
                unit_dir = self.make_unit(pos_pair - pos_self)
                r = np.linalg.norm(pos_pair - pos_self)
                pair_force += self.beta * unit_dir
            # Cell force: push node away from each active cell
            cell_force = np.array([0.0, 0.0])
            
            pers_force_mod = 0
            for jj in cells:
                pos_cell = self.cell_list[jj].pos
                diff = pos_self - pos_cell
                norm_diff = np.linalg.norm(diff)
                if norm_diff != 0:
                    unit_diff = self.make_unit(diff)
                    asym_modifier = 1.0
                    if self.cell_force_asym_func is not None and callable(self.cell_force_asym_func):
                        asym_modifier = self.cell_force_asym_func(pos_cell, pos_self, unit_diff)
                    cell_force += self.alpha * asym_modifier * unit_diff

                    if self.pers_force_func is not None and callable(self.pers_force_func):
                        pers_force_mod += self.pers_force_func(pos_cell, pos_self, unit_diff)
                    else:
                        pers_force_mod += 1.0

            if len(cells) > 0:
                pers_force_mod /= len(cells)
            else:
                pers_force_mod = 1.0
            
            node_direc = self.oss_graph.nodes[node]['direction']
            pers_force = self.eta * pers_force_mod * node_direc
            forces.append(pair_force + cell_force + pers_force)
        return forces

    def grow_active_tips(self, forces, cells, nodes):
        """
        For each active node, grow a new tip along the direction of the net force.
        The step length is modulated by a scaling factor (if provided) that can depend on the
        node's current position or direction.
        """
        new_active_nodes = []
        sister_switch_dir = {}
        for node, force in zip(list(self.active_nodes), forces):
            pos_node = np.array(self.oss_graph.nodes[node]['pos'])

            asym_modifier = 0
            for cell in cells:
                pos_cell = self.cell_list[cell].pos
                diff = pos_node - pos_cell
                norm_diff = np.linalg.norm(diff)
                if norm_diff != 0:
                    unit_diff = self.make_unit(diff)
                    if self.growth_step_scaling_func is not None and callable(self.growth_step_scaling_func):
                        asym_modifier += self.growth_step_scaling_func(pos_cell, pos_node, unit_diff)
                    else:
                        asym_modifier += 1.0

            if len(cells) > 0:
                asym_modifier /= len(cells)
            else:
                asym_modifier = 1.0

            effective_step_length = self.step_length * asym_modifier

            node_direc = (self.make_unit(force)
                        if np.linalg.norm(force) != 0
                        else self.oss_graph.nodes[node]['direction'])
            
            new_pos = pos_node + node_direc * effective_step_length
            new_node = next(self.node_id_counter)
            parent_last_event = self.oss_graph.nodes[node].get('last_event_time', 0)
            prev_sister_nodes = self.oss_graph.nodes[node].get('sister_nodes', [])
            self.oss_graph.add_node(new_node, pos=new_pos, parent=node,
                                    direction=node_direc,
                                    pair_nodes=self.oss_graph.nodes[node].get('pair_nodes', []),
                                    sister_nodes=prev_sister_nodes,
                                    last_event_time=parent_last_event, create_time=self.time,
                                    active=True)
            self.oss_graph.add_edge(node, new_node)
            new_active_nodes.append(new_node)
            self.remove_active_node(node)
            sister_switch_dir[node] = new_node

        for active_node in new_active_nodes:
            self.add_active_node(active_node)

        for node in new_active_nodes:
            sister_nodes = self.oss_graph.nodes[node]['sister_nodes']
            new_sister_nodes = sister_nodes.copy()
            for sister_node in sister_nodes:
                new_sister_nodes.remove(sister_node)
                if sister_node in sister_switch_dir:
                    new_sister_nodes.append(sister_switch_dir[sister_node])
            self.oss_graph.nodes[node]['sister_nodes'] = new_sister_nodes

    def sim_growth(self):
        """
        Simulate one time step of growth:
        - Calculate forces on active nodes.
        - Grow new tips from active nodes.
        """
        node_forces = self.calc_node_forces(self.active_nodes, self.active_cells)
        self.grow_active_tips(node_forces, self.active_cells, self.active_nodes)

    #%% Topological Changes
    def get_rep_time_for_node(self, node):
        """
        Calculate the replication time for a given node based on the net local morphogen concentration.
        The local concentration is computed by summing the steady-state contributions from all cells.
        """
        pos_node = np.array(self.oss_graph.nodes[node]['pos'])
        rep_int = self.rep_interval_func(self.cell_list, pos_node)
        noise = np.random.uniform(-self.time_noise, self.time_noise)
        return rep_int + noise

    def check_node_replicate(self, active_nodes):
        """
        Check which active nodes are ready to replicate based on their replication intervals.
        """
        rep_nodes = []
        for node in active_nodes:
            rep_interval = self.get_rep_time_for_node(node)
            last_event = self.oss_graph.nodes[node].get('last_event_time', 0)
            if self.time - last_event >= rep_interval:
                rep_nodes.append(node)
        return rep_nodes

    def replicate(self, nodes):
        """
        For each eligible node, create two daughter nodes that split at an angle from the parent's direction.
        When replication occurs:
          - The parent node is tagged as "replicate"
          - The two daughter nodes are added without any special tag.
        """
        for node_id in nodes:
            # Tag the parent node as "replicate"
            self.oss_graph.nodes[node_id]['node_role'] = "replicate"
            
            parent_direc = self.oss_graph.nodes[node_id]['direction']
            parent_pos = np.array(self.oss_graph.nodes[node_id]['pos'])
            
            noise1 = np.random.uniform(-self.angle_noise, self.angle_noise)
            noise2 = np.random.uniform(-self.angle_noise, self.angle_noise)
            
            angle_offset1 = self.angle/2 + noise1
            angle_offset2 = -self.angle/2 + noise2
            
            candidate_direc1 = self.rotate_unit(self.make_unit(parent_direc), angle_offset1)
            candidate_direc2 = self.rotate_unit(self.make_unit(parent_direc), angle_offset2)
            direc1 = candidate_direc1
            direc2 = candidate_direc2

            new_node1 = next(self.node_id_counter)
            new_node2 = next(self.node_id_counter)

            # Daughters get no special tag (node_role is set to an empty string)
            self.oss_graph.add_node(new_node1, pos=parent_pos, parent=node_id,
                                    direction=direc1, pair_nodes=[], sister_nodes=[new_node2],
                                    last_event_time=self.time, create_time=self.time,
                                    active=True, node_role="")
            self.oss_graph.add_edge(node_id, new_node1)
            self.oss_graph.add_node(new_node2, pos=parent_pos, parent=node_id,
                                    direction=direc2, pair_nodes=[], sister_nodes=[new_node1],
                                    last_event_time=self.time, create_time=self.time,
                                    active=True, node_role="")
            self.oss_graph.add_edge(node_id, new_node2)
            self.remove_active_node(node_id)
            self.add_active_node(new_node1)
            self.add_active_node(new_node2)

    def replicate_nodes(self, active_nodes):
        """
        Check for nodes ready to replicate and perform replication.
        """
        rep_nodes = self.check_node_replicate(active_nodes)
        if rep_nodes:
            self.replicate(rep_nodes)

    def get_shortest_path(self, node1, node2):
        """
        Return the shortest path between two nodes in the graph.
        """
        return nx.shortest_path(self.oss_graph, node1, node2)
    
    def check_struct_collision(self, node, distance, box_size, node_list):
        """
        Check for structural collisions for a given node.
        """
        if node not in self.oss_graph.nodes():
            return False, None, False
        x, y = self.oss_graph.nodes[node]['pos']
        min_x, max_x = x - box_size/2, x + box_size/2
        min_y, max_y = y - box_size/2, y + box_size/2
        collision_found = False
        other_node_found = None
        terminal_merge = False
        cand_list = []
        cand_act_list = []
        for other in list(self.oss_graph.nodes()):
            if other == node:
                continue
            if other not in self.oss_graph.nodes():
                continue
            other_x, other_y = self.oss_graph.nodes[other]['pos']
            if min_x <= other_x <= max_x and min_y <= other_y <= max_y:
                dist = np.sqrt((other_x - x)**2 + (other_y - y)**2)
                if dist <= distance:
                    shortest_path = self.get_shortest_path(node, other)
                    if len(shortest_path) > 10:
                        cand_list.append([other, dist])
                        if other in node_list:
                            cand_act_list.append([other, dist])

        if cand_list:
            collision_found = True
            cand_list = np.array(cand_list)
            cand_act_list = np.array(cand_act_list) if cand_act_list else np.empty((0,2))
            if cand_act_list.size > 0:
                other_node_found = int(cand_act_list[np.argmin(cand_act_list[:, 1])][0])
                terminal_merge = True
            else:
                other_node_found = int(cand_list[np.argmin(cand_list[:, 1])][0])
        return collision_found, other_node_found, terminal_merge

    def grow_from_merged_nodes(self, node1, node2, mid_node):
        """
        Update the mid_node's growth direction based on merging nodes.
        Instead of creating a new node, tag the mid_node itself as a "bud" node.
        """
        direc1 = self.oss_graph.nodes[node1]['direction']
        direc2 = self.oss_graph.nodes[node2]['direction']
        direc = self.make_unit((direc1 + direc2) / 2)
        mid_node_pos = self.oss_graph.nodes[mid_node]['pos']
        if np.dot(direc, (mid_node_pos - self.growth_center)) < 0:
            direc = -direc
        self.oss_graph.nodes[mid_node]['direction'] = direc
        self.oss_graph.nodes[mid_node]['last_event_time'] = self.time
        # Tag the mid_node as "bud" and mark it active
        self.oss_graph.nodes[mid_node]['node_role'] = "bud"

        # Create a new node at the same position as mid_node
        new_node = next(self.node_id_counter)
        self.oss_graph.add_node(new_node, pos=mid_node_pos, parent=mid_node,
                                direction=direc, pair_nodes=[], sister_nodes=[],
                                last_event_time=self.time, create_time=self.time,
                                active=True, node_role="")
        self.oss_graph.add_edge(mid_node, new_node)
        self.add_active_node(new_node)

    def check_near_topo_node(self, node, other_node):
        """
        Check whether the two nodes are near a topologically significant node.
        """
        check = False
        rep_interval_node = self.get_rep_time_for_node(node)
        rep_interval_other = self.get_rep_time_for_node(other_node)
        avg_rep_interval = (rep_interval_node + rep_interval_other) / 2.0
        base_node_dist = 5 // self.step_length
        node_dist = int(base_node_dist * (avg_rep_interval / self.rep_time)) if self.rep_time != 0 else base_node_dist

        node_path = nx.single_source_shortest_path(self.oss_graph, node, cutoff=node_dist)
        node_list = set(node_path.keys())
        other_node_path = nx.single_source_shortest_path(self.oss_graph, other_node, cutoff=node_dist)
        other_node_list = set(other_node_path.keys())
        common_nodes = node_list.union(other_node_list)

        for common_node in common_nodes:
            if self.oss_graph.degree(common_node) > 2:
                check = True
                break

        return check

    def annihilate_nodes(self, nodes):
        """
        Process nodes that collide by merging them.
        """
        node_list = list(nodes)
        while node_list:
            node = node_list.pop(0)
            if node not in self.oss_graph.nodes():
                continue
            collision, other_node, terminal_merge = self.check_struct_collision(node, 2 * self.step_length, self.box_size, nodes)
            if collision and other_node is not None:
                pos_new = (np.array(self.oss_graph.nodes[node]['pos']) + np.array(self.oss_graph.nodes[other_node]['pos'])) / 2
                new_node = next(self.node_id_counter)
                self.oss_graph.add_node(new_node, pos=pos_new, parent=node,
                                direction=self.oss_graph.nodes[node]['direction'],
                                pair_nodes=[], sister_nodes=[],
                                last_event_time=self.time, create_time=self.time,
                                active=False, node_role="collision")
                self.oss_graph.add_edge(node, new_node, thickness=self.tip_dia)
                self.oss_graph.add_edge(other_node, new_node, thickness=self.tip_dia)

                mid_node = new_node

                if terminal_merge and (other_node in node_list):
                    node_list.remove(other_node)
                    self.grow_from_merged_nodes(node, other_node, mid_node)
                self.remove_active_node(node)
                self.remove_active_node(other_node)

    def check_node_pairs_prox(self, active_nodes):
        """
        Set pair nodes based on proximity.
        """
        for node in active_nodes:
            self.oss_graph.nodes[node]['pair_nodes'] = []

        for i in range(len(active_nodes)):
            for j in range(i+1, len(active_nodes)):
                node1 = active_nodes[i]
                node2 = active_nodes[j]
                pos1 = self.oss_graph.nodes[node1]['pos']
                pos2 = self.oss_graph.nodes[node2]['pos']
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                is_sister = node2 in self.oss_graph.nodes[node1]['sister_nodes']
                if dist < self.pair_thresh_length and not is_sister:
                    self.oss_graph.nodes[node1]['pair_nodes'].append(node2)
                    self.oss_graph.nodes[node2]['pair_nodes'].append(node1)
    
    def translate_initial_cell(self, cell_index=0, translation_vector=np.array([1, 0]), speed=1):
        """
        Translate one of the initial cells in the center along a selected direction.
        """
        translation_vector = self.make_unit(np.array(translation_vector))
        step = self.cell_translation_step
        new_pos = np.array(self.cell_list[cell_index].pos) + translation_vector * step * speed
        self.cell_list[cell_index].pos = new_pos.tolist()
        self.cell_list[cell_index].pos_hist.append(new_pos.tolist())

    def deactivate_nodes_at_min_rep_time(self, minimal_value=0.5, tol=1e-3):
        """
        Deactivate active nodes if their replication time has reached the minimal value.
        """
        minimal_value = self.min_rep_int

        nodes_to_deactivate = []
        for node in list(self.active_nodes):
            rep_time = self.get_rep_time_for_node(node)
            if rep_time <= minimal_value + tol:
                nodes_to_deactivate.append(node)
        
        for node in nodes_to_deactivate:
            self.remove_active_node(node)

    def prune_graph(self):
        """
        Remove all hanging branches in the structure.
        """
        H = self.oss_graph.copy()
        while True:
            bridge_edges = list(nx.bridges(H))
            if not bridge_edges:
                break
            H.remove_edges_from(bridge_edges)
            isolated_nodes = list(nx.isolates(H))
            if isolated_nodes:
                H.remove_nodes_from(isolated_nodes)
        self.oss_graph = H
        return

    def modify_topology(self):
        """
        Update the structure topology by handling node annihilation (merges),
        checking for nodes ready to replicate, and then translating a selected cell.
        """
        self.annihilate_nodes(self.active_nodes)
        self.replicate_nodes(self.active_nodes)
        self.check_node_pairs_prox(self.active_nodes)
        self.deactivate_nodes_at_min_rep_time()

        # # Translate the cells
        # self.translate_initial_cell(cell_index=0, translation_vector=np.array([1, 1]), speed = 1)
        # self.translate_initial_cell(cell_index=1, translation_vector=np.array([1, -1]), speed = 1)
        # self.translate_initial_cell(cell_index=2, translation_vector=np.array([-1, 1]), speed = 1)
        # self.translate_initial_cell(cell_index=3, translation_vector=np.array([-1, -1]), speed = 1)

        if len(self.active_nodes) == 0:
            return False
        return True

    #function to compute concentration field in the entire domain
    def compute_concentration_field(self, X, Y):
        """
        Compute the concentration field in the entire domain.
        """

        c_total = np.zeros_like(X)
        for cell in self.cell_list:
            cell_pos = np.array(cell.pos)
            dist_array = np.sqrt((X - cell_pos[0])**2 + (Y - cell_pos[1])**2)
            c_total += self.Q/(2*np.pi*self.D) * k0(np.sqrt(self.lambda0/self.D)*dist_array)

        return c_total

    #%% Plotting
    def plot_graph(self, fig, ax):
        """
        Plot the current state of the ossicle graph.
        Nodes are colored by their topological distance from the center.
        In addition, bud, collision, and replicate nodes are highlighted with distinct markers.
        Also plots the current positions of all cells.
        """
        ax.cla()
        pos_dict = nx.get_node_attributes(self.oss_graph, 'pos')

        # Draw edges.
        edge_collection = nx.draw_networkx_edges(
            self.oss_graph, 
            pos_dict, 
            edge_color='#4d4d4d', 
            width=self.edge_width, 
            ax=ax, 
            alpha=0.5,
        )

        if hasattr(edge_collection, 'set_capstyle'):
            edge_collection.set_capstyle('round')

        edge_collection.set_path_effects([
            pe.SimpleLineShadow(
                offset=(1, -1),
                shadow_color="darkgrey",
                alpha=0.75
            ),
            pe.Normal()
        ])


        # Plot the growth center.
        ax.scatter([self.growth_center[0]], [self.growth_center[1]], color='black', s=800, zorder=4)

        # First, plot active nodes as red (this covers all active nodes).
        for active_node in self.active_nodes:
            ax.scatter([pos_dict[active_node][0]], [pos_dict[active_node][1]], color='red', s=100, zorder=4, alpha=0.5)

        # Now, overlay special nodes with distinct markers by role.
        # Group nodes based on their "node_role"
        bud_nodes = [node for node in self.oss_graph.nodes() if self.oss_graph.nodes[node].get('node_role') == "bud"]
        collision_nodes = [node for node in self.oss_graph.nodes() if self.oss_graph.nodes[node].get('node_role') == "collision"]
        replicate_nodes = [node for node in self.oss_graph.nodes() if self.oss_graph.nodes[node].get('node_role') == "replicate"]

        # Use different blue shades for special nodes (all circles)
        role_params = [
            (bud_nodes, {'color': 'white', 'marker': 'o', 's': 50, 'label': 'Bud'}),
            (collision_nodes, {'color': 'steelblue', 'marker': 'o', 's': 50, 'label': 'Collision'}),
            (replicate_nodes, {'color': 'black', 'marker': 'o', 's': 50, 'label': 'Replicate'})
        ]


        # for nodes_group, params in role_params:
        #     if nodes_group:
        #         x_vals = [pos_dict[node][0] for node in nodes_group]
        #         y_vals = [pos_dict[node][1] for node in nodes_group]
        #         ax.scatter(x_vals, y_vals, color=params['color'], marker=params['marker'],
        #                 s=params['s'], label=params['label'], zorder=6)

        # Plot cells with a distinct orange marker.
        cell_positions = [cell.pos for cell in self.cell_list]
        cell_x = [pos[0] for pos in cell_positions]
        cell_y = [pos[1] for pos in cell_positions]
        ax.scatter(cell_x, cell_y, color='orange', s=250, marker='o', label='Cells',
                zorder=5, edgecolors='black', linewidth=2) 
        

        # plot local total concentration fields and add a colorbar
        rad_size = self.plot_lim_size
        ax.set_xlim(-rad_size, rad_size)
        ax.set_ylim(-rad_size, rad_size)
        x_val = np.linspace(-rad_size, rad_size, 300)
        y_val = np.linspace(-rad_size, rad_size, 300)
        X, Y = np.meshgrid(x_val, y_val)
        c_total = self.compute_concentration_field(X,Y)


        #set min and max values for the colorbar
        vmin = 0.5
        vmax = 50

        #map values to cmap logarithmically
        cmap = plt.cm.Blues


        #edit to reduce intensity of the color map
        # cmap.set_under('white', alpha=0.0)

        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        levels = np.geomspace(vmin, vmax, 30)

        # cbar = ax.contourf(X, Y, c_total, levels = levels, cmap=cmap, zorder = -1, norm = norm, extend='both')

        #flip around the x axis
        c_total = np.flip(c_total, axis=0)
        cbar = ax.imshow(c_total, extent=(-rad_size, rad_size, -rad_size, rad_size), cmap=cmap, norm=norm, zorder=-1, alpha=0.9)

        #add colorbar only once
        if not hasattr(self, 'colorbar'):
            self.colorbar = fig.colorbar(cbar, ax=ax, orientation='vertical', format = '%s', ticks=[5e-1, 5e0, 5e1])
            self.colorbar.set_label('Morphogen Concentration', fontsize=15)
            self.colorbar.set_ticks([5e-1, 5e0, 5e1])
            self.colorbar.update_ticks()
            self.colorbar.ax.tick_params(labelsize=15)

        ax.set_aspect('equal')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # ax.set_xlabel('X (um)', fontsize=15)
        # ax.set_ylabel('Y (um)', fontsize=15)
        # ax.set_title('Ossicle Growth Simulation', fontsize=20)  
        ax.text(0.98, 0.98, f"Step: {self.time}", fontsize=15, ha='right', va='top', transform=ax.transAxes)

