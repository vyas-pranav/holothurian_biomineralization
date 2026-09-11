# 04 · Cytoskeletal drug perturbations

**Figure:** Fig. 5C.

Juveniles were incubated for up to 7 days with calcein and one of four treatments:
- no drug (Ctrl);
- DMSO + ethanol;
- latrunculin A (Lat; 50 nM);
- nocodazole (Noc; 5 µM).

The ossicles were then extracted and imaged, and those with calcein signal (new growth) were counted by eye (SI, "Drug perturbation experiments").

`plot_growth_fraction.py` draws the horizontal bar chart of the fraction of ossicles with growth for animals 1 and 2 and their total. Each bar is labelled with its count.

- **Inputs:** none. The counts are typed into the script.
- **Output:** `fraction_values_plot_hori.png`, written to the current working directory.
- **Environment:** `environment-analysis.yml`.

| Condition | Animal 1 | Animal 2 | Total |
|---|---|---|---|
| Ctrl | 23/29 | 28/37 | 51/66 |
| DMSO | 31/34 | 31/51 | 62/85 |
| Lat | 23/41 | 52/68 | 75/109 |
| Noc | 18/97 | 6/39 | 24/136 |

Fig. S14F (nocodazole dose series, five animals per condition) was made with a variant of this script that was not archived. Its counts are printed on the published panel.
