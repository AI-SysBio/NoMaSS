# Reproducing the figures of the paper

Scripts, data and precomputed results behind

> Pélissier A., Phan M., Beerenwinkel N., Rodríguez Martínez M. *Gillespie-based simulation and inference for non-Markovian stochastic reaction networks.* Briefings in Bioinformatics 27(5), 2026. [doi:10.1093/bib/bbag486](https://doi.org/10.1093/bib/bbag486)

Every script imports the engine from the `nomass/` folder at the root of the repository and uses paths relative to its own folder, so it can be run from anywhere:

```bash
pip install -r requirements.txt -r paper/requirements.txt
python paper/stem_cell_differentiation/fig3_stem_cell_model.py
```

In the paper, the rejection-based method is called REGIR (its original name); it is the `MOSAIC` method of NoMaSS.

## Figure to script

| Figure / table | Script | Notes |
| --- | --- | --- |
| Fig. 1B (speed) | `benchmarking/fig1B_speed.py` | Compares against nMGA (`benchmarking/nmga.py`) |
| Fig. 1C (rate bound ratio) | `benchmarking/fig1C_rmax_ratio.py` | |
| Fig. 1D, 1E (accuracy) | `benchmarking/fig1DE_accuracy.py` | Earth mover's distance (`pyemd`) |
| Fig. 2 (bacterial growth) | `bacterial_growth/fig2_bacterial_growth_model.py` | Single-cell data in `bacterial_growth/data/` |
| Fig. S1, S2 (division time fits) | `bacterial_growth/figS1S2_division_time_fit.py` | |
| Fig. 3 (stem cell differentiation) | `stem_cell_differentiation/fig3_stem_cell_model.py` | Data in `stem_cell_differentiation/data/` |
| Table S1 (fitted parameters) | `stem_cell_differentiation/tableS1_stem_cell_optimization.py` | Results cached in `optimization/` |
| Fig. 4B, 4C (mRNA phase diagrams) | `rna_transcription/fig4BC_phase_diagram.py` | Results written to `phase_diagrams/` |
| Fig. 4D (sensitivity analysis) | `rna_transcription/fig4D_sensitivity.py` | Sobol indices (`SALib`) |
| Fig. 4E (shape parameter) | `rna_transcription/fig4E_shape_parameter.py` | |
| SBML import/export | `sbml/` | See `sbml/README.md` |

The benchmarking and optimisation scripts are slow (hours). They ship with their precomputed results (`benchmarking/computed/`, `stem_cell_differentiation/optimization/`, `rna_transcription/*.npy`) and plot those by default. Each script has a `recompute_*` flag near the top of `main()` to regenerate the results.

## Data

* `bacterial_growth/data/CC_data/`: *Caulobacter crescentus* single-cell division times, from Iyer-Biswas et al., PNAS 111(45), 2014 (see `ReadMe.txt` in that folder).
* `bacterial_growth/data/E-coli.xlsx`, `B-subtilis.xlsx`: single-cell growth data, see the accompanying PDFs and `bacterial_growth/literature/`.
* `stem_cell_differentiation/data/stem_cell_data.csv`: population fractions during ESC to EPI to NPC differentiation, see the accompanying PDF.

The `literature/` folders contain the reference articles of the methods compared in the paper.
