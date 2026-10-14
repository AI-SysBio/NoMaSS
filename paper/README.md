# Reproducing the figures of the paper

Scripts, data and precomputed results behind

> Pélissier A., Phan M., Beerenwinkel N., Rodríguez Martínez M. *Gillespie-based simulation and inference for non-Markovian stochastic reaction networks.* Briefings in Bioinformatics 27(5), bbag486 (2026). [doi:10.1093/bib/bbag486](https://doi.org/10.1093/bib/bbag486)

Every script imports the engine from the `nomass/` folder at the root of the repository and uses paths relative to its own folder, so it can be run from anywhere:

```bash
pip install -r requirements.txt -r paper/requirements.txt
python paper/stem_cell_differentiation/fig3_stem_cell_model.py
```

## Figure to script

| Figure / table | Script | Notes |
| --- | --- | --- |
| Fig. 2A, 2B (symmetric division) | `bacterial_growth/fig2AB_symmetric_division.py` | Growth curves cached in `results/` |
| Fig. 2C (runtime per event) | not provided | |
| Fig. 2D-F (crowding, MOSAIC vs DelaySSA) | `bacterial_growth/fig2DEF_crowding_mosaic_vs_delay.py` | 50 realisations per method cached in `results/` |
| Supplementary microbial growth figure | `bacterial_growth/figS_microbial_growth_SI.py` | Single-cell data in `bacterial_growth/data/` |
| Supplementary division-time fits | `bacterial_growth/figS1S2_division_time_fit.py` | |
| Fig. 3 (stem cell differentiation) | `stem_cell_differentiation/fig3_stem_cell_model.py` | Data in `stem_cell_differentiation/data/` |
| Table S1 (fitted parameters) | `stem_cell_differentiation/tableS1_stem_cell_optimization.py` | maxLIPO (`dlib`, optional); results cached in `optimization/` |
| Fig. 4B, 4C (mRNA phase diagrams) | `rna_transcription/fig4BC_phase_diagram.py` | Results written to `phase_diagrams/` |
| Fig. 4D (sensitivity analysis) | `rna_transcription/fig4D_sensitivity.py` | Sobol indices (`SALib`) |
| Fig. 4E (shape parameter) | `rna_transcription/fig4E_shape_parameter.py` | |

The optimisation and phase-diagram scripts are slow (hours). They ship with their precomputed results (`bacterial_growth/results/`, `stem_cell_differentiation/optimization/`, `rna_transcription/*.npy`) and plot those by default. Each script has a `recompute_*` flag near the top of `main()` to regenerate the results.

The figures as published are in `figures/` for comparison with the script outputs.

## Data

* `bacterial_growth/data/CC_data/`: *Caulobacter crescentus* single-cell division times, from Iyer-Biswas et al., PNAS 111(45), 2014 (see `ReadMe.txt` in that folder).
* `bacterial_growth/data/E-coli.xlsx`, `B-subtilis.xlsx`: single-cell growth data, see the accompanying PDFs and `bacterial_growth/literature/`.
* `stem_cell_differentiation/data/stem_cell_data.csv`: population fractions during ESC to EPI to NPC differentiation, see the accompanying PDF.

The `literature/` folders contain the reference articles of the data sets and of the Laplace Gillespie and nMGA methods.
