# Reproducing the figures of the paper

Scripts, data and precomputed results behind

> Pélissier A., Phan M., Beerenwinkel N., Rodríguez Martínez M. *Gillespie-based simulation and inference for non-Markovian stochastic reaction networks.* Briefings in Bioinformatics 27(5), bbag486 (2026). [doi:10.1093/bib/bbag486](https://doi.org/10.1093/bib/bbag486)

The article with its supplementary material is included as [Pelissier2026_NoMaSS_main_and_SI.pdf](Pelissier2026_NoMaSS_main_and_SI.pdf), and the published figures are in `figures/` for comparison with the script outputs.

Every script imports the engine from the `nomass/` folder at the root of the repository and uses paths relative to its own folder, so it can be run from anywhere:

```bash
pip install -r requirements.txt -r paper/requirements.txt
python paper/stem_cell_differentiation/fig3_stem_cell_model.py
```

Slow computations ship with their precomputed results (`results/` and `optimization/` folders) and are plotted from them by default. Each script has a `recompute` flag near the top to regenerate the results with the NoMaSS engine of this repository; the docstring of each script says how long that takes. Results cached from the original implementation are noted as such in the docstrings.

## Main figures

| Figure | Script | Notes |
| --- | --- | --- |
| Fig. 2A, 2B (symmetric division) | `bacterial_growth/fig2AB_symmetric_division.py` | Growth curves cached |
| Fig. 2C (runtime per event) | `bacterial_growth/fig2C_runtime_per_event.py` | Raw timings of the paper cached; recomputing takes hours |
| Fig. 2D-F (crowding, MOSAIC vs DelaySSA) | `bacterial_growth/fig2DEF_crowding_mosaic_vs_delay.py` | 50 realisations per method cached |
| Fig. 3 (stem cell differentiation) | `stem_cell_differentiation/fig3_stem_cell_model.py` | Data in `stem_cell_differentiation/data/` |
| Fig. 4B (SLC7A5 after EGF stimulation) | `rna_transcription/fig4B_figS7_tableS3_egf_stimulation.py` | qPCR data in `rna_transcription/data/qpcr/` |
| Fig. 4C (steady-state mRNA distributions) | `rna_transcription/fig4C_steady_state_distributions.py` | |
| Fig. 4D (Sobol sensitivity of the CV) | `rna_transcription/fig4D_figS8_sobol_sensitivity.py` | Needs `SALib` |

## Supplementary figures and tables

| Figure / table | Script | Notes |
| --- | --- | --- |
| Fig. S1, Table S1 (division-time fits) | `bacterial_growth/figS1_tableS1_division_time_fits.py` | Single-cell data in `bacterial_growth/data/` |
| Fig. S2 (symmetric division, analytics) | `bacterial_growth/figS2_symmetric_division_analytics.py` | |
| Fig. S3 (growth-rate heatmaps) | `bacterial_growth/figS3_growth_rate_heatmaps.py` | 19 x 19 grids cached; recomputing takes hours |
| Fig. S4 (microbial growth of *C. crescentus*) | `bacterial_growth/figS4_microbial_growth.py` | |
| Fig. S5, Table S2 (stem cell fits, logit transform) | `stem_cell_differentiation/figS5_tableS2_stem_cell_fits.py` | maxLIPO (`dlib`, optional); results cached in `optimization/` |
| Fig. S6 (mRNA oscillations) | `rna_transcription/figS6_mRNA_oscillations.py` | Panels B-C regenerated with NoMaSS |
| Fig. S7, Table S3 (four genes after EGF stimulation) | `rna_transcription/fig4B_figS7_tableS3_egf_stimulation.py` | See the docstring for the score definition |
| Fig. S8 (Sobol sensitivity of mean, CV, entropy) | `rna_transcription/fig4D_figS8_sobol_sensitivity.py` | |
| Fig. S9 (phase diagrams) | `rna_transcription/figS9_phase_diagrams.py` | No cached results: always recomputes (hours) |

Fig. 1 and Fig. 4A are schematics. The shared model of the transcription scripts is in `rna_transcription/rna_model.py`.

## Data

* `bacterial_growth/data/CC_data/`: *Caulobacter crescentus* single-cell division times, from Iyer-Biswas et al., PNAS 111(45), 2014 (see `ReadMe.txt` in that folder).
* `bacterial_growth/data/E-coli.xlsx`, `B-subtilis.xlsx`: single-cell growth data, see the accompanying PDFs and `bacterial_growth/literature/`.
* `stem_cell_differentiation/data/stem_cell_data.csv`: population fractions during ESC to EPI to NPC differentiation (Stumpf et al., Cell Systems 2017), see the accompanying PDF.
* `rna_transcription/data/qpcr/`: mRNA fold changes after EGF stimulation for SLC7A5, DDB2, NFYC and CREG1, digitised from Zeisel et al., Mol. Syst. Biol. 2011 (see the README in that folder).

The `literature/` folders contain the reference articles of the data sets and of the Laplace Gillespie and nMGA methods.
