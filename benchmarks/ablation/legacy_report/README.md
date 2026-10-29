# Ablation Study Reproducibility Package

This folder contains the code, source result workbooks, processed outputs, figures, and final report produced for the VND0–VNDzou computational comparison and the Barreto ablation study.

## Folder structure

- `analysis_pipeline.py`: statistical analysis and figure-generation code.
- `build_report.py`: Word-report generation code.
- `sources/code`: baseline VND0 and VNDzou implementations.
- `sources/ablation_code`: all algorithm variants used in the Barreto ablation experiments.
- `sources/core`: VND0 summaries and VNDzou Stage-2 workbooks for Barreto, Prins, and Tuzun.
- `sources/vnd0_raw`: instance-level VND0 run workbooks, separated by benchmark family.
- `sources/ablation`: the 13 configuration-level Barreto result workbooks.
- `sources/literature`: benchmark result tables used programmatically by the analysis pipeline.
- `analysis_outputs`: normalized tables, instance-level and aggregate comparisons, statistical tests, ablation contrasts, figures, logs, and validation outputs.
- `VND0_VNDzou_Computational_Study.docx` and `.pdf`: the final manuscript-ready Word report and its verified PDF rendering.

## Reproduction

Run the scripts from the package root after installing Python 3 and the required packages (`pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`, and `python-docx`):

```bash
python analysis_pipeline.py
python build_report.py
```

The analysis script recreates the files in `analysis_outputs`; the report script builds `VND0_VNDzou_Computational_Study.docx` from those outputs. The folder hierarchy mirrors the paths expected by both scripts, so the package can be downloaded and run from its root without path changes.

## Scope notes

- The main comparison covers 75 common instances across Barreto, Prins, and Tuzun.
- Prins uses 50 VND0 runs and 30 VNDzou runs; Barreto and Tuzun use 30 runs per method. The report therefore distinguishes best-run comparisons from instance-blocked statistical inference.
- The Barreto ablation results are interpreted most strongly for matched configuration contrasts and more cautiously for non-matched exploratory comparisons.
- Duplicate intermediate report renders and temporary PDF text extractions are intentionally excluded; the verified final PDF is retained.

Prepared: 22 July 2026
