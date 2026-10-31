# Results Folder Guide


## Authoritative manuscript-facing results
The authoritative consolidated analysis is the Google Sheet:
- Reproducibility_Analysis_Master


The exact manuscript-facing CSV extracts are in:
- Manuscript_Tables/


These files are intended to match the values reported in the revised LaTeX manuscript. In particular, combined_anova.csv uses the expanded capacity-block specification including W x alpha.


## Full statistical outputs
Full_Statistical_Outputs/ contains detailed intermediate and diagnostic outputs retained for reproducibility:
- capacity_expanded_spec.csv — expanded capacity-block Type-II ANOVA including W x alpha, HC3 p-values, and Holm-adjusted inference.
- anova_diagnostics.csv — residual diagnostics used in the robustness discussion.
- capacity_normalization.csv — standardized capacity quantities.
- capacity_ratio_sensitivity.csv — sensitivity analysis based on N_g W / D_c and N_v Q / D_c.
- convergence.csv and convergence_extra.csv — detailed stored-history convergence outputs.
- structural_precheck.csv — aggregate structural-feasibility calculations.
- benchmark_summary.csv — repeated-run benchmark summary.


## Archived pre-expanded ANOVA files
The earlier ANOVA/HC3/multiplicity CSVs that preceded addition of W x alpha were moved to:
05_Supplementary/Archive_PreExpanded_ANOVA/


They are retained only as an audit trail and are **not** the final manuscript-facing specification. For manuscript values use Reproducibility_Analysis_Master and Manuscript_Tables/combined_anova.csv.


## Manuscript table extracts
Manuscript_Tables/ currently includes:
- case_study_characteristics.csv
- feasibility_scalability.csv
- benchmark_robustness.csv
- table5_variability.csv
- table6_variability.csv
- convergence_summary.csv
- combined_anova.csv
- nonparametric_priority.csv
- statistical_robustness_selected.csv