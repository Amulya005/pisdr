# Methodology

1. Validation: temporal and rolling-origin splits for all headline results.
   Random splits are computed only to quantify their overstatement (RQ1).
2. Uncertainty: split-conformal prediction with empirical coverage reported
   next to every interval claim; calibration error (ECE) and Brier score
   reported alongside discrimination (PR-AUC).
3. Shift: performance is reported separately for the injected disruption
   window and normal periods; PSI drift scores accompany feature drift claims.
4. Decisions: policies are compared in the transparent simulator
   (src/pisdr/simulator.py) against rule-based and oracle references;
   decision regret is the primary decision metric. Simulator parameters live
   in configs/config.yaml; sensitivity analysis varies them before any
   conclusion is drawn.
5. Honesty rules: real and synthetic results are never merged; simulated
   benefits are never described as real-world benefits; no causal claims are
   made from observational data; negative results are reported.
6. Reproducibility: every experiment runs from configs/config.yaml with a
   fixed seed; environment is pinned by requirements.txt.

## Findings on real data (Nov 26, 2026)
See reports/dataco_experiment_summary.md and reports/sensitivity_summary.md.
Headlines: H1 not supported on DataCo (negligible temporal drift, PSI
<= 0.065); H2 rejected at lot level under threshold/set-membership policy
definitions (0/36 scenarios); the cost-minimizing oracle shows uncaptured
headroom in high-pressure scenarios. The scheduled-days rule never fires
on DataCo (max scheduled days 4 < rule threshold 7) and is reported as
equivalent to no-action there.

## Improvement A - subgroup coverage audit (Nov 26, 2026)
See reports/coverage_audit_summary.md. Overall test coverage 0.896 hides
subgroup failure: Second Class shipping coverage 0.768 (confident and
wrong, 2.7% uncertain), p<0.3 risk band 0.594, while Standard Class
reaches 0.921 only via 77.3% both-class sets. Marginal conformal coverage
does not imply conditional coverage; group-conditional (Mondrian)
calibration is the motivated future phase.

## Improvements B-J (Nov 26, 2026)
See reports/improvements_summary.md for the consolidated results and
docs/04-causal-assessment.md for the Improvement G verdict (causal
claims not defensible on these data; none are made). Key additions:
data-quality stress tests (shipping-mode mislabels are the fragile
input), Gaussian-copula dependence-aware lot aggregation (rankings
unchanged), policy regret and value-of-information analysis (model
information has positive value in 7/36 scenarios), recovery-tail
simulation, permutation-importance stability (top drivers stable, the
rest is noise), severe-disruption testing (coverage holds by vagueness
while calibration degrades), and an Olist external replication
(coverage 0.960 - over-coverage at an 8.1% base rate).
