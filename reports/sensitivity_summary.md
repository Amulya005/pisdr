# Sensitivity Analysis + RQ1 + Drift on Real DataCo (run Nov 26, 2026)

Full grid: reports/sensitivity_analysis.csv (36 scenarios x 5 policies).
All runs use the leakage-safe temporal model (test PR-AUC 0.810) and
conformal threshold calibrated only on the calibration period.

## RQ1 - random vs temporal validation on real data: NOT supported here
- Random-trained model, evaluated on a random test slice: PR-AUC 0.822
- The SAME model evaluated on the future temporal test: PR-AUC 0.824
- Temporal-trained model on the temporal test: PR-AUC 0.810
There is no meaningful random-split inflation on DataCo. The drift numbers
explain why: PSI train->test is negligible for every numeric feature
(0.000-0.065) and for predicted risk (0.009). DataCo is temporally almost
homogeneous, so random and temporal evaluation coincide and conformal
coverage survives (0.897 vs 0.90 nominal). H1 remains plausible for
genuinely shifting streams - the synthetic demo, which injects a real
disruption window, is where split choice matters - but on this dataset
H1 is not supported. Reported as measured, not as hoped.

## Lot-level policy comparison (8,335 lots; median 6 units; lot = any
constituent order late, so lot late rate is 0.82; 79.7% of lots contain at
least one conformal-uncertain order)
- The scheduled-days rule NEVER fires on DataCo (rule triggers at >= 7
  scheduled days; DataCo's maximum is 4), so rule_based is identical to
  no_action in all 36 scenarios. Counting ties honestly: doing nothing is
  cost-minimal among the active policies in 29 of 36 scenarios.
- point_prediction wins the remaining 7 scenarios - all high-demand /
  low-safety-stock / cheap-expedite cases where stockout pressure is real.
- uncertainty_aware is cheaper than point_prediction in 0 of 36 scenarios.
  H2 is REJECTED under the current policy definitions on real lots: at lot
  level almost every lot is "uncertain", so the policy degenerates into
  expedite-almost-everything and buys service level at the highest cost.
- The oracle's margin over no-action in high-pressure scenarios (e.g.
  expedite $60, demand 50/day, safety 60: oracle 289,844 vs no-action
  1,054,784) shows real headroom exists - but neither model policy captures
  it, because both expedite indiscriminately at lot level.

## Diagnosis and the defined next iteration (future work, not done here)
Threshold policies degenerate when the unit of decision is a lot whose
"late" label means ANY constituent order was late. The decision layer
should use expected cost, not set membership: expedite iff the PREDICTED
avertable stockout cost (units-at-risk x stockout cost) exceeds the
expedite cost - the oracle's rule applied to predictions instead of known
outcomes. Lot-level probability should also be modeled directly rather
than aggregated from order-level sets. These are design changes for a
next stage; no parameter was tuned after seeing these results.

## Hypothesis status after this stage
- H1 (random splits overstate performance): NOT supported on DataCo
  (gap ~0.002 PR-AUC); supported only in the synthetic shift demo.
- H2 (uncertainty-aware policy lowers regret at equal service level):
  REJECTED on real lots under current definitions (0/36 scenarios).
- H3 (simulator impact ranking diverges from classifier ranking): untested.
- H4/H5: partially exercised via the grid; no dominance claim survives.
