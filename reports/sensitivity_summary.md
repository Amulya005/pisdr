# Sensitivity Analysis + RQ1 + Drift on Real DataCo (run Nov 26, 2026)

Full grid: reports/sensitivity_analysis.csv (36 scenarios x 5 policies).
All runs use the leakage-safe temporal model (test PR-AUC 0.810) and
conformal threshold calibrated only on the calibration period.

## RQ1 - random vs temporal validation on real data (CORRECTED Nov 26, 2026)
Correction: the first version of this comparison permuted the FULL
dataframe for the random split, so the random model trained on
test-period rows and the figures below replace the contaminated ones
(random-test 0.822 / temporal-test 0.824) originally reported here.
The corrected design trains BOTH models only on pre-test information
(the random split is drawn from the train+calibration pool, same
training size as the temporal model) and evaluates on the same
untouched future test period.
- Random-trained model, evaluated on a random pre-test slice: PR-AUC 0.831
- The SAME model evaluated on the future temporal test: PR-AUC 0.809
- Temporal-trained model on the temporal test: PR-AUC 0.809
Measured cleanly, a small random-split inflation exists (~0.022 PR-AUC,
~2.7% relative) - the direction H1 predicts, but far below the >=10%
relative gap H1 hypothesized, and random vs temporal TRAINING makes no
difference on the future test (0.811 vs 0.810). The drift numbers
explain why: PSI train->test is negligible for every numeric feature
(0.000-0.065) and for predicted risk (0.009). DataCo is temporally
almost homogeneous, so evaluation choice matters little and conformal
coverage survives (0.896 vs 0.90 nominal). H1's strong form is not
supported on this dataset; it remains plausible for genuinely shifting
streams - the synthetic demo, which injects a real disruption window,
is where split choice matters. Reported as measured, not as hoped.

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
- H1 (random splits overstate performance): strong form NOT supported
  on DataCo (corrected gap ~0.017 PR-AUC, ~2% relative, vs the >=10%
  hypothesized); supported only in the synthetic shift demo.
- H2 (uncertainty-aware policy lowers regret at equal service level):
  REJECTED on real lots under current definitions (0/36 scenarios).
- H3 (simulator impact ranking diverges from classifier ranking): untested.
- H4/H5: partially exercised via the grid; no dominance claim survives.

## Corrective fixes applied Nov 26, 2026 (second review round)
- Issue 1 (RQ1 leakage): fixed as described above; this section corrected.
- Issue 3 (conformal edge case): conformal_threshold now returns inf when
  the required rank exceeds the calibration size (standard split-conformal
  convention) instead of clipping to n; edge test added. No effect on the
  real run (calibration n = 36,104).
- Issue 4 (ECE boundary): the first ECE bin now includes p == 0.0 (it was
  silently dropped; a discriminating probe returned 0.0 where the true
  ECE is 1.0). Boundary tests added for p = 0 and p = 1. No effect on the
  reported ECE (0.011): the model emits no exact-0.0 probabilities here.
- Issue 5 (lot dependence): aggregate_lots now supports method="max"
  (perfect positive dependence, the Frechet lower bound) alongside the
  independence baseline, with a test; the two bracket the dependence
  range. Correlated-event simulation remains future work.
- Issue 6 (dashboard): the dashboard now discovers the real-DataCo and
  synthetic-demo reports separately, labels them, and defaults to the
  real one.

## Split boundary hardened (Nov 26, 2026)
All temporal splits are now DATE-CUT: boundaries are timestamp values,
so no timestamp appears on both sides of a split (previously the split
sliced by row position and a few orders sharing the exact boundary
timestamp straddled it). Every experiment was re-run under the hardened
split; all headline numbers moved by <= 0.003 and no verdict changed.
Current split sizes: train 99,282 | calibration 36,104 | test 45,133.
