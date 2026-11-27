# DataCo Experiment Summary (real data, run Nov 26, 2026)

Design: three disjoint temporal periods on DataCo (2015-01-01 to 2018-01-31).
Train 99,285 | Calibration 36,104 | Test 45,130 orders. Model:
HistGradientBoosting (300 iterations) on pre-outcome features only
(docs/02 lists every excluded column and why). A 300-tree random forest was
attempted first and abandoned on runtime grounds on this hardware; the
change of model is recorded here and in the script.

## Measured results (test period, untouched by fitting or calibration)
- PR-AUC 0.809 | ROC-AUC 0.734 | Brier 0.197 | Calibration error (ECE) 0.012
- Conformal: q = 0.644 calibrated on the calibration period ONLY.
  Empirical coverage on the future test period: 0.896 (nominal 0.90).
  Prediction sets: 23,297 of 45,130 uncertain (51.6%), 0 empty.
- Context: published DataCo results reporting ~99% accuracy use features
  this project bans as leakage (docs/02). These lower numbers are what the
  same prediction task looks like without leakage and with a temporal test.

## Policy comparison - an honest negative/parameter finding
Under the current simulator parameters the cost-minimizing oracle expedites
NOTHING on DataCo, so oracle = no-action = rule-based (total simulated cost
342,900; the scheduled-days rule also never fires because DataCo scheduled
days max out at 4, below the rule's 7-day trigger). The model policies
expedite aggressively and cost far more (point: 3,122,308; uncertainty-aware:
7,005,608) while reducing stockouts (3,763 and 778 vs 8,308).

Interpretation, stated plainly: the simulator's parameters (daily demand 25
units/day, safety stock 120) were calibrated to the synthetic generator's
shipment sizes (1-40 units). DataCo order quantities are far smaller, so a
single order's avertable stockout cost almost never exceeds the $180
expedite cost. The comparison therefore measures PARAMETER SENSITIVITY, not
a meaningful policy ranking on real data yet.

Defined next step (not yet done): aggregate the order stream into
replenishment lots (e.g., by day x region x category) so simulator units
match the demand parameters, and run a sensitivity analysis over
expedite_cost / daily_demand / safety_stock before drawing any policy
conclusion on real data. No parameter was retuned after seeing these
results; doing so silently would invalidate the comparison.
