# Research Improvements B-J - Consolidated Results (Nov 26, 2026)

All runs use the unchanged protocol: model and conformal threshold fitted
on train/calibration only, judged on the untouched future test period.
All figures refreshed Nov 26, 2026 under the hardened DATE-CUT temporal
split (no timestamp shared across a split boundary); no verdict changed.

## B - Data-quality stress testing (reports/stress_tests.csv)
Baseline: PR-AUC 0.809, ECE 0.012, coverage 0.896, uncertain 51.7%.
- Shipping-mode mislabels (20% flipped): the worst stressor. PR-AUC
  0.778 (-0.031), ECE 0.038 (3x), coverage 0.886. The most important
  feature is also the most fragile input - consistent with Improvement A.
- Missing numeric values (30%): PR-AUC 0.804, ECE doubles to 0.022, but
  coverage RISES to 0.906 because imputation pushes predictions toward
  the middle and uncertain sets grow to 56.8% - coverage preserved by
  vagueness, calibration genuinely worse.
- Category noise (10%), timestamp delay (30 days), duplicates (+10%):
  no measurable degradation (all within 0.001-0.004 of baseline). The
  pipeline is robust to these, and the report says so plainly.

## C - Dependence-aware lot aggregation (reports/dependence_*.csv)
Gaussian-copula Monte-Carlo P(any order in lot late), validated at
rho = 0 against the analytic independence value (0.829 = 0.829).
Mean lot risk: independence 0.829 | rho 0.3: 0.810 | rho 0.6: 0.786 |
rho 0.9: 0.755 | max bound 0.736 | true lot late rate 0.820.
Dependence assumptions move lot risk by up to 7 points and change which
lots get expedited (6,835 vs 6,715 at the checked scenario), but policy
RANKINGS and the H2 verdict do not change: aggregation choice is a
second-order effect, and the uncertainty-aware policy's failure at lot
level is not an artifact of the independence assumption.

## D - Recovery-tail analysis (reports/recovery_tail.csv)
300 re-simulations per policy at three scenarios. Tails are thin in this
simulator (delay draws bounded at 6 days): e.g. high-pressure no-action
mean $1,072,759 vs P99 $1,100,949 (+2.6%). No policy's ranking flips in
the tail. Caveat recorded: the oracle mask is optimal for the reference
delay draw only; under re-drawn delays it is a strong heuristic, not a
per-replication optimum (at the high-pressure scenario its re-drawn mean,
~$479,000, exceeds point prediction's $426,054 for exactly this reason).

## E - Cost robustness and policy regret (reports/policy_regret_summary.csv)
Mean decision regret vs the oracle across the 36 scenarios:
uncertainty-aware $1,101,074 (worst) | point prediction $920,996 |
no action / rule-based ~$199,672 | oracle $0 by construction.
Worst-case regret tells the same story ($2,388,000 for
uncertainty-aware). Under this simulator's economics, acting on
uncertainty is the most expensive mistake available.

## F - Value of information (reports/value_of_information.csv)
Model information has positive value in only 7 of 36 scenarios;
uncertainty information in only 4 of 36. Where perfect information has
value, the model's mean efficiency is -3.30: acting on the model
destroys roughly three times the value perfect information would
create. The exception is the high-pressure corner (cheap expedites,
high demand, low safety stock), where prediction pays - the conditions,
not the average, are the finding.

## G - Causal analysis (docs/04-causal-assessment.md)
Verdict: NOT defensible on these data, and no causal claim is made.
Shipping mode is self-selected and confounded; the operationally
relevant intervention (expediting) has no observed counterpart in the
data at all; unobserved confounding (weather, carrier capacity) is
plausible and unmeasurable. The document states what evidence would
change the verdict (randomized pilot, natural experiment, or an
explicitly assumption-laden design reported as hypothesis-generating).

## H - Explainability stability (reports/importance_stability.csv)
Permutation importance (average-precision drop) per period, one model.
The top of the ranking is stable: Shipping Mode dominates in all three
periods (0.129 / 0.109 / 0.108), then scheduled days, then payment Type.
Below the top three, importances are near-zero and ranks shuffle - the
full-ranking Spearman correlations are only 0.28-0.42 across periods,
LOW precisely because the remaining features are interchangeable noise
in every period. Defensible operator explanation: the top drivers.
Anything deeper is period-specific noise and should not be presented
as fact.

## I - Severe disruption testing (scripts/run_disruption_test.py)
Outcome shock injected into the final 30 test days (Jan 2018): flipping
25% / 50% of on-time shipments to late moves the window late rate
0.558 -> 0.668 / 0.779. Findings:
- Coverage does NOT collapse (0.932 baseline -> 0.933 / 0.936): sets in
  that window are predominantly both-class, so coverage-by-vagueness
  absorbs outcome shocks.
- The model is blind to the shock (predicted mean p 0.564 vs actual
  0.779 at the high dose): probabilities and calibration degrade while
  set coverage holds - the two must be monitored separately.
- Lot-level policy costs are identical across doses: the any-late lot
  definition saturates (a lot already late cannot get later), a stated
  limitation of the aggregation for disruption analysis.

## J - Independent-dataset validation: Olist, Brazil
(reports: scripts/run_olist_validation.py output; data gitignored)
Compatibility confirmed: purchase timestamps, estimated vs actual
delivery dates, order-time features. 96,470 delivered orders,
2016-2018, late rate 8.1% (vs DataCo 54.8%).
Test period: PR-AUC 0.130 (base rate 0.081), ROC-AUC 0.717, Brier
0.057, ECE 0.011. Conformal coverage 0.960 vs 0.90 nominal -
OVER-coverage, the opposite direction from DataCo, with only 13.6%
uncertain sets. The pipeline transfers mechanically; the conformal
layer's behaviour does not transfer numerically: at a low base rate the
calibration scores concentrate near zero, the threshold lands high
(q = 0.895), and sets cover by including the negative class almost
always. Same lesson as Improvement A, from the opposite side: coverage
is a per-deployment audit, never an assumption.
