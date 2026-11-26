# Improvement G - Causal Intervention Analysis: Defensibility Assessment

## Verdict: causal claims are NOT defensible on the available data.
No causal effect is estimated in this project, and none should be
claimed from DataCo or Olist. The policy comparisons in this repository
are DECISION ANALYSIS inside a simulator whose intervention effects are
assumed by construction (expediting removes a shipment's stockout cost),
not estimated from data. That distinction is maintained everywhere.

## Why the observational data cannot carry causal claims
1. No randomization and no instrument. Nothing in either dataset
   randomly assigns shipping mode, expediting, or any other lever.
2. The headline "treatment" is self-selected. Shipping mode is chosen by
   the customer and is confounded by price sensitivity, distance,
   product mix, and promised lead time - all plausibly related to
   lateness itself. The Improvement A audit shows shipping-mode groups
   differ sharply in reliability; reading that as an EFFECT of the mode
   would be exactly the confounding error.
3. The intervention that matters operationally - expediting a shipment -
   has no observed counterpart in the data at all. There is no column
   recording who was expedited and what then happened, so its effect is
   unidentified, not merely confounded.
4. Unobserved confounding is plausible and unmeasurable here: weather,
   carrier capacity, and regional disruption affect both choices and
   outcomes and appear in neither dataset.
5. Positivity is doubtful in subgroups (e.g. Same Day is rare and
   concentrated), so even adjustment-based estimates would rest on
   extrapolation.

## What WOULD make causal analysis defensible
- A randomized expedite/intervention pilot (even a small one) run by an
  operator, with pre-registered outcomes; or
- A natural experiment: a carrier policy change, a regional service
  suspension, or a threshold rule (e.g. expedite offered above a weight
  cutoff) enabling difference-in-differences or regression discontinuity;
- Failing those, an explicitly assumption-laden observational analysis
  (named confounders, overlap diagnostics, negative controls, and a
  sensitivity analysis for unmeasured confounding) reported as
  hypothesis-generating only.

## What this project claims instead
Descriptive reliability statements (coverage audits), predictive
performance under temporal validation, and simulator-based decision
analysis under stated assumptions. Those are the claims the evidence
supports, and they are the contribution.
