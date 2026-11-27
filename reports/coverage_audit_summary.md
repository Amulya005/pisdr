# Improvement A - Subgroup Uncertainty Reliability Audit (Nov 26, 2026)

Full table: reports/coverage_audit.csv (76 groups x 6 dimensions).
Method: conformal threshold calibrated on the calibration period only
(q = 0.644, nominal 0.90); coverage broken down on the untouched test
period (n = 45,133). A group is flagged when its Wilson 95% CI excludes
0.90 and n >= 200.

## Headline answer: YES - the overall 0.896 hides subgroup failure
Marginal coverage is an average over groups that fail in opposite ways.

### Shipping Mode - the serious finding
- Second Class (n = 8,805): coverage 0.768, gap -0.132, CI (0.759, 0.777).
  Only 2.7% of its prediction sets are uncertain: the model is CONFIDENT
  AND WRONG there - singleton sets that miss the true outcome 23% of the
  time. This is the reliability-to-decision gap in concrete form: an
  uncertainty-aware policy keyed on "uncertain" flags would not flag
  exactly the shipments whose sets fail.
- First Class (n = 6,960): 0.950 (+0.050), 1.7% uncertain - over-covered.
- Standard Class (n = 26,950): 0.921, but 77.3% uncertain - coverage is
  achieved by issuing both-class (uninformative) sets.
- Same Day (n = 2,418): 0.938, 88.0% uncertain - same pattern.

### Risk bands - the tails fail by overconfidence
- p < 0.3 (n = 490): coverage 0.594, gap -0.306, 0% uncertain. When the
  model says "confidently not late", it is wrong 41% of the time.
- 0.5-0.7 (n = 1,957): 0.819 (-0.081). p >= 0.7: 0.857
  (-0.043), 0% uncertain.
- 0.3-0.5: 0.928 - covered, but ~79% uncertain.
Pattern: a single global conformal threshold delivers MARGINAL coverage
only. The middle of the risk range is covered by vagueness; the tails
are covered by nothing - they are simply overconfident.

### Time windows - statistically flagged, practically negligible
W1 0.891, W3 0.894, W4 0.907 (n ~ 11,283 each). CIs exclude 0.90 only
because n is large; gaps are <= 0.009. No meaningful temporal degradation
of coverage inside the test period - consistent with the negligible PSI
drift measured earlier.

### Regions and categories - moderate, localized
Northern Europe (n = 4,360): 0.862 (-0.038) - real under-coverage.
Central America (n = 10,408): 0.887 (-0.013) - mild. Eastern Asia and
South Asia over-cover (+0.03). Among large categories, Men's Footwear
(n = 4,879) under-covers most at 0.880 (-0.020); several small categories
over-cover by +0.03 to +0.05.

## What this means
1. Reporting only overall coverage (0.896) for this system would be
   misleading: one large subgroup sits at 0.768 and the confident tails
   at 0.59-0.86.
2. "Uncertain" is not a reliable proxy for "risky" in this system -
   failures concentrate where sets are singleton. Any decision policy
   that trusts singleton sets inherits the Second Class failure mode.
3. The defined remedy for a future phase is group-conditional
   (Mondrian) conformal calibration - e.g. a separate threshold per
   shipping mode - evaluated on the same untouched test period. Not
   implemented here; this audit is the evidence that motivates it.

## Limitations
Subgroup coverage is descriptive, not causal: Second Class shipments
differ from First Class in more than the label. Small groups (n < 200)
are reported in the CSV but not judged. Multiple comparisons: with 76
groups, some CI exclusions are expected by chance - the findings above
rest on effect size (gaps of -0.13 and -0.29), not on flag counts.
