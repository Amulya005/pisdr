# PISDR Research Foundation (Stage 0)

## Originality assessment
The individual components of PISDR exist elsewhere: disruption prediction with ML is a
crowded field; calibrated probabilistic disruption forecasting exists (Foresight
Learning, Lightning Rod Labs); uncertainty-aware graph models exist (UQ-CRGATNet);
commercial platforms (Everstream Analytics, Resilinc, Interos) sell prediction plus
impact mapping at enterprise prices. PISDR does not claim novelty for prediction or
uncertainty quantification themselves.

What is missing — and is this project's contribution — is a public, reproducible
evaluation of whether predictions improve operational decisions: decision regret,
service level, and recovery cost of an uncertainty-aware policy vs. point-prediction
and rule-based policies, under temporal shift, with leakage-safe validation. Academic
papers optimize classification metrics (several with leakage-inflated ~99% accuracies
on DataCo); commercial tools are black boxes with no published decision evaluation.

## Existing approaches reviewed
1. Foresight Learning (Lightning Rod Labs, arXiv) — LLM trained end-to-end for
   calibrated probabilistic forecasts of disruption indexes from news. No operational
   decision layer.
2. UQ-CRGATNet (Frontiers in AI, 2026) — graph attention + tabular transformer + MC
   dropout ensemble on DataCo; reported 99.87% accuracy / AUC 1.000, treated here as a
   leakage cautionary example (DataCo's late-delivery label is date-derived).
3. Temporal GNN for cascading disruptions (Mannava et al., 2025) — GCN-GRU on a
   semi-synthetic network (AUROC 0.844); authors note no public benchmark combines
   multi-tier supplier relationships, real-time logistics, and verified disruption
   labels.
4. Li et al. (Int. J. Production Research, 2025) — deep learning + survival analysis +
   XAI for part time-to-survive at a US automotive manufacturer; proprietary data.
5. Everstream / Resilinc / Interos — commercial, proprietary data and methods, no
   published reproducible decision evaluation.

## Defensible research gap
No open, reproducible system evaluates whether uncertainty estimates improve
supply-chain recovery decisions (decision regret, service level, simulated recovery
cost) under temporal distribution shift using leakage-safe validation.

## Hypotheses (falsifiable)
- H1: Random-split PR-AUC exceeds rolling temporal-test PR-AUC by >=10% relative on
  DataCo after a documented leakage audit.
- H2: At equal simulated service level, an uncertainty-aware policy achieves lower
  decision regret than a point-prediction policy (or higher service level at equal cost).
- H3: The impact simulator's top-decile high-impact shipments overlap the classifier's
  top-decile risk list by <70%.
- H4: Under injected forecast error and shift, the uncertainty-aware policy's regret
  grows more slowly than fixed rule-based policies.
- H5: A multi-objective policy (cost, service risk, uncertainty penalty)
  Pareto-dominates the best rule-based policy in a majority of simulated scenarios.
  A negative result is a finding and will be reported as such.

## Datasets
- DataCo Smart Supply Chain (Kaggle, CC0) — primary; known leakage traps, audited in W2.
  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
- NY Fed Global Supply Chain Pressure Index — external disruption indicator.
  https://www.newyorkfed.org/research/policy/gscpi
- GDELT Project — global events. https://www.gdeltproject.org
- NOAA Storm Events — timestamped weather disruptions. https://www.ncdc.noaa.gov/stormevents/
- Lightning Rod Labs eval set (Hugging Face) — calibration benchmark.
  https://huggingface.co/datasets/LightningRodLabs/supply-chain-predictions
- Synthetic supplier-network data — simulator-generated, documented parameters, always
  labeled synthetic, reported separately.

## Architecture (cost-conscious AWS)
S3 (raw/processed) -> Python/pandas ETL -> versioned Parquet features -> model training
(local + one SageMaker job) -> versioned model registry -> FastAPI inference (Lambda +
API Gateway) -> Streamlit dashboard -> scheduled drift checks (PSI/KS) with CloudWatch
logs. Every dashboard recommendation shows the prediction, its interval, simulator
assumptions, and the policy rule that fired.

## Major risks
DataCo leakage; no public dataset linking disruptions to inventory outcomes (hence the
simulator, with explicit assumptions and sensitivity analysis); class imbalance and
drift may make models look weak (a finding, not a failure); scope creep (GNN, LLM
forecasting, causal claims are parked as stretch goals); calibration degrades under
shift (H4 tests graceful decision degradation); cloud cost (free-tier-first, teardown
after demos).
