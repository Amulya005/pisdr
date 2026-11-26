"""Models. Advanced models are always compared against meaningful baselines:
a majority-class rule, logistic regression, and a random forest. LightGBM is
used when installed; otherwise sklearn's HistGradientBoosting is the fallback
so results never depend on an optional dependency being present."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_models(seed=42):
    models = {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed,
                                                n_jobs=-1),
    }
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(random_state=seed, verbose=-1)
    except ImportError:
        models["gradient_boosting"] = HistGradientBoostingClassifier(random_state=seed)
    return models
