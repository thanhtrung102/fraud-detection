"""
Optuna Hyperparameter Tuning Module
====================================

Hyperparameter optimization using Optuna for XGBoost, LightGBM, and CatBoost.
"""

import os
import numpy as np
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import Pool, cv as catboost_cv
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress LightGBM warnings
os.environ['LIGHTGBM_VERBOSITY'] = '-1'


def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 20,
                 cv_folds: int = 5, random_state: int = 42, n_jobs: int = 1) -> dict:
    """
    Tune XGBoost hyperparameters using Optuna.

    Args:
        X: Training features
        y: Training labels
        n_trials: Number of Optuna trials
        cv_folds: Number of CV folds
        random_state: Random seed
        n_jobs: Number of parallel jobs for model training

    Returns:
        Best hyperparameters dictionary
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': random_state,
            'n_jobs': n_jobs
        }

        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # Use n_jobs=1 for CV to avoid nested parallelism deadlocks
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['n_jobs'] = n_jobs

    print(f"  XGBoost best AUC-ROC: {study.best_value:.4f}")
    return best_params


def tune_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 20,
                  cv_folds: int = 5, random_state: int = 42, n_jobs: int = 1) -> dict:
    """
    Tune LightGBM hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': random_state,
            'verbose': -1,
            'force_col_wise': True,  # Avoid OpenMP conflicts
            'n_jobs': n_jobs
        }

        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        # Use n_jobs=1 for CV to avoid nested parallelism deadlocks
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['verbose'] = -1
    best_params['n_jobs'] = n_jobs

    print(f"  LightGBM best AUC-ROC: {study.best_value:.4f}")
    return best_params


def tune_catboost(X: np.ndarray, y: np.ndarray, n_trials: int = 20,
                  cv_folds: int = 5, random_state: int = 42, n_jobs: int = 1) -> dict:
    """
    Tune CatBoost hyperparameters using Optuna.
    Uses CatBoost's native CV to avoid sklearn compatibility issues.
    """
    # Create CatBoost Pool
    pool = Pool(X, y)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': random_state,
            'verbose': 0,
            'thread_count': n_jobs,
            'allow_writing_files': False,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC'
        }

        # Use CatBoost's native cross-validation
        cv_results = catboost_cv(
            pool=pool,
            params=params,
            fold_count=cv_folds,
            shuffle=True,
            partition_random_seed=random_state,
            verbose=False
        )

        # Return the best AUC from CV
        return cv_results['test-AUC-mean'].max()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['random_seed'] = random_state
    best_params['verbose'] = 0
    best_params['thread_count'] = n_jobs

    print(f"  CatBoost best AUC-ROC: {study.best_value:.4f}")
    return best_params


def tune_all_models(X: np.ndarray, y: np.ndarray, n_trials: int = 20,
                    cv_folds: int = 5, random_state: int = 42, n_jobs: int = -1) -> dict:
    """
    Tune all base models using Optuna.

    Args:
        X: Training features
        y: Training labels
        n_trials: Number of trials per model
        cv_folds: Number of CV folds
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with best params for each model
    """
    # Determine actual n_jobs
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 4

    print(f"Starting Optuna hyperparameter tuning ({n_trials} trials per model, {n_jobs} cores)...")

    print("\n  Tuning XGBoost...")
    xgb_params = tune_xgboost(X, y, n_trials, cv_folds, random_state, n_jobs)

    print("\n  Tuning LightGBM...")
    lgbm_params = tune_lightgbm(X, y, n_trials, cv_folds, random_state, n_jobs)

    print("\n  Tuning CatBoost...")
    catboost_params = tune_catboost(X, y, n_trials, cv_folds, random_state, n_jobs)

    return {
        'xgboost': xgb_params,
        'lightgbm': lgbm_params,
        'catboost': catboost_params
    }


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20,
                               n_classes=2, random_state=42)

    best_params = tune_all_models(X, y, n_trials=5)
    print("\nBest parameters:")
    for model, params in best_params.items():
        print(f"\n{model}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
