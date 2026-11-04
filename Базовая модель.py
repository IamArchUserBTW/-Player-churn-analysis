#!/usr/bin/env python3
"""
Базовый скрипт для всей команды: baseline + единые метрики.

Запуск:
    python baseline.py путь/к/файлу.csv
    python baseline.py путь/к/файлу.csv --model lgbm

Каждый участник берёт свою модель (функция make_model), а загрузка данных,
разбиение и метрики остаются общими, чтобы результаты можно было сравнивать.

Зависимости: pandas, numpy, scikit-learn  (для --model lgbm ещё lightgbm)
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
GROUP = "UserID"
SEED = 42

# Столбцы, которых не должно быть среди признаков
DROP = [TARGET, GROUP, "ServerTime"]


# --------------------------------------------------------------------------
# ЕДИНЫЕ МЕТРИКИ (одинаковые для всех участников, не менять без согласования)
# --------------------------------------------------------------------------
def evaluate(y_true, proba, threshold=None):
    """
    Основная метрика: PR-AUC (average precision) - из-за сильного дисбаланса.
    Дополнительно: ROC AUC, recall/precision/F1 по классу 1, log-loss, Brier.
    Порог: если не задан, берётся порог с лучшим F1 на переданных данных.
    ВАЖНО: для честной оценки порог подбирайте на валидации, не на тесте.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if threshold is None:
        threshold = best_f1_threshold(y_true, proba)
    pred = (proba >= threshold).astype(int)
    return {
        "PR_AUC": average_precision_score(y_true, proba),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "recall_class1": recall_score(y_true, pred, zero_division=0),
        "precision_class1": precision_score(y_true, pred, zero_division=0),
        "F1_class1": f1_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y_true, proba),
        "threshold": float(threshold),
        "positive_rate_true": float(y_true.mean()),
    }


def best_f1_threshold(y_true, proba):
    qs = np.unique(np.quantile(proba, np.linspace(0.5, 0.999, 200)))
    best_t, best_f = 0.5, -1
    for t in qs:
        f = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


# --------------------------------------------------------------------------
# ДАННЫЕ И РАЗБИЕНИЕ
# --------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path, low_memory=False)
    # убираем столбцы с одним значением (неинформативные)
    const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    return df


def split_by_user(df, test_size=0.2):
    """Игрок не попадает одновременно в train и test (нет утечки по игроку)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df[GROUP]))
    return df.iloc[tr], df.iloc[te]


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


# --------------------------------------------------------------------------
# МОДЕЛИ: каждый участник заменяет/добавляет свою
# --------------------------------------------------------------------------
def make_model(name):
    if name == "logreg":      # baseline
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                      class_weight="balanced_subsample",
                                      random_state=SEED)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05,
                              is_unbalance=True, random_state=SEED,
                              verbose=-1)
    raise ValueError(f"Неизвестная модель: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="logreg",
                    choices=["logreg", "rf", "lgbm"])
    args = ap.parse_args()

    df = load(args.csv)
    train, test = split_by_user(df)
    # порог подбираем на части train (валидация), а не на тесте
    train_in, val = split_by_user(train, test_size=0.2)

    feats = [c for c in df.columns if c not in DROP]
    pipe = Pipeline([("prep", build_preprocessor(df[feats])),
                     ("model", make_model(args.model))])
    pipe.fit(train_in[feats], train_in[TARGET])

    thr = best_f1_threshold(val[TARGET].values,
                            pipe.predict_proba(val[feats])[:, 1])
    proba_test = pipe.predict_proba(test[feats])[:, 1]
    res = evaluate(test[TARGET].values, proba_test, threshold=thr)

    print(f"Модель: {args.model}")
    print(json.dumps({k: round(v, 4) for k, v in res.items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
