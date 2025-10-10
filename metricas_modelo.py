#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitários compartilhados para avaliação de modelos.

Concentra funções usadas em treinamento e avaliação para reduzir
duplicação e facilitar futuras alterações no pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn import metrics


def probabilidades_ou_scores(estimator, X) -> np.ndarray:
    """
    Retorna probabilidades da classe positiva quando disponíveis.
    Caso contrário, devolve scores normalizados ou previsões (0/1).
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, list):
            proba = proba[0]
        return proba[:, 1]

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim > 1:
            scores = scores[:, 0]
        denom = scores.max() - scores.min() + 1e-9
        return (scores - scores.min()) / denom

    preds = estimator.predict(X)
    return np.asarray(preds, dtype=float)


def _divisao_segura(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metricas_classificacao(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    scores: Optional[Iterable[float]] = None,
    positive_label: Any = 1,
) -> Dict[str, Any]:
    """
    Calcula métricas binárias básicas, retornando floats (ou NaN) e a matriz
    de confusão no formato [[tn, fp], [fn, tp]].
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = metrics.accuracy_score(y_true_arr, y_pred_arr)

    auc = float("nan")
    if scores is not None:
        try:
            auc = metrics.roc_auc_score(y_true_arr, scores)
        except Exception:
            auc = float("nan")

    try:
        cm = metrics.confusion_matrix(
            y_true_arr, y_pred_arr, labels=[0, positive_label]
        )
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            cm = np.array([[0, 0], [0, 0]])
    except Exception:
        tn = fp = fn = tp = 0
        cm = np.array([[0, 0], [0, 0]])

    precision = _divisao_segura(tp, tp + fp)
    recall = _divisao_segura(tp, tp + fn)
    specificity = _divisao_segura(tn, tn + fp)
    f1 = _divisao_segura(2 * tp, 2 * tp + fp + fn)

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(auc) if not np.isnan(auc) else float("nan"),
        "confusion_matrix": [
            [int(cm[0, 0]), int(cm[0, 1])],
            [int(cm[1, 0]), int(cm[1, 1])],
        ],
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }
