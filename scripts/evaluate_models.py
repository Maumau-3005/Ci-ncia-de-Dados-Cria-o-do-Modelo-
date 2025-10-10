#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avalia o modelo salvo para fumante atual (`_SMOKER3`) usando o dataset
BRFSS 2015. Reaproveita as mesmas rotinas de limpeza e derivação do
script de treinamento.

Uso básico:
  python scripts/evaluate_models.py \
    --dataset data/2015.csv \
    --models-dir models \
    [--sample 50000]
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import load
import sys
from pathlib import Path

from model_metrics import classification_metrics, proba_or_score

# Garante que o diretório raiz (pai de scripts/) esteja no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importa utilitários do script de treino (no diretório raiz)
from train_brfss_2015 import (
    SELECTED_FEATURES,
    read_dataset,
    apply_manual_cleaning,
    build_missing_map,
    apply_missing_map,
    prepare_smoker_dataset,
    IQRClipper,
)

# Compatibilidade com artefatos treinados quando o script de treino foi executado como __main__
# e a classe IQRClipper foi serializada com esse qualname.
_main_mod = sys.modules.get("__main__")
if _main_mod is not None and not hasattr(_main_mod, "IQRClipper"):
    setattr(_main_mod, "IQRClipper", IQRClipper)


def evaluate(pipe, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
    mask = y_true.notna()
    X_eval = X.loc[mask]
    y_eval = y_true.loc[mask].astype(int)

    if len(X_eval) == 0:
        raise ValueError("Sem exemplos válidos para avaliação (y ausente).")

    y_pred = pipe.predict(X_eval)

    try:
        scores = proba_or_score(pipe, X_eval)
    except Exception:
        scores = None

    metrics_summary = classification_metrics(
        y_eval, y_pred, scores=scores, positive_label=1
    )
    metrics_summary["n"] = int(len(X_eval))
    return metrics_summary


def as_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x*100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join("data", "2015.csv"))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    # Carregar dataset
    df = read_dataset(args.dataset)
    if args.sample is not None and 0 < args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    required_cols = sorted(set(SELECTED_FEATURES + ["_SMOKER3"]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Dataset não contém colunas necessárias: " + ", ".join(missing))

    df = df.loc[:, required_cols]

    # Limpeza equivalente ao treino
    df = apply_manual_cleaning(df)
    missing_map = build_missing_map(df)
    df = apply_missing_map(df, missing_map)

    # Features e alvo focados em tabagismo
    X, y, _, _ = prepare_smoker_dataset(df)
    if len(X) == 0:
        raise ValueError("Nenhum registro válido restante após limpeza e remoção de NaNs.")

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    print(
        f"Registros avaliados após limpeza: {len(X)} "
        f"(fumantes={positives}, não fumantes={negatives})."
    )

    smoke_path = os.path.join(args.models_dir, "smoker_current_model.joblib")
    if not os.path.exists(smoke_path):
        raise FileNotFoundError(
            f"Modelo não encontrado: {smoke_path}. Treine antes de avaliar."
        )

    smoke_pipe = load(smoke_path)
    smoke_res = evaluate(smoke_pipe, X, y)

    print("\n== FUMANTE ATUAL ==")
    print(f"Amostras avaliadas: {smoke_res['n']}")
    print(
        f"Accuracy={as_pct(smoke_res['accuracy'])} | ROC-AUC={as_pct(smoke_res['roc_auc'])} | "
        f"Precisão={smoke_res['precision']:.3f} | Recall={smoke_res['recall']:.3f} | "
        f"Especificidade={smoke_res['specificity']:.3f} | F1={smoke_res['f1']:.3f}"
    )
    print("Matriz de confusão [real 0/1 x prev 0/1]:")
    cm = smoke_res["confusion_matrix"]
    print(f"  [TN={cm[0][0]:>6}  FP={cm[0][1]:>6}]")
    print(f"  [FN={cm[1][0]:>6}  TP={cm[1][1]:>6}]")


if __name__ == "__main__":
    main()
