#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avalia os modelos salvos (binge e fumante atual) de forma conjunta
usando o dataset BRFSS 2015. Reaproveita as mesmas rotinas de limpeza
e derivação de alvos do script de treinamento.

Uso básico:
  python scripts/evaluate_models.py \
    --dataset data/2015.csv \
    --models-dir models \
    [--sample 50000] [--only binge|smoke|both]
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
    derive_targets,
    select_features,
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
    parser.add_argument("--only", choices=["binge", "smoke", "both"], default="both")
    args = parser.parse_args()

    # Carregar dataset
    df = read_dataset(args.dataset)
    if args.sample is not None and 0 < args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    required_cols = sorted(set(SELECTED_FEATURES + ["_RFDRHV5", "_SMOKER3"]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Dataset não contém colunas necessárias: " + ", ".join(missing))

    df = df.loc[:, required_cols]

    # Limpeza equivalente ao treino
    df = apply_manual_cleaning(df)
    missing_map = build_missing_map(df)
    df = apply_missing_map(df, missing_map)

    # Alvos e features
    y_binge, y_smoke, leakage_cols = derive_targets(df)
    cat_cols, num_cols = select_features(df, leakage_cols)
    feat_cols = cat_cols + num_cols
    if not feat_cols:
        raise RuntimeError("Nenhuma feature disponível após filtragem.")

    X = df.loc[:, feat_cols].copy()

    # Caminhos dos modelos
    binge_path = os.path.join(args.models_dir, "alcohol_binge_model.joblib")
    smoke_path = os.path.join(args.models_dir, "smoker_current_model.joblib")

    def print_block(title: str, res: Dict[str, Any]) -> None:
        print(f"\n== {title} ==")
        print(f"Amostras avaliadas: {res['n']}")
        print(
            f"Accuracy={as_pct(res['accuracy'])} | ROC-AUC={as_pct(res['roc_auc'])} | "
            f"Precisão={res['precision']:.3f} | Recall={res['recall']:.3f} | "
            f"Especificidade={res['specificity']:.3f} | F1={res['f1']:.3f}"
        )
        print("Matriz de confusão [real 0/1 x prev 0/1]:")
        cm = res["confusion_matrix"]
        print(f"  [TN={cm[0][0]:>6}  FP={cm[0][1]:>6}]")
        print(f"  [FN={cm[1][0]:>6}  TP={cm[1][1]:>6}]")

    # Binge
    if args.only in ("binge", "both"):
        if not os.path.exists(binge_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {binge_path}. Treine antes de avaliar."
            )
        binge_pipe = load(binge_path)
        binge_res = evaluate(binge_pipe, X, y_binge)
        print_block("BINGE", binge_res)

    # Fumante atual
    if args.only in ("smoke", "both"):
        if not os.path.exists(smoke_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {smoke_path}. Treine antes de avaliar."
            )
        smoke_pipe = load(smoke_path)
        smoke_res = evaluate(smoke_pipe, X, y_smoke)
        print_block("FUMANTE ATUAL", smoke_res)


if __name__ == "__main__":
    main()
