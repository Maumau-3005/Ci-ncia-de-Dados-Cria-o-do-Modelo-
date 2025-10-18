#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avalia o modelo salvo para fumante atual (`_SMOKER3`) usando o dataset
BRFSS 2015 com a mesma preparação empregada no treinamento simplificado.

Uso básico:
  python scripts/avaliar_modelo.py \
    --dataset data/2015.csv \
    --models-dir models \
    [--sample 50000]
"""

from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metricas_modelo import metricas_classificacao, probabilidades_ou_scores

# Importa utilitários do script de treino (no diretório raiz)
from treinar_tabagismo_brfss_2015 import (
    carregar_dataset,
    preparar_dados,
)


def avaliar_modelo(modelo, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
    mask = y_true.notna()
    X_eval = X.loc[mask]
    y_eval = y_true.loc[mask].astype(int)

    if len(X_eval) == 0:
        raise ValueError("Sem exemplos válidos para avaliação (y ausente).")

    y_pred = modelo.predict(X_eval)

    try:
        scores = probabilidades_ou_scores(modelo, X_eval)
    except Exception:
        scores = None

    metrics_summary = metricas_classificacao(
        y_eval, y_pred, scores=scores, positive_label=1
    )
    metrics_summary["n"] = int(len(X_eval))
    return metrics_summary


def formata_pct(x: float | None) -> str:
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
    df = carregar_dataset(args.dataset)
    if args.sample is not None and 0 < args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)

    X, y = preparar_dados(df)
    if len(X) == 0:
        raise ValueError("Nenhum registro válido restante após limpeza e remoção de NaNs.")

    total_fumantes = int((y == 1).sum())
    total_nao_fumantes = int((y == 0).sum())
    print(
        f"Registros avaliados após limpeza: {len(X)} "
        f"(fumantes={total_fumantes}, não fumantes={total_nao_fumantes})."
    )

    caminho_modelo = os.path.join(args.models_dir, "smoker_current_model.joblib")
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}. Treine antes de avaliar."
        )

    modelo_salvo = load(caminho_modelo)
    resultado = avaliar_modelo(modelo_salvo, X, y)

    print("\n== FUMANTE ATUAL ==")
    print(f"Amostras avaliadas: {resultado['n']}")
    print(
        f"Accuracy={formata_pct(resultado['accuracy'])} | ROC-AUC={formata_pct(resultado['roc_auc'])} | "
        f"Precisão={resultado['precision']:.3f} | Recall={resultado['recall']:.3f} | "
        f"Especificidade={resultado['specificity']:.3f} | F1={resultado['f1']:.3f}"
    )
    print("Matriz de confusão [real 0/1 x prev 0/1]:")
    cm = resultado["confusion_matrix"]
    print(f"  [TN={cm[0][0]:>6}  FP={cm[0][1]:>6}]")
    print(f"  [FN={cm[1][0]:>6}  TP={cm[1][1]:>6}]")


if __name__ == "__main__":
    main()
