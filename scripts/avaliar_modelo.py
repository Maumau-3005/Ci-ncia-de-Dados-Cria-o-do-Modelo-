#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Avalia o modelo salvo para fumante atual (`_SMOKER3`) usando o dataset
BRFSS 2015. Reaproveita as mesmas rotinas de limpeza e derivação do
script de treinamento.

Uso básico:
  python scripts/avaliar_modelo.py \
    --dataset data/2015.csv \
    --models-dir models \
    [--sample 50000]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from joblib import load

# Garante que o diretório raiz (pai de scripts/) esteja no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metricas_modelo import (
    formatar_percentual,
    metricas_classificacao,
    probabilidades_ou_scores,
)

# Importa utilitários do script de treino (no diretório raiz)
from treinar_tabagismo_brfss_2015 import (
    COLUNAS_SELECIONADAS,
    carregar_dataset,
    aplicar_limpeza_manual,
    construir_mapa_ausentes,
    aplicar_mapa_ausentes,
    preparar_dados_fumante,
    LimitadorIQR,
)

# Compatibilidade com artefatos treinados quando o script de treino foi executado como __main__
# e a classe LimitadorIQR foi serializada com esse qualname.
_main_mod = sys.modules.get("__main__")
if _main_mod is not None:
    if not hasattr(_main_mod, "LimitadorIQR"):
        setattr(_main_mod, "LimitadorIQR", LimitadorIQR)
    if not hasattr(_main_mod, "IQRClipper"):
        setattr(_main_mod, "IQRClipper", LimitadorIQR)


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

    colunas_necessarias = sorted(set(COLUNAS_SELECIONADAS + ["_SMOKER3"]))
    ausentes = [c for c in colunas_necessarias if c not in df.columns]
    if ausentes:
        raise ValueError("Dataset não contém colunas necessárias: " + ", ".join(ausentes))

    df = df.loc[:, colunas_necessarias]

    # Limpeza equivalente ao treino
    df = aplicar_limpeza_manual(df)
    mapa_ausentes = construir_mapa_ausentes(df)
    df = aplicar_mapa_ausentes(df, mapa_ausentes)

    # Features e alvo focados em tabagismo
    X, y, _, _ = preparar_dados_fumante(df)
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
    fpr = resultado.get("false_positive_rate", float("nan"))
    print(
        f"Accuracy={formatar_percentual(resultado['accuracy'])} | ROC-AUC={formatar_percentual(resultado['roc_auc'])} | "
        f"Taxa de falsos positivos (FPR)={fpr:.3f}"
    )
    print("Matriz de confusão [real 0/1 x prev 0/1]:")
    cm = resultado["confusion_matrix"]
    print(f"  [TN={cm[0][0]:>6}  FP={cm[0][1]:>6}]")
    print(f"  [FN={cm[1][0]:>6}  TP={cm[1][1]:>6}]")


if __name__ == "__main__":
    main()
