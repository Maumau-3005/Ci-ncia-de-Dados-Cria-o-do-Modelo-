#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline enxuto para treinar uma árvore de decisão que identifica fumantes
atuais (`_SMOKER3`) no BRFSS 2015.

O foco é manter o fluxo simples:

1. Seleciona apenas as colunas relevantes.
2. Converte códigos clássicos de não resposta em valores ausentes.
3. Prepara alvo binário (1 = fumante, 0 = não fumante) e remove linhas com NaNs.
4. Treina uma árvore de decisão com pré-processamento mínimo.
5. Reporta métricas essenciais e salva o pipeline treinado.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from metricas_modelo import metricas_classificacao, probabilidades_ou_scores


RANDOM_STATE = 42

# Colunas escolhidas manualmente após exploração do dataset.
COLUNAS_SELECIONADAS = [
    "_AGE_G",
    "EDUCA",
    "MARITAL",
    "INCOME2",
    "GENHLTH",
    "MENTHLTH",
    "PHYSHLTH",
    "EXERANY2",
    "SEX",
    "CHILDREN",
    "HLTHPLN1",
    "_TOTINDA",
    "DIABETE3",
    "ASTHMA3",
]

COLUNAS_NUMERICAS = ["MENTHLTH", "PHYSHLTH", "CHILDREN"]
COLUNAS_CATEGORICAS = [c for c in COLUNAS_SELECIONADAS if c not in COLUNAS_NUMERICAS]

# Códigos padrão do BRFSS que representam não resposta ou valores inválidos.
CODIGOS_AUSENTES: Tuple[int, ...] = (7, 9, 77, 88, 99, 555, 777, 888, 999)


def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    path = Path(caminho_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Informe o caminho correto para o BRFSS 2015."
        )
    return pd.read_csv(path, low_memory=False)


def substituir_codigos_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    to_replace = list(CODIGOS_AUSENTES) + [str(c) for c in CODIGOS_AUSENTES]
    df_limpo = df.replace(to_replace, np.nan)
    return df_limpo


def construir_alvo(series: pd.Series) -> pd.Series:
    alvo = series.map({1: 1, 2: 1, 3: 0, 4: 0}).astype(float)
    return alvo


def preparar_dados(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "_SMOKER3" not in df.columns:
        raise ValueError("Coluna `_SMOKER3` ausente: não é possível construir o alvo.")

    colunas_faltantes = [c for c in COLUNAS_SELECIONADAS if c not in df.columns]
    if colunas_faltantes:
        raise ValueError(
            "As colunas necessárias não foram encontradas no dataset: "
            + ", ".join(sorted(colunas_faltantes))
        )

    df_relevante = df[COLUNAS_SELECIONADAS + ["_SMOKER3"]].copy()
    df_relevante = substituir_codigos_ausentes(df_relevante)

    alvo = construir_alvo(df_relevante.pop("_SMOKER3"))
    dataset = df_relevante.join(alvo.rename("target")).dropna()

    X = dataset[COLUNAS_SELECIONADAS].copy()
    y = dataset["target"].astype(int)
    return X, y


def montar_preprocessador() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # Compatibilidade com scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessador = ColumnTransformer(
        transformers=[
            ("categoricas", cat_pipeline, COLUNAS_CATEGORICAS),
            ("numericas", num_pipeline, COLUNAS_NUMERICAS),
        ]
    )
    return preprocessador


def montar_modelo(max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    preprocessador = montar_preprocessador()
    arvore = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    modelo = Pipeline(
        steps=[
            ("prep", preprocessador),
            ("clf", arvore),
        ]
    )
    return modelo


def avaliar(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predicoes = modelo.predict(X_test)
    try:
        scores = probabilidades_ou_scores(modelo, X_test)
    except Exception:
        scores = None
    return metricas_classificacao(y_test, predicoes, scores=scores, positive_label=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina uma árvore de decisão simples para o BRFSS 2015.")
    parser.add_argument("--dataset", default="data/2015.csv", help="Caminho para o CSV do BRFSS 2015.")
    parser.add_argument("--sample", type=int, default=None, help="Quantidade de linhas para amostragem rápida.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção do conjunto de teste (0-1).")
    parser.add_argument("--max-depth", type=int, default=None, help="Profundidade máxima da árvore (None = livre).")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=25,
        help="Mínimo de amostras em cada folha da árvore.",
    )
    parser.add_argument(
        "--output",
        default="models/smoker_current_model.joblib",
        help="Caminho para salvar o pipeline treinado.",
    )
    args = parser.parse_args()

    df = carregar_dataset(args.dataset)
    if args.sample is not None and 0 < args.sample < len(df):
        df = df.sample(n=args.sample, random_state=RANDOM_STATE)

    X, y = preparar_dados(df)
    if len(X) == 0:
        raise ValueError("Nenhum registro válido após a limpeza; revise o dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    modelo = montar_modelo(args.max_depth, args.min_samples_leaf)
    modelo.fit(X_train, y_train)

    resultados = avaliar(modelo, X_test, y_test)

    print("== RESULTADOS DECISION TREE ==")
    print(f"Amostras de teste: {len(y_test)}")
    print(
        "Accuracy={acc:.4f} | ROC-AUC={auc:.4f} | Precisão={prec:.4f} | "
        "Recall={rec:.4f} | Especificidade={spec:.4f} | F1={f1:.4f}".format(
            acc=resultados["accuracy"],
            auc=resultados["roc_auc"],
            prec=resultados["precision"],
            rec=resultados["recall"],
            spec=resultados["specificity"],
            f1=resultados["f1"],
        )
    )
    print("Matriz de confusão [real 0/1 x prev 0/1]:")
    cm = resultados["confusion_matrix"]
    print(f"  [TN={cm[0][0]:>6}  FP={cm[0][1]:>6}]")
    print(f"  [FN={cm[1][0]:>6}  TP={cm[1][1]:>6}]")

    caminho_modelo = Path(args.output)
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    dump(modelo, caminho_modelo)
    print(f"\nPipeline salvo em: {caminho_modelo}")


if __name__ == "__main__":
    main()
