#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: treinar_tabagismo_brfss_2015.py

Objetivo: Treinar modelos para o alvo `_SMOKER3` (fumante atual) no BRFSS 2015
com pipeline de pré-processamento e salvar o artefato resultante. O script imprime
mensagens didáticas no formato "PASSO X: ..." ao longo do processo.

Observação importante: códigos de não resposta variam por coluna. Este
script aplica regras específicas por coluna (limpeza manual) e uma heurística
segura por coluna baseada em códigos clássicos presentes em cada coluna. Não há
dependência de JSON externo.
"""

from __future__ import annotations

import os
import argparse
import warnings
from typing import Dict, Set, Tuple, Any, List, Iterable

import numpy as np
import pandas as pd
from joblib import dump

from sklearn import metrics
from sklearn import model_selection as skms
from sklearn import compose
from sklearn import preprocessing as skprep
from sklearn import impute
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance

from metricas_modelo import metricas_classificacao, probabilidades_ou_scores
# ==========================
# Utilitário de logging
# ==========================
_STEP = 0


def log(msg: str) -> None:
    global _STEP
    _STEP += 1
    print(f"PASSO {_STEP}: {msg}")


warnings.simplefilter("ignore")
SEMENTE_ALEATORIA = 42


def _fmt_pct(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "--"
    return f"{value*100:.2f}%"


def _print_confusion_matrix(matrix: List[List[int]]) -> None:
    if not matrix or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        return
    print("    Matriz de confusão (linhas=real 0/1, colunas=prev 0/1)")
    print(f"      [TN={matrix[0][0]:>6}  FP={matrix[0][1]:>6}]")
    print(f"      [FN={matrix[1][0]:>6}  TP={matrix[1][1]:>6}]")


# Lista de colunas selecionadas manualmente para o treinamento
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

# Subconjunto tratado como numérico contínuo dentro das features selecionadas
COLUNAS_NUMERICAS = {"MENTHLTH", "PHYSHLTH", "CHILDREN"}


# ==========================
# Funções auxiliares
# ==========================
def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_csv}. Coloque o BRFSS 2015 em 'data/2015.csv'."
        )
    return pd.read_csv(caminho_csv, low_memory=False)


def construir_mapa_ausentes(df: pd.DataFrame) -> Dict[str, Set[Any]]:
    """
    Constrói um mapa coluna->conjunto de códigos a serem tratados como ausentes (NaN).

    Heurística segura por coluna: adiciona aos ausentes somente os códigos
    clássicos que existirem naquela coluna (e.g. {7,9,77,88,99,555,777,888,999}).
    """
    mapa_ausentes: Dict[str, Set[Any]] = {c: set() for c in df.columns}

    classic_codes = {7, 9, 77, 88, 99, 555, 777, 888, 999}
    classic_str = {str(x) for x in classic_codes}

    for col in df.columns:
        try:
            uniques = set(pd.Series(df[col]).dropna().unique().tolist())
        except Exception:
            uniques = set()

        for code in classic_codes:
            if code in uniques:
                mapa_ausentes[col].add(code)
        for code in classic_str:
            if code in uniques:
                mapa_ausentes[col].add(code)

    return mapa_ausentes


def aplicar_mapa_ausentes(df: pd.DataFrame, mapa_ausentes: Dict[str, Set[Any]]) -> pd.DataFrame:
    for col, codes in mapa_ausentes.items():
        if not codes:
            continue
        try:
            df[col] = df[col].replace(list(codes), np.nan)
        except Exception:
            # Em último caso, converte para string e tenta novamente
            try:
                df[col] = df[col].astype(str).replace({str(c): np.nan for c in codes})
            except Exception:
                # Melhor não falhar – deixa a coluna como está
                pass
    return df


def aplicar_limpeza_manual(df: pd.DataFrame) -> pd.DataFrame:
    # Limpeza manual conforme análise dos metadados e documentação
    replacements: Dict[str, Dict[Any, Any]] = {
        "_RFDRHV5": {9: np.nan},
        "_SMOKER3": {9: np.nan},
        "GENHLTH": {9: np.nan, 7: np.nan},
        "PHYSHLTH": {99: np.nan, 77: np.nan, 88: 0.0},
        "MENTHLTH": {99: np.nan, 77: np.nan, 88: 0.0},
        "CHILDREN": {88: 0.0, 99: np.nan},
        "INCOME2": {99: np.nan, 77: 0.0},
        "EDUCA": {9: np.nan},
        "HLTHPLN1": {9: np.nan, 7: np.nan, 1: 0.0, 2: 1.0},
        "MARITAL": {9: np.nan},
        "_TOTINDA": {9: np.nan},
        "DIABETE3": {9: np.nan, 7: np.nan},
        "ASTHMA3": {9: np.nan, 7: np.nan},
    }

    # Trabalhar sobre uma cópia para não alterar o DataFrame original inadvertidamente
    df_limpo = df.copy()

    for col, mapping in replacements.items():
        if col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].replace(mapping)

    return df_limpo


class LimitadorIQR(TransformerMixin, BaseEstimator):
    """Winsoriza valores fora do intervalo IQR multiplicado por um fator."""

    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):  # type: ignore[override]
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_features = X_arr.shape[1]
        self.lower_ = np.full(n_features, np.nan)
        self.upper_ = np.full(n_features, np.nan)

        for idx in range(n_features):
            col = X_arr[:, idx]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue

            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1

            if iqr == 0:
                lower = q1
                upper = q3
            else:
                lower = q1 - self.multiplier * iqr
                upper = q3 + self.multiplier * iqr

            self.lower_[idx] = lower
            self.upper_[idx] = upper

        return self

    def transform(self, X):  # type: ignore[override]
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("LimitadorIQR deve ser ajustado antes do uso.")
        X_arr = np.asarray(X, dtype=float)
        reshape_needed = False
        if X_arr.ndim == 1:
            reshape_needed = True
            X_arr = X_arr.reshape(-1, 1)

        clipped = X_arr.copy()
        for idx in range(clipped.shape[1]):
            lower = self.lower_[idx]
            upper = self.upper_[idx]
            if np.isnan(lower) or np.isnan(upper):
                continue
            clipped[:, idx] = np.clip(clipped[:, idx], lower, upper)

        if reshape_needed:
            clipped = clipped.reshape(-1)

        return clipped


# Alias mantido para compatibilidade com pipelines antigos serializados
IQRClipper = LimitadorIQR


def criar_amostra_balanceada(
    dados: pd.DataFrame,
    alvo: pd.Series,
    rotulo_positivo: float = 1.0,
    rotulo_negativo: float = 0.0,
    registros_por_classe: int = 25000,
    semente: int = SEMENTE_ALEATORIA,
    proporcao_holdout: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series, int, np.ndarray]:
    """Seleciona amostra balanceada 50/50 entre classes positiva e negativa."""

    mascara_valida = alvo.isin([rotulo_positivo, rotulo_negativo])
    alvo_valido = alvo.loc[mascara_valida]

    indices_pos = alvo_valido[alvo_valido == rotulo_positivo].index
    indices_neg = alvo_valido[alvo_valido == rotulo_negativo].index

    proporcao_holdout = min(max(proporcao_holdout, 0.0), 0.5)
    holdout_pos = max(1, int(len(indices_pos) * proporcao_holdout)) if len(indices_pos) > 0 else 0
    holdout_neg = max(1, int(len(indices_neg) * proporcao_holdout)) if len(indices_neg) > 0 else 0

    disponiveis_pos = len(indices_pos) - holdout_pos
    disponiveis_neg = len(indices_neg) - holdout_neg

    registros_efetivos = min(registros_por_classe, disponiveis_pos, disponiveis_neg)

    if registros_efetivos <= 0:
        raise ValueError(
            "Não há amostras suficientes para balanceamento: "
            f"positivos disponíveis={len(indices_pos)}, negativos disponíveis={len(indices_neg)}, "
            f"necessário por classe={registros_por_classe} após reservar {holdout_pos} positivos e {holdout_neg} negativos para teste."
        )

    rng = np.random.default_rng(semente)
    amostra_pos = rng.choice(indices_pos, size=registros_efetivos, replace=False)
    amostra_neg = rng.choice(indices_neg, size=registros_efetivos, replace=False)

    indices_selecionados = np.concatenate([amostra_pos, amostra_neg])
    rng.shuffle(indices_selecionados)

    return (
        dados.loc[indices_selecionados].copy(),
        alvo.loc[indices_selecionados].copy(),
        registros_efetivos,
        indices_selecionados,
    )


def construir_alvo_fumante(df: pd.DataFrame) -> pd.Series:
    """
    Constrói o alvo binário de fumante atual (1 = fumante, 0 = não fumante)
    com base na coluna `_SMOKER3`. Outros códigos são convertidos para NaN.
    """
    if "_SMOKER3" not in df.columns:
        raise ValueError(
            "Não foi possível construir o alvo: coluna obrigatória `_SMOKER3` ausente."
        )

    s_smoke = pd.to_numeric(df["_SMOKER3"], errors="coerce")
    alvo_fumante = pd.Series(
        np.where(s_smoke.isin([1, 2]), 1.0, np.where(s_smoke.isin([3, 4]), 0.0, np.nan)),
        index=s_smoke.index,
    )
    return alvo_fumante.astype(float)


def selecionar_variaveis(
    df: pd.DataFrame, leakage_cols: Iterable[str] | None = None
) -> Tuple[List[str], List[str]]:
    """Seleciona as colunas definidas manualmente e separa tipos para o pipeline."""

    available = [c for c in COLUNAS_SELECIONADAS if c in df.columns]

    missing = [c for c in COLUNAS_SELECIONADAS if c not in df.columns]
    if missing:
        warnings.warn(
            "As seguintes colunas selecionadas não estão presentes no dataset e serão ignoradas: "
            + ", ".join(missing)
        )

    features = available.copy()

    # Remover potenciais vazamentos (colunas usadas na derivação do alvo corrente)
    for leak in leakage_cols or ():
        if leak in features:
            features.remove(leak)

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in features:
        if c in COLUNAS_NUMERICAS:
            num_cols.append(c)
        else:
            # Demais tratadas como categóricas (incluindo codificadas numericamente)
            cat_cols.append(c)

    return cat_cols, num_cols


def preparar_dados_fumante(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepara o conjunto de dados para treinamento/avaliação do alvo fumante.

    - Seleciona features válidas
    - Constrói o alvo `_SMOKER3` (binário)
    - Remove linhas com qualquer NaN em features ou target
    """
    alvo_fumante = construir_alvo_fumante(df)
    colunas_categoricas, colunas_numericas = selecionar_variaveis(df, leakage_cols=["_SMOKER3"])
    colunas_caracteristicas = colunas_categoricas + colunas_numericas

    if not colunas_caracteristicas:
        raise RuntimeError("Nenhuma feature disponível após filtragem.")

    X = df.loc[:, colunas_caracteristicas].copy()
    dataset = X.join(alvo_fumante.rename("target"))
    dataset = dataset.dropna()

    y_limpo = dataset.pop("target").astype(int)
    return dataset, y_limpo, colunas_categoricas, colunas_numericas


def montar_preprocessamento(colunas_categoricas: List[str], colunas_numericas: List[str]) -> compose.ColumnTransformer:
    # Compatibilidade com versões do scikit-learn: 'sparse' foi substituído por 'sparse_output'.
    try:
        ohe = skprep.OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback para versões mais antigas
        ohe = skprep.OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", impute.SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    num_pipe = Pipeline(
        steps=[
            ("imputer", impute.SimpleImputer(strategy="median")),
            ("clipper", LimitadorIQR()),
            ("scaler", skprep.StandardScaler()),
        ]
    )

    preprocessador = compose.ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, colunas_categoricas),
            ("num", num_pipe, colunas_numericas),
        ]
    )
    return preprocessador


def obter_modelos(modo: str = "full") -> Dict[str, Any]:
    modo = (modo or "full").lower()
    if modo == "quick":
        return {
            "RF": ensemble.RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEMENTE_ALEATORIA,
                n_jobs=-1,
            ),
            "GB": ensemble.GradientBoostingClassifier(random_state=SEMENTE_ALEATORIA),
        }
    return {
        "RF": ensemble.RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEMENTE_ALEATORIA,
            n_jobs=-1,
        ),
        "GB": ensemble.GradientBoostingClassifier(random_state=SEMENTE_ALEATORIA),
    }


def treinar_e_avaliar(
    dados: pd.DataFrame,
    alvo: pd.Series,
    preprocessador: compose.ColumnTransformer,
    colunas_categoricas: List[str],
    colunas_numericas: List[str],
    nome_alvo: str,
    modelos: Dict[str, Any],
    tamanho_teste: float | int = 0.15,
    fracao_validacao_no_treino: float = 15 / 85,
    teste_externo_X: pd.DataFrame | None = None,
    teste_externo_y: pd.Series | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Tuple[str, float, Pipeline]]:
    """
    - Filtra registros com alvo ausente.
    - Realiza splits estratificados conforme os parâmetros informados.
    - Treina e avalia os modelos, retornando o resumo completo e o melhor pipeline
      (Accuracy de teste interno ou externo reportado após o ajuste final).
    """
    # Filtrar apenas registros com y válido
    colunas_modelo = colunas_categoricas + colunas_numericas
    mask = alvo.notna()
    dados_validos = dados.loc[mask, colunas_modelo].copy()
    alvo_valido = alvo.loc[mask].astype(int)
    removed_rows = int((~mask).sum())

    log(f"Linhas com target ausente removidas para {nome_alvo}: {removed_rows}.")

    X_train_val, X_test, y_train_val, y_test = skms.train_test_split(
        dados_validos, alvo_valido, test_size=tamanho_teste, stratify=alvo_valido, random_state=SEMENTE_ALEATORIA
    )

    if not 0 < fracao_validacao_no_treino < 1:
        raise ValueError("fracao_validacao_no_treino deve estar entre 0 e 1.")

    X_train, X_val, y_train, y_val = skms.train_test_split(
        X_train_val,
        y_train_val,
        test_size=fracao_validacao_no_treino,
        stratify=y_train_val,
        random_state=SEMENTE_ALEATORIA,
    )

    log(
        f"Split estratificado definido para {nome_alvo}: "
        f"treino={len(X_train)}, val={len(X_val)}, teste={len(X_test)}."
    )

    # Resultados
    resultados_modelo: Dict[str, Dict[str, float]] = {}
    melhor_modelo: str | None = None
    melhor_acc_validacao = -np.inf

    for nome_modelo, estimador in modelos.items():
        pipe = Pipeline(
            steps=[("prep", clone(preprocessador)), ("clf", clone(estimador))]
        )

        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_acc = metrics.accuracy_score(y_val, val_pred)

        resultados_modelo[nome_modelo] = {
            "val_acc": float(val_acc),
            "test_acc": np.nan,
            "test_auc": np.nan,
        }

        print(
            f"Modelo={nome_modelo} | Alvo={nome_alvo.upper()} | Val Accuracy={val_acc*100:.2f}%"
        )

        if val_acc > melhor_acc_validacao:
            melhor_acc_validacao = val_acc
            melhor_modelo = nome_modelo

    if melhor_modelo is None:
        raise RuntimeError("Não foi possível selecionar o melhor modelo.")

    log(
        f"Reajustando o modelo {melhor_modelo} com dados de treino+validação para {nome_alvo}."
    )
    pipeline_final = Pipeline(
        steps=[("prep", clone(preprocessador)), ("clf", clone(modelos[melhor_modelo]))]
    )
    pipeline_final.fit(X_train_val, y_train_val)

    X_avaliacao = X_test
    y_avaliacao = y_test
    descricao_avaliacao = "teste interno estratificado"
    if teste_externo_X is not None and teste_externo_y is not None:
        X_avaliacao = teste_externo_X
        y_avaliacao = teste_externo_y
        descricao_avaliacao = "teste externo"
        log(
            f"Executando avaliação externa para {nome_alvo} com {len(X_avaliacao)} registros."
        )

    predicoes_teste = pipeline_final.predict(X_avaliacao)
    try:
        scores_teste = probabilidades_ou_scores(pipeline_final, X_avaliacao)
    except Exception:
        scores_teste = None

    resumo_metricas = metricas_classificacao(
        y_avaliacao, predicoes_teste, scores=scores_teste
    )

    acuracia_teste = resumo_metricas["accuracy"]
    auc_teste = resumo_metricas["roc_auc"]

    resultados_modelo[melhor_modelo]["test_acc"] = acuracia_teste
    resultados_modelo[melhor_modelo]["test_auc"] = auc_teste
    resultados_modelo[melhor_modelo]["confusion_matrix"] = resumo_metricas["confusion_matrix"]
    resultados_modelo[melhor_modelo]["precision"] = resumo_metricas["precision"]
    resultados_modelo[melhor_modelo]["recall"] = resumo_metricas["recall"]
    resultados_modelo[melhor_modelo]["specificity"] = resumo_metricas["specificity"]
    resultados_modelo[melhor_modelo]["f1"] = resumo_metricas["f1"]

    colunas_caracteristicas = colunas_modelo
    importancias_ordenadas: List[Tuple[str, float]] | None = None
    if X_avaliacao is not None and len(X_avaliacao) > 0:
        try:
            resultado_permutacao = permutation_importance(
                pipeline_final,
                X_avaliacao,
                y_avaliacao,
                n_repeats=10,
                random_state=SEMENTE_ALEATORIA,
                n_jobs=-1,
            )
            importancias = resultado_permutacao.importancias_mean
            importancias_ordenadas = sorted(
                [
                    (colunas_caracteristicas[idx], float(importancias[idx]))
                    for idx in range(len(colunas_caracteristicas))
                ],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        except Exception:
            importancias_ordenadas = None

    resultados_modelo[melhor_modelo]["feature_importances"] = importancias_ordenadas or []

    print(
        f"Modelo selecionado (validação)={melhor_modelo} | Alvo={nome_alvo.upper()} | "
        f"Val Accuracy={melhor_acc_validacao*100:.2f}% | {descricao_avaliacao.title()} Accuracy={acuracia_teste*100:.2f}% | "
        f"{descricao_avaliacao.title()} ROC-AUC={(auc_teste*100 if not np.isnan(auc_teste) else np.nan):.2f}"
    )

    return resultados_modelo, (melhor_modelo, float(acuracia_teste), pipeline_final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.path.join("data", "2015.csv"),
        help="Caminho para o CSV do BRFSS 2015.",
    )
    parser.add_argument("--quick", action="store_true", help="Executa com modelos mais leves.")
    parser.add_argument("--sample", type=int, default=None, help="Amostragem de N linhas do dataset.")
    parser.add_argument(
        "--balance-per-class",
        type=int,
        default=25000,
        help="Quantidade máxima de registros por classe após balanceamento 50/50.",
    )
    parser.add_argument(
        "--balance-holdout",
        type=float,
        default=0.1,
        help="Proporção mínima por classe reservada para avaliação externa (0-0.5).",
    )
    args = parser.parse_args()

    log("Carregando dataset BRFSS 2015…")
    df = carregar_dataset(args.dataset)
    if args.sample is not None and args.sample > 0 and args.sample < len(df):
        log(f"Amostrando {args.sample} linhas do dataset para execução rápida…")
        df = df.sample(n=args.sample, random_state=SEMENTE_ALEATORIA)

    colunas_necessarias = sorted(set(COLUNAS_SELECIONADAS + ["_SMOKER3"]))
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_faltantes:
        raise ValueError(
            "O dataset não contém as colunas necessárias para o treinamento: "
            + ", ".join(colunas_faltantes)
        )

    # Garantir que somente as colunas necessárias sejam utilizadas
    df = df.loc[:, colunas_necessarias]

    log("Aplicando limpeza manual nas colunas selecionadas…")
    df = aplicar_limpeza_manual(df)

    # Construir e aplicar mapa de ausentes (heurística por coluna)
    log("Mapeando códigos de não resposta por coluna (heurística segura)…")
    mapa_ausentes = construir_mapa_ausentes(df)
    df = aplicar_mapa_ausentes(df, mapa_ausentes)
    log("Ausentes tratados respeitando a especificidade por coluna (heurística).")

    log("Preparando dataset limpo para o alvo fumante atual…")
    X_full, y_full, cat_cols, num_cols = preparar_dados_fumante(df)
    total = len(y_full)
    total_fumantes = int((y_full == 1).sum())
    total_nao_fumantes = int((y_full == 0).sum())
    log(
        f"Após limpeza, restaram {total} registros válidos "
        f"({total_fumantes} fumantes, {total_nao_fumantes} não fumantes)."
    )

    log("Balanceando o dataset (50/50) via subamostragem estratificada…")
    X_bal, y_bal, registros_balanceados, indices_selecionados = criar_amostra_balanceada(
        X_full,
        y_full,
        rotulo_positivo=1.0,
        rotulo_negativo=0.0,
        registros_por_classe=args.balance_per_class,
        semente=SEMENTE_ALEATORIA,
        proporcao_holdout=args.balance_holdout,
    )
    log(
        f"Amostra balanceada contém {len(X_bal)} registros ({registros_balanceados} por classe)."
    )

    indices_selecionados = pd.Index(indices_selecionados)
    indices_restantes = y_full.index.difference(indices_selecionados)
    X_externo: pd.DataFrame | None = None
    y_externo: pd.Series | None = None
    if args.balance_holdout > 0 and not indices_restantes.empty:
        X_externo = X_full.loc[indices_restantes].copy()
        y_externo = y_full.loc[indices_restantes].copy()
        log(
            f"Teste externo utilizará {len(X_externo)} registros remanescentes sem sobreposição com a amostra balanceada."
        )
    else:
        log("Sem amostra remanescente suficiente para teste externo; usando apenas teste interno estratificado.")

    log(
        "Montando pipeline de pré-processamento (imputação + clipping IQR + one-hot + padronização)…"
    )
    preprocessador = montar_preprocessamento(cat_cols, num_cols)

    modelos = obter_modelos("quick" if args.quick else "full")

    log("Treinando e avaliando modelos para o alvo Fumante atual…")
    resultados, (melhor_nome, melhor_acc, melhor_pipeline) = treinar_e_avaliar(
        X_bal,
        y_bal,
        preprocessador,
        cat_cols,
        num_cols,
        nome_alvo="Fumante atual",
        modelos=modelos,
        teste_externo_X=X_externo,
        teste_externo_y=y_externo,
    )

    log("Treinamento concluído.")

    log("Salvando pipeline treinado em models/smoker_current_model.joblib…")
    os.makedirs("models", exist_ok=True)
    dump(melhor_pipeline, os.path.join("models", "smoker_current_model.joblib"))

    print("\n========== RESULTADOS ==========")

    print("\n>> FUMANTE ATUAL")
    print("  Modelos avaliados:")
    for nome_modelo in sorted(resultados.keys()):
        metrics_map = resultados[nome_modelo]
        val_txt = _fmt_pct(metrics_map.get("val_acc"))
        test_txt = _fmt_pct(metrics_map.get("test_acc"))
        auc_txt = _fmt_pct(metrics_map.get("test_auc"))
        print(f"    - {nome_modelo:<30} | Val={val_txt:<8} Test={test_txt:<8} ROC-AUC={auc_txt:<8}")

    metricas_melhor_modelo = resultados[melhor_nome]
    print(f"\n  Melhor modelo: {melhor_nome}")
    print(
        "    → Val={val} | Test={test} | ROC-AUC={auc}"
        .format(
            val=_fmt_pct(metricas_melhor_modelo.get("val_acc")),
            test=_fmt_pct(metricas_melhor_modelo.get("test_acc")),
            auc=_fmt_pct(metricas_melhor_modelo.get("test_auc")),
        )
    )

    _print_confusion_matrix(metricas_melhor_modelo.get("confusion_matrix", []))

    print(
        "    Métricas complementares: Precisão={p:.3f} | Recall={r:.3f} | "
        "Especificidade={s:.3f} | F1={f:.3f}"
        .format(
            p=metricas_melhor_modelo.get("precision", float("nan")),
            r=metricas_melhor_modelo.get("recall", float("nan")),
            s=metricas_melhor_modelo.get("specificity", float("nan")),
            f=metricas_melhor_modelo.get("f1", float("nan")),
        )
    )

    importancias_ordenadas = metricas_melhor_modelo.get("feature_importances")
    if importancias_ordenadas:
        print("\n  Principais variáveis associadas ao tabagismo (importância por permutação):")
        for rank, (variavel, importancia) in enumerate(importancias_ordenadas[:10], start=1):
            print(f"    {rank:>2}. {variavel:<30} Δ={importancia:.5f}")
    else:
        print("\n  Não foi possível calcular importâncias de features.")



if __name__ == "__main__":
    main()
