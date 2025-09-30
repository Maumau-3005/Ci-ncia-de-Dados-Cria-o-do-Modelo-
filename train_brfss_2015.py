#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: train_brfss_2015.py

Objetivo: Treinar modelos para dois alvos no BRFSS 2015 (binge drinking e fumante atual)
com pipeline de pré-processamento e salvar os artefatos. O script imprime mensagens
didáticas no formato "PASSO X: ..." ao longo do processo.

Observação importante: códigos de não resposta variam por coluna. Este
script aplica regras específicas por coluna (limpeza manual) e uma heurística
segura por coluna baseada em códigos clássicos presentes em cada coluna. Não há
dependência de JSON externo.
"""

from __future__ import annotations

import os
import argparse
import warnings
from typing import Dict, Set, Tuple, Any, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn import metrics
from sklearn import model_selection as skms
from sklearn import compose
from sklearn import preprocessing as skprep
from sklearn import impute
from sklearn import linear_model
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone

# ==========================
# Utilitário de logging
# ==========================
_STEP = 0


def log(msg: str) -> None:
    global _STEP
    _STEP += 1
    print(f"PASSO {_STEP}: {msg}")


warnings.simplefilter("ignore")
RANDOM_STATE = 42


# Lista de colunas selecionadas manualmente para o treinamento
SELECTED_FEATURES = [
    "_AGE_G",
    "EDUCA",
    "MARITAL",
    "INCOME2",
    "GENHLTH",
    "MENTHLTH",
    "PHYSHLTH",
    "EXERANY2",
    "_RFDRHV5",
    "SEX",
    "CHILDREN",
    "HLTHPLN1",
    "_TOTINDA",
    "DIABETE3",
    "ASTHMA3",
    "_SMOKER3",
]

# Subconjunto tratado como numérico contínuo dentro das features selecionadas
NUMERIC_SELECTED = {"MENTHLTH", "PHYSHLTH", "CHILDREN"}


# ==========================
# Funções auxiliares
# ==========================
def read_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}. Coloque o BRFSS 2015 em 'data/2015.csv'."
        )
    return pd.read_csv(csv_path, low_memory=False)


def build_missing_map(df: pd.DataFrame) -> Dict[str, Set[Any]]:
    """
    Constrói um mapa coluna->conjunto de códigos a serem tratados como ausentes (NaN).

    Heurística segura por coluna: adiciona aos ausentes somente os códigos
    clássicos que existirem naquela coluna (e.g. {7,9,77,88,99,555,777,888,999}).
    """
    missing_map: Dict[str, Set[Any]] = {c: set() for c in df.columns}

    classic_codes = {7, 9, 77, 88, 99, 555, 777, 888, 999}
    classic_str = {str(x) for x in classic_codes}

    for col in df.columns:
        try:
            uniques = set(pd.Series(df[col]).dropna().unique().tolist())
        except Exception:
            uniques = set()

        for code in classic_codes:
            if code in uniques:
                missing_map[col].add(code)
        for code in classic_str:
            if code in uniques:
                missing_map[col].add(code)

    return missing_map


def apply_missing_map(df: pd.DataFrame, missing_map: Dict[str, Set[Any]]) -> pd.DataFrame:
    for col, codes in missing_map.items():
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


def apply_manual_cleaning(df: pd.DataFrame) -> pd.DataFrame:

#Limpeza manual conforme análise dos metadados e PDF
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
    df_clean = df.copy()

    for col, mapping in replacements.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(mapping)

    return df_clean


class IQRClipper(TransformerMixin, BaseEstimator):
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
            raise RuntimeError("IQRClipper deve ser ajustado antes do uso.")
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
def create_balanced_sample(
    df: pd.DataFrame,
    y: pd.Series,
    positive_label: float = 1.0,
    negative_label: float = 0.0,
    per_class: int = 25000,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Seleciona amostra balanceada 50/50 entre classes positiva e negativa."""

    valid_mask = y.isin([positive_label, negative_label])
    y_valid = y.loc[valid_mask]

    pos_idx = y_valid[y_valid == positive_label].index
    neg_idx = y_valid[y_valid == negative_label].index

    actual_per_class = min(per_class, len(pos_idx), len(neg_idx))

    if actual_per_class <= 0:
        raise ValueError(
            "Não há amostras suficientes para balanceamento: "
            f"positivos disponíveis={len(pos_idx)}, negativos disponíveis={len(neg_idx)}, "
            f"necessário por classe={per_class}."
        )

    rng = np.random.default_rng(random_state)
    pos_sample = rng.choice(pos_idx, size=actual_per_class, replace=False)
    neg_sample = rng.choice(neg_idx, size=actual_per_class, replace=False)

    selected_idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(selected_idx)

    return df.loc[selected_idx].copy(), y.loc[selected_idx].copy(), actual_per_class


def derive_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[str]]:
    """
    Constrói os dois alvos a partir das colunas já presentes no dataset reduzido.

    - Binge (binário) usa `_RFDRHV5`, onde 2 representa consumo excessivo e 1,
      não consumo. O valor é transformado para 1.0 (binge) / 0.0 (não binge) e
      outros códigos viram NaN.
    - Fumante atual (binário) usa `_SMOKER3`: códigos 1/2 => 1.0 (fumantes),
      códigos 3/4 => 0.0 (não fumantes); demais => NaN.

    Retorna (y_binge, y_smoke, leakage_cols) – leakage_cols devem ser removidas
    das features para evitar vazamentos.
    """
    for required in ["_RFDRHV5", "_SMOKER3"]:
        if required not in df.columns:
            raise ValueError(
                f"Não foi possível construir os alvos: coluna obrigatória {required} ausente."
            )

    leakage_cols: List[str] = []

    s_binge = pd.to_numeric(df["_RFDRHV5"], errors="coerce")
    if not s_binge.dropna().isin({0.0, 1.0}).all():
        s_binge = pd.Series(
            np.where(s_binge == 2, 1.0, np.where(s_binge == 1, 0.0, np.nan)),
            index=s_binge.index,
        )
    y_binge = s_binge.astype(float)
    leakage_cols.append("_RFDRHV5")

    s_smoke = pd.to_numeric(df["_SMOKER3"], errors="coerce")
    y_smoke = pd.Series(
        np.where(s_smoke.isin([1, 2]), 1.0, np.where(s_smoke.isin([3, 4]), 0.0, np.nan)),
        index=s_smoke.index,
    )
    leakage_cols.append("_SMOKER3")

    return y_binge, y_smoke, leakage_cols


def select_features(df: pd.DataFrame, leakage_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Seleciona as colunas definidas manualmente e separa tipos para o pipeline."""

    available = [c for c in SELECTED_FEATURES if c in df.columns]

    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        warnings.warn(
            "As seguintes colunas selecionadas não estão presentes no dataset e serão ignoradas: "
            + ", ".join(missing)
        )

    features = available.copy()

    # Remover potenciais vazamentos (colunas usadas na derivação do alvo corrente)
    for leak in leakage_cols:
        if leak in features:
            features.remove(leak)

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in features:
        if c in NUMERIC_SELECTED:
            num_cols.append(c)
        else:
            # Demais tratadas como categóricas (incluindo codificadas numericamente)
            cat_cols.append(c)

    return cat_cols, num_cols


def build_preprocess(cat_cols: List[str], num_cols: List[str]) -> compose.ColumnTransformer:
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
            ("clipper", IQRClipper()),
            ("scaler", skprep.StandardScaler()),
        ]
    )

    preprocess = compose.ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ]
    )
    return preprocess


def get_models(mode: str = "full") -> Dict[str, Any]:
    mode = (mode or "full").lower()
    if mode == "quick":
        return {
            "LogReg": linear_model.LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            # Opção rápida: RF menor
            "RF": ensemble.RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }
    return {
        "LogReg": linear_model.LogisticRegression(
            max_iter=200, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
        ),
        "RF": ensemble.RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GB": ensemble.GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def _proba_or_score(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        if isinstance(proba, list):
            proba = proba[0]
        # Prob da classe positiva
        return proba[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        # Normalizar para 0-1 via min-max caso necessário
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    # Fallback: usar previsões binárias
    preds = clf.predict(X)
    return preds.astype(float)


def fit_and_evaluate(
    df: pd.DataFrame,
    y: pd.Series,
    preprocess: compose.ColumnTransformer,
    cat_cols: List[str],
    num_cols: List[str],
    target_name: str,
    models: Dict[str, Any],
    test_size: float | int = 0.15,
    val_fraction_within_train: float = 15 / 85,
) -> Tuple[Dict[str, Dict[str, float]], Tuple[str, float, Pipeline]]:
    """
    - Filtra linhas com y ausente
    - Split estratificado conforme parâmetros (test_size pode ser fração ou inteiro)
    - Treina e avalia modelos, devolve resultados e o melhor selecionado por Val Accuracy
      (Test Accuracy reportado apenas após ajuste em treino+val)
    """
    # Filtrar apenas registros com y válido
    feature_cols = cat_cols + num_cols
    mask = y.notna()
    df_ = df.loc[mask, feature_cols].copy()
    y_ = y.loc[mask].astype(int)
    removed_rows = int((~mask).sum())

    log(f"Linhas com target ausente removidas para {target_name}: {removed_rows}.")

    X_train_val, X_test, y_train_val, y_test = skms.train_test_split(
        df_, y_, test_size=test_size, stratify=y_, random_state=RANDOM_STATE
    )

    if not 0 < val_fraction_within_train < 1:
        raise ValueError("val_fraction_within_train deve estar entre 0 e 1.")

    X_train, X_val, y_train, y_val = skms.train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_within_train,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    log(
        f"Split estratificado definido para {target_name}: "
        f"treino={len(X_train)}, val={len(X_val)}, teste={len(X_test)}."
    )

    # Resultados
    results: Dict[str, Dict[str, float]] = {}
    best_name: str | None = None
    best_val_acc = -np.inf

    for name, model in models.items():
        pipe = Pipeline(
            steps=[("prep", clone(preprocess)), ("clf", clone(model))]
        )

        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_acc = metrics.accuracy_score(y_val, val_pred)

        results[name] = {
            "val_acc": float(val_acc),
            "test_acc": np.nan,
            "test_auc": np.nan,
        }

        print(
            f"Modelo={name} | Alvo={target_name.upper()} | Val Accuracy={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_name = name

    if best_name is None:
        raise RuntimeError("Não foi possível selecionar o melhor modelo.")

    log(
        f"Reajustando o modelo {best_name} com dados de treino+validação para {target_name}."
    )
    final_pipe = Pipeline(
        steps=[("prep", clone(preprocess)), ("clf", clone(models[best_name]))]
    )
    final_pipe.fit(X_train_val, y_train_val)

    final_test_pred = final_pipe.predict(X_test)
    final_test_acc = metrics.accuracy_score(y_test, final_test_pred)
    try:
        final_test_scores = _proba_or_score(final_pipe, X_test)
        final_test_auc = metrics.roc_auc_score(y_test, final_test_scores)
    except Exception:
        final_test_auc = np.nan

    results[best_name]["test_acc"] = float(final_test_acc)
    results[best_name]["test_auc"] = (
        float(final_test_auc) if not np.isnan(final_test_auc) else np.nan
    )

    try:
        cm = metrics.confusion_matrix(y_test, final_test_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            cm = np.array([[0, 0], [0, 0]])
    except Exception:
        tn = fp = fn = tp = 0
        cm = np.array([[0, 0], [0, 0]])

    def _safe_div(a: int, b: int) -> float:
        return float(a / b) if b else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    results[best_name]["confusion_matrix"] = [
        [int(cm[0, 0]), int(cm[0, 1])],
        [int(cm[1, 0]), int(cm[1, 1])],
    ]
    results[best_name]["precision"] = precision
    results[best_name]["recall"] = recall
    results[best_name]["specificity"] = specificity
    results[best_name]["f1"] = f1

    print(
        f"Modelo selecionado (validação)={best_name} | Alvo={target_name.upper()} | "
        f"Val Accuracy={best_val_acc*100:.2f}% | Test Accuracy (reajustado)={final_test_acc*100:.2f}% | "
        f"Test ROC-AUC (reajustado)={(final_test_auc*100 if not np.isnan(final_test_auc) else np.nan):.2f}"
    )

    return results, (best_name, float(final_test_acc), final_pipe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Executa com modelos mais leves.")
    parser.add_argument("--sample", type=int, default=None, help="Amostragem de N linhas do dataset.")
    parser.add_argument(
        "--balanced-binge",
        action="store_true",
        help="Cria amostra balanceada (50/50) para o alvo Binge antes do treinamento.",
    )
    parser.add_argument(
        "--balanced-binge-size",
        type=int,
        default=25000,
        help="Quantidade de amostras por classe (positiva/negativa) para balancear Binge.",
    )
    parser.add_argument(
        "--balanced-binge-train",
        type=int,
        default=40000,
        help="Tamanho desejado do conjunto de treino (treino+validação) após balanceamento de Binge.",
    )
    parser.add_argument(
        "--balanced-binge-val-frac",
        type=float,
        default=0.2,
        help="Fração do conjunto de treino (após balancear Binge) destinada à validação.",
    )
    args = parser.parse_args()
    # Carregar dados
    log("Carregando dataset BRFSS 2015…")
    csv_path = os.path.join("data", "2015.csv")
    required_cols = sorted(set(SELECTED_FEATURES + ["_RFDRHV5", "_SMOKER3"]))
    df = read_dataset(csv_path)
    if args.sample is not None and args.sample > 0 and args.sample < len(df):
        log(f"Amostrando {args.sample} linhas do dataset para execução rápida…")
        df = df.sample(n=args.sample, random_state=RANDOM_STATE)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "O dataset não contém as colunas necessárias para o treinamento: "
            + ", ".join(missing_cols)
        )

    # Garantir que somente as colunas necessárias sejam utilizadas
    df = df.loc[:, required_cols]

    log("Aplicando limpeza manual nas colunas selecionadas…")
    df = apply_manual_cleaning(df)

    # Construir e aplicar mapa de ausentes (heurística por coluna)
    log("Mapeando códigos de não resposta por coluna (heurística segura)…")
    missing_map = build_missing_map(df)
    df = apply_missing_map(df, missing_map)
    log("Ausentes tratados respeitando a especificidade por coluna (heurística).")

    # Construção dos alvos
    log("Construindo os alvos (Binge e Fumante atual)…")
    y_binge, y_smoke, leakage_cols = derive_targets(df)

    # Seleção de features
    log("Selecionando features disponíveis e separando tipos…")
    cat_cols, num_cols = select_features(df, leakage_cols)
    if not cat_cols and not num_cols:
        raise RuntimeError("Nenhuma feature disponível após filtragem.")

    # Pré-processamento
    log(
        "Montando pipeline de pré-processamento (imputação + clipping IQR + one-hot + padronização)…"
    )
    preprocess = build_preprocess(cat_cols, num_cols)

    # Preparar modelos
    models = get_models("quick" if args.quick else "full")

    # Treinamento e avaliação por alvo
    binge_df = df
    binge_y = y_binge
    binge_test_size: float | int = 0.15
    binge_val_fraction = 15 / 85

    if args.balanced_binge:
        per_class = args.balanced_binge_size
        total_requested = per_class * 2
        log(
            "Gerando amostra balanceada para Binge (" \
            f"{per_class} positivos e {per_class} negativos; total={total_requested})…"
        )
        binge_df, binge_y, actual_per_class = create_balanced_sample(
            df,
            y_binge,
            positive_label=1.0,
            negative_label=0.0,
            per_class=per_class,
            random_state=RANDOM_STATE,
        )

        total_selected = actual_per_class * 2
        if actual_per_class < per_class:
            log(
                f"Aviso: somente {actual_per_class} amostras por classe disponíveis para Binge; "
                f"utilizando total={total_selected}."
            )

        train_target = min(args.balanced_binge_train, total_selected - 1)
        if train_target <= 0:
            raise ValueError(
                "balanced-binge-train deve ser positivo e menor que o total selecionado."
            )
        binge_test_size = len(binge_df) - train_target
        if binge_test_size <= 0:
            raise ValueError(
                "Configuração de balanceamento deixou o conjunto de teste vazio."
            )
        binge_val_fraction = args.balanced_binge_val_frac
        log(
            f"Amostra balanceada criada: total={len(binge_df)}, treino+val={train_target}, "
            f"teste={binge_test_size}. Validação dentro do treino: {binge_val_fraction*100:.2f}%"
        )

    log("Treinando e avaliando modelos para Binge…")
    binge_results, (binge_best_name, binge_best_acc, binge_best_pipe) = fit_and_evaluate(
        binge_df,
        binge_y,
        preprocess,
        cat_cols,
        num_cols,
        target_name="Binge",
        models=models,
        test_size=binge_test_size,
        val_fraction_within_train=binge_val_fraction,
    )

    log("Treinando e avaliando modelos para Fumante atual…")
    smoke_results, (smoke_best_name, smoke_best_acc, smoke_best_pipe) = fit_and_evaluate(
        df, y_smoke, preprocess, cat_cols, num_cols, target_name="Fumante atual", models=models
    )

    log("Treinamento e avaliação concluídos para todos os modelos.")

    # Salvar artefatos
    log("Salvando pipelines treinados em models/…")
    os.makedirs("models", exist_ok=True)
    dump(binge_best_pipe, os.path.join("models", "alcohol_binge_model.joblib"))
    dump(smoke_best_pipe, os.path.join("models", "smoker_current_model.joblib"))

    # Relatório final amigável
    print("===== RESUMO =====")
    def _fmt_res(res: Dict[str, Dict[str, float]]) -> List[str]:
        lines = []
        for name in sorted(res.keys()):
            v = res[name]
            test_display = "--"
            if not np.isnan(v["test_acc"]):
                test_display = f"{v['test_acc']*100:.2f}%"
            lines.append(
                f"  - {name}: Val {v['val_acc']*100:.2f}% | Test {test_display}"
            )
        return lines

    print("Alvo: Binge")
    for line in _fmt_res(binge_results):
        print(line)
    binge_best_val = binge_results[binge_best_name]["val_acc"] * 100
    print(
        f"  -> Melhor: {binge_best_name} (Val {binge_best_val:.2f}% | Test {binge_best_acc*100:.2f}%)"
    )
    cm_b = binge_results[binge_best_name].get("confusion_matrix")
    if cm_b:
        print("  Matriz de confusão (Teste) [verdadeiro 0/1 x previsto 0/1]:")
        print(f"    [{cm_b[0][0]}  {cm_b[0][1]}]")
        print(f"    [{cm_b[1][0]}  {cm_b[1][1]}]")
        prec_b = binge_results[binge_best_name].get("precision")
        rec_b = binge_results[binge_best_name].get("recall")
        spec_b = binge_results[binge_best_name].get("specificity")
        f1_b = binge_results[binge_best_name].get("f1")
        print(
            "  Análise: "
            f"Precisão={prec_b:.3f} | Recall={rec_b:.3f} | Especificidade={spec_b:.3f} | F1={f1_b:.3f}"
        )

    print("Alvo: Fumante atual")
    for line in _fmt_res(smoke_results):
        print(line)
    smoke_best_val = smoke_results[smoke_best_name]["val_acc"] * 100
    print(
        f"  -> Melhor: {smoke_best_name} (Val {smoke_best_val:.2f}% | Test {smoke_best_acc*100:.2f}%)"
    )
    cm_s = smoke_results[smoke_best_name].get("confusion_matrix")
    if cm_s:
        print("  Matriz de confusão (Teste) [verdadeiro 0/1 x previsto 0/1]:")
        print(f"    [{cm_s[0][0]}  {cm_s[0][1]}]")
        print(f"    [{cm_s[1][0]}  {cm_s[1][1]}]")
        prec_s = smoke_results[smoke_best_name].get("precision")
        rec_s = smoke_results[smoke_best_name].get("recall")
        spec_s = smoke_results[smoke_best_name].get("specificity")
        f1_s = smoke_results[smoke_best_name].get("f1")
        print(
            "  Análise: "
            f"Precisão={prec_s:.3f} | Recall={rec_s:.3f} | Especificidade={spec_s:.3f} | F1={f1_s:.3f}"
        )

    print(
        "Observação: accuracy em %; dataset: BRFSS 2015; ausentes tratados por coluna "
        "via regras específicas e heurística segura."
    )


if __name__ == "__main__":
    main()
