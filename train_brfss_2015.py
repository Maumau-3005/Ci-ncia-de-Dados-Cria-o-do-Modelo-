#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: train_brfss_2015.py

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
from sklearn import linear_model
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance

from model_metrics import classification_metrics, proba_or_score
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
SELECTED_FEATURES = [
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
    holdout_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series, int, np.ndarray]:
    """Seleciona amostra balanceada 50/50 entre classes positiva e negativa."""

    valid_mask = y.isin([positive_label, negative_label])
    y_valid = y.loc[valid_mask]

    pos_idx = y_valid[y_valid == positive_label].index
    neg_idx = y_valid[y_valid == negative_label].index

    holdout_ratio = min(max(holdout_ratio, 0.0), 0.5)
    holdout_pos = max(1, int(len(pos_idx) * holdout_ratio)) if len(pos_idx) > 0 else 0
    holdout_neg = max(1, int(len(neg_idx) * holdout_ratio)) if len(neg_idx) > 0 else 0

    available_pos = len(pos_idx) - holdout_pos
    available_neg = len(neg_idx) - holdout_neg

    actual_per_class = min(per_class, available_pos, available_neg)

    if actual_per_class <= 0:
        raise ValueError(
            "Não há amostras suficientes para balanceamento: "
            f"positivos disponíveis={len(pos_idx)}, negativos disponíveis={len(neg_idx)}, "
            f"necessário por classe={per_class} após reservar {holdout_pos} positivos e {holdout_neg} negativos para teste."
        )

    rng = np.random.default_rng(random_state)
    pos_sample = rng.choice(pos_idx, size=actual_per_class, replace=False)
    neg_sample = rng.choice(neg_idx, size=actual_per_class, replace=False)

    selected_idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(selected_idx)

    return (
        df.loc[selected_idx].copy(),
        y.loc[selected_idx].copy(),
        actual_per_class,
        selected_idx,
    )


def derive_smoker_target(df: pd.DataFrame) -> pd.Series:
    """
    Constrói o alvo binário de fumante atual (1 = fumante, 0 = não fumante)
    com base na coluna `_SMOKER3`. Outros códigos são convertidos para NaN.
    """
    if "_SMOKER3" not in df.columns:
        raise ValueError(
            "Não foi possível construir o alvo: coluna obrigatória `_SMOKER3` ausente."
        )

    s_smoke = pd.to_numeric(df["_SMOKER3"], errors="coerce")
    y_smoke = pd.Series(
        np.where(s_smoke.isin([1, 2]), 1.0, np.where(s_smoke.isin([3, 4]), 0.0, np.nan)),
        index=s_smoke.index,
    )
    return y_smoke.astype(float)


def select_features(
    df: pd.DataFrame, leakage_cols: Iterable[str] | None = None
) -> Tuple[List[str], List[str]]:
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
    for leak in leakage_cols or ():
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


def prepare_smoker_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepara o conjunto de dados para treinamento/avaliação do alvo fumante.

    - Seleciona features válidas
    - Constrói o alvo `_SMOKER3` (binário)
    - Remove linhas com qualquer NaN em features ou target
    """
    y_smoke = derive_smoker_target(df)
    cat_cols, num_cols = select_features(df, leakage_cols=["_SMOKER3"])
    feature_cols = cat_cols + num_cols

    if not feature_cols:
        raise RuntimeError("Nenhuma feature disponível após filtragem.")

    X = df.loc[:, feature_cols].copy()
    dataset = X.join(y_smoke.rename("target"))
    dataset = dataset.dropna()

    y_clean = dataset.pop("target").astype(int)
    return dataset, y_clean, cat_cols, num_cols


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
            "RF": ensemble.RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                max_features="sqrt",
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
            n_estimators=500,
            max_depth=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GB": ensemble.GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


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
    external_X_test: pd.DataFrame | None = None,
    external_y_test: pd.Series | None = None,
) -> Tuple[Dict[str, Dict[str, float]], Tuple[str, float, Pipeline]]:
    """
    - Filtra linhas com y ausente
    - Split estratificado conforme parâmetros (test_size pode ser fração ou inteiro)
    - Treina e avalia modelos, devolve resultados e o melhor selecionado por Val Accuracy
      (Test Accuracy reportado após ajuste final; se external_X_test estiver definido, usa-o)
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

    eval_X = X_test
    eval_y = y_test
    eval_desc = "teste interno estratificado"
    if external_X_test is not None and external_y_test is not None:
        eval_X = external_X_test
        eval_y = external_y_test
        eval_desc = "teste externo"
        log(
            f"Executando avaliação externa para {target_name} com {len(eval_X)} registros."
        )

    final_test_pred = final_pipe.predict(eval_X)
    try:
        final_test_scores = proba_or_score(final_pipe, eval_X)
    except Exception:
        final_test_scores = None

    metric_summary = classification_metrics(
        eval_y, final_test_pred, scores=final_test_scores
    )

    final_test_acc = metric_summary["accuracy"]
    final_test_auc = metric_summary["roc_auc"]

    results[best_name]["test_acc"] = final_test_acc
    results[best_name]["test_auc"] = final_test_auc
    results[best_name]["confusion_matrix"] = metric_summary["confusion_matrix"]
    results[best_name]["precision"] = metric_summary["precision"]
    results[best_name]["recall"] = metric_summary["recall"]
    results[best_name]["specificity"] = metric_summary["specificity"]
    results[best_name]["f1"] = metric_summary["f1"]

    feature_cols = cat_cols + num_cols
    feature_summary: List[Tuple[str, float]] | None = None
    if eval_X is not None and len(eval_X) > 0:
        try:
            pi_result = permutation_importance(
                final_pipe,
                eval_X,
                eval_y,
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            importances = pi_result.importances_mean
            feature_summary = sorted(
                [
                    (feature_cols[idx], float(importances[idx]))
                    for idx in range(len(feature_cols))
                ],
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        except Exception:
            feature_summary = None

    results[best_name]["feature_importances"] = feature_summary or []

    print(
        f"Modelo selecionado (validação)={best_name} | Alvo={target_name.upper()} | "
        f"Val Accuracy={best_val_acc*100:.2f}% | {eval_desc.title()} Accuracy={final_test_acc*100:.2f}% | "
        f"{eval_desc.title()} ROC-AUC={(final_test_auc*100 if not np.isnan(final_test_auc) else np.nan):.2f}"
    )

    return results, (best_name, float(final_test_acc), final_pipe)


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
    df = read_dataset(args.dataset)
    if args.sample is not None and args.sample > 0 and args.sample < len(df):
        log(f"Amostrando {args.sample} linhas do dataset para execução rápida…")
        df = df.sample(n=args.sample, random_state=RANDOM_STATE)

    required_cols = sorted(set(SELECTED_FEATURES + ["_SMOKER3"]))
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

    log("Preparando dataset limpo para o alvo fumante atual…")
    X_full, y_full, cat_cols, num_cols = prepare_smoker_dataset(df)
    total = len(y_full)
    positives = int((y_full == 1).sum())
    negatives = int((y_full == 0).sum())
    log(
        f"Após limpeza, restaram {total} registros válidos "
        f"({positives} fumantes, {negatives} não fumantes)."
    )

    log("Balanceando o dataset (50/50) via subamostragem estratificada…")
    X_bal, y_bal, actual_per_class, selected_idx = create_balanced_sample(
        X_full,
        y_full,
        positive_label=1.0,
        negative_label=0.0,
        per_class=args.balance_per_class,
        random_state=RANDOM_STATE,
        holdout_ratio=args.balance_holdout,
    )
    log(
        f"Amostra balanceada contém {len(X_bal)} registros ({actual_per_class} por classe)."
    )

    selected_idx = pd.Index(selected_idx)
    remaining_idx = y_full.index.difference(selected_idx)
    external_X: pd.DataFrame | None = None
    external_y: pd.Series | None = None
    if args.balance_holdout > 0 and not remaining_idx.empty:
        external_X = X_full.loc[remaining_idx].copy()
        external_y = y_full.loc[remaining_idx].copy()
        log(
            f"Teste externo utilizará {len(external_X)} registros remanescentes sem sobreposição com a amostra balanceada."
        )
    else:
        log("Sem amostra remanescente suficiente para teste externo; usando apenas teste interno estratificado.")

    log(
        "Montando pipeline de pré-processamento (imputação + clipping IQR + one-hot + padronização)…"
    )
    preprocess = build_preprocess(cat_cols, num_cols)

    models = get_models("quick" if args.quick else "full")

    log("Treinando e avaliando modelos para o alvo Fumante atual…")
    smoke_results, (best_name, best_acc, best_pipe) = fit_and_evaluate(
        X_bal,
        y_bal,
        preprocess,
        cat_cols,
        num_cols,
        target_name="Fumante atual",
        models=models,
        external_X_test=external_X,
        external_y_test=external_y,
    )

    log("Treinamento concluído.")

    log("Salvando pipeline treinado em models/smoker_current_model.joblib…")
    os.makedirs("models", exist_ok=True)
    dump(best_pipe, os.path.join("models", "smoker_current_model.joblib"))

    print("\n========== RESULTADOS ==========")

    print("\n>> FUMANTE ATUAL")
    print("  Modelos avaliados:")
    for name in sorted(smoke_results.keys()):
        metrics_map = smoke_results[name]
        val_txt = _fmt_pct(metrics_map.get("val_acc"))
        test_txt = _fmt_pct(metrics_map.get("test_acc"))
        auc_txt = _fmt_pct(metrics_map.get("test_auc"))
        print(f"    - {name:<30} | Val={val_txt:<8} Test={test_txt:<8} ROC-AUC={auc_txt:<8}")

    best_metrics = smoke_results[best_name]
    print(f"\n  Melhor modelo: {best_name}")
    print(
        "    → Val={val} | Test={test} | ROC-AUC={auc}"
        .format(
            val=_fmt_pct(best_metrics.get("val_acc")),
            test=_fmt_pct(best_metrics.get("test_acc")),
            auc=_fmt_pct(best_metrics.get("test_auc")),
        )
    )

    _print_confusion_matrix(best_metrics.get("confusion_matrix", []))

    print(
        "    Métricas complementares: Precisão={p:.3f} | Recall={r:.3f} | "
        "Especificidade={s:.3f} | F1={f:.3f}"
        .format(
            p=best_metrics.get("precision", float("nan")),
            r=best_metrics.get("recall", float("nan")),
            s=best_metrics.get("specificity", float("nan")),
            f=best_metrics.get("f1", float("nan")),
        )
    )

    feature_summary = best_metrics.get("feature_importances")
    if feature_summary:
        print("\n  Principais variáveis associadas ao tabagismo (permutation importance):")
        for rank, (feat, importance) in enumerate(feature_summary[:10], start=1):
            print(f"    {rank:>2}. {feat:<30} Δ={importance:.5f}")
    else:
        print("\n  Não foi possível calcular importâncias de features.")



if __name__ == "__main__":
    main()
