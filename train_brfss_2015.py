#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: train_brfss_2015.py

Objetivo: Treinar modelos para dois alvos no BRFSS 2015 (binge drinking e fumante atual)
com pipeline de pré-processamento e salvar os artefatos. O script imprime mensagens
didáticas no formato "PASSO X: ..." ao longo do processo.

Observação importante: códigos de não resposta variam por coluna. A fonte de verdade são
os metadados do arquivo JSON (2015_formats.json) e o PDF “dados do dataset.pdf”. Este
script respeita a especificidade por coluna quando o JSON está disponível; caso contrário,
aplica heurísticas seguras apenas aos códigos presentes em cada coluna.
"""

from __future__ import annotations

import os
import json
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

# Tentar importar pacotes opcionais
_HAS_XGB = False
_HAS_LGBM = False
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


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


def load_formats_json(json_path: str) -> Dict[str, Any] | None:
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Se o JSON estiver inválido, ignore de forma segura.
        return None


def build_missing_map(
    df: pd.DataFrame, fmt_json: Dict[str, Any] | None
) -> Dict[str, Set[Any]]:
    """
    Constrói um mapa coluna->conjunto de códigos a serem tratados como ausentes (NaN).

    - Se houver JSON de formatos (2015_formats.json), usa rótulos que indicam não resposta
      (e.g., "Don't know", "Refused", "Missing", "Unknown") para coletar os códigos por coluna.
    - Caso contrário, aplica heurística segura por coluna: somente adiciona aos ausentes
      os códigos clássicos que existirem naquela coluna (p.ex. {7,9,77,88,99,555,777,888,999}).

    Importante: os códigos de não resposta variam por coluna; JAMAS aplicar um conjunto
    genérico a todas as colunas sem verificar presença na coluna.
    """
    missing_map: Dict[str, Set[Any]] = {c: set() for c in df.columns}

    classic_codes = {7, 9, 77, 88, 99, 555, 777, 888, 999}
    classic_str = {str(x) for x in classic_codes}

    keywords = [
        "don't know",
        "dont know",
        "refused",
        "missing",
        "unknown",
        "not ascertained",
        "dk",
        "na",
        "blank",
        "illegible",
    ]

    if fmt_json is not None and isinstance(fmt_json, dict):
        # Tentar inferir estrutura: col -> {code->label} ou col -> {"values": {code->label}} etc.
        for col in df.columns:
            try:
                meta = fmt_json.get(col)
                if meta is None:
                    continue

                # Diversas formas possíveis
                if isinstance(meta, dict):
                    # 1) Pode ser diretamente {code: label}
                    candidates = meta
                    # 2) Ou aninhado sob algumas chaves comuns
                    for key in [
                        "values",
                        "codes",
                        "labels",
                        "mapping",
                        "map",
                        "value_labels",
                    ]:
                        if key in meta and isinstance(meta[key], dict):
                            candidates = meta[key]
                            break

                    for k, v in candidates.items():
                        # Tentar normalizar o código (k) e rótulo (v)
                        try:
                            code = int(k)
                        except Exception:
                            code = k

                        label = str(v).strip().lower()

                        if any(word in label for word in keywords) or (
                            str(k).strip() in classic_str
                        ):
                            missing_map[col].add(code)
            except Exception:
                # Seja resiliente a inconsistências
                continue

    else:
        # Heurística segura: por coluna, somente códigos clássicos existentes.
        for col in df.columns:
            try:
                uniques = set(pd.Series(df[col]).dropna().unique().tolist())
            except Exception:
                uniques = set()

            # Adicionar apenas os que existem na coluna (numérico e string)
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
        "_RFDRHV5": {9: np.nan, 1: 0.0, 2: 1.0},
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


def derive_targets(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[str]]:
    """
    Constrói os dois alvos: binge e fumante atual.

    - Binge (binário):
        Preferir _RFBING5 se existir (mapear 2->1, 1->0, outros->NaN). Caso contrário,
        derivar de DRNK3GE5: valores válidos > 0 nos últimos 30 dias => 1; 0/"none" => 0;
        não resposta => NaN (já tratados por missing_map).

    - Fumante atual (binário):
        Preferir _SMOKER3: 1/2 => 1; 3/4 => 0; demais => NaN. Se ausente, SMOKDAY2 com
        mapeamento 1/2 => 1; 3 => 0; demais => NaN.

    Retorna (y_binge, y_smoke, leakage_cols) – leakage_cols devem ser removidas das features.
    """
    leakage_cols: List[str] = []

    # Binge
    if "_RFBING5" in df.columns:
        leakage_cols.append("_RFBING5")
        s = df["_RFBING5"]
        y_binge = pd.Series(np.where(s == 2, 1.0, np.where(s == 1, 0.0, np.nan)), index=s.index)
    elif "DRNK3GE5" in df.columns:
        leakage_cols.append("DRNK3GE5")
        s = df["DRNK3GE5"].copy()
        # Tratar possíveis strings 'none'
        s = s.replace({"none": 0, "None": 0, "NONE": 0})
        s_num = pd.to_numeric(s, errors="coerce")
        y_binge = (s_num.notna() & (s_num > 0)).astype(float)
        # Valores 0 (ou 'none' convertido) => 0.0; NaNs já tratados
    else:
        raise ValueError(
            "Não foi possível construir o alvo Binge: faltam colunas _RFBING5 e DRNK3GE5."
        )

    # Fumante atual
    if "_SMOKER3" in df.columns:
        leakage_cols.append("_SMOKER3")
        s = df["_SMOKER3"]
        y_smoke = pd.Series(
            np.where(s.isin([1, 2]), 1.0, np.where(s.isin([3, 4]), 0.0, np.nan)), index=s.index
        )
    elif "SMOKDAY2" in df.columns:
        leakage_cols.append("SMOKDAY2")
        s = df["SMOKDAY2"]
        y_smoke = pd.Series(
            np.where(s.isin([1, 2]), 1.0, np.where(s == 3, 0.0, np.nan)), index=s.index
        )
    else:
        raise ValueError(
            "Não foi possível construir o alvo Fumante atual: faltam colunas _SMOKER3 e SMOKDAY2."
        )

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


def get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
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

    if _HAS_XGB:
        try:
            models["XGB"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            )
        except Exception:
            pass

    if _HAS_LGBM:
        try:
            models["LGBM"] = LGBMClassifier(
                n_estimators=400,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        except Exception:
            pass

    return models


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


def evaluate_model(clf, X_val, y_val, X_test, y_test) -> Tuple[float, float, float]:
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_acc = metrics.accuracy_score(y_val, val_pred)
    test_acc = metrics.accuracy_score(y_test, test_pred)

    try:
        test_scores = _proba_or_score(clf, X_test)
        test_auc = metrics.roc_auc_score(y_test, test_scores)
    except Exception:
        test_auc = np.nan

    return val_acc, test_acc, test_auc


def _compute_scale_pos_weight(y: pd.Series) -> float:
    try:
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos == 0:
            return 1.0
        return max(1.0, neg / max(1, pos))
    except Exception:
        return 1.0


def tune_and_evaluate(
    name: str,
    base_pipe: Pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    X_train_val,
    y_train_val,
    X_test,
    y_test,
) -> Tuple[Pipeline, float, float, float]:
    """
    Faz busca de hiperparâmetros para XGB/LGBM no conjunto de treino (com CV),
    avalia no holdout de validação para referência e refita em train+val para
    avaliar no teste final.
    """
    assert name in {"XGB", "LGBM"}

    spw = _compute_scale_pos_weight(pd.Series(y_train))

    if name == "XGB":
        # Espaço de busca XGBoost — focado em controlar overfitting/complexidade
        param_distributions = {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [3, 4, 5, 6, 8, 10],
            "clf__min_child_weight": [1, 2, 3, 5, 7, 10],
            "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__gamma": [0, 0.5, 1, 2],
            "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "clf__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__scale_pos_weight": [1.0, spw, max(1.0, np.sqrt(spw))],
        }
        n_iter = 20
    else:  # LGBM
        param_distributions = {
            "clf__n_estimators": [300, 400, 600, 800, 1000],
            "clf__num_leaves": [15, 31, 63, 127],
            "clf__max_depth": [-1, 4, 6, 8, 12],
            "clf__min_child_samples": [5, 10, 20, 30, 50],
            "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "clf__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__min_split_gain": [0.0, 0.1, 0.5, 1.0],
        }
        n_iter = 20

    search = skms.RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="accuracy",
        n_jobs=-1,
        cv=3,
        refit=False,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    # Busca em X_train (sem usar validação)
    search.fit(X_train, y_train)

    # Construir pipeline final com melhores hiperparâmetros
    best_params = search.best_params_
    tuned_pipe: Pipeline = Pipeline(steps=[("prep", base_pipe.named_steps["prep"]), ("clf", base_pipe.named_steps["clf"])])
    tuned_pipe.set_params(**best_params)

    # 1) Métrica de validação (fit em X_train, pred em X_val)
    tuned_pipe.fit(X_train, y_train)
    val_acc, _, _ = evaluate_model(tuned_pipe, X_val, y_val, X_val, y_val)

    # 2) Refit em train+val para avaliação final de teste
    tuned_pipe.fit(X_train_val, y_train_val)
    _, test_acc, test_auc = evaluate_model(tuned_pipe, X_val, y_val, X_test, y_test)

    return tuned_pipe, float(val_acc), float(test_acc), float(test_auc)


def fit_and_evaluate(
    df: pd.DataFrame,
    y: pd.Series,
    preprocess: compose.ColumnTransformer,
    cat_cols: List[str],
    num_cols: List[str],
    target_name: str,
    models: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, float]], Tuple[str, float, Pipeline]]:
    """
    - Filtra linhas com y ausente
    - Split estratificado 70/15/15 (via 85/15 e depois 70/15 dentro do 85)
    - Treina e avalia modelos, devolve resultados e o melhor por Test Accuracy
    """
    # Filtrar apenas registros com y válido
    mask = y.notna()
    df_ = df.loc[mask, cat_cols + num_cols].copy()
    y_ = y.loc[mask].astype(int)

    # Split 85/15 para treino+val e teste
    X_train_val, X_test, y_train_val, y_test = skms.train_test_split(
        df_, y_, test_size=0.15, stratify=y_, random_state=RANDOM_STATE
    )
    # Split treino/val para obter 70/15 do total
    val_ratio_within_train = 15 / 85
    X_train, X_val, y_train, y_val = skms.train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_within_train,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    log(f"Split estratificado em treino/val/test definido para {target_name} (70/15/15).")

    # Resultados
    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_acc = -np.inf
    best_pipe: Pipeline | None = None

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])

        # Tunar XGB/LGBM conforme solicitado; manter demais como antes
        if name in {"XGB", "LGBM"}:
            try:
                tuned_pipe, val_acc, test_acc, test_auc = tune_and_evaluate(
                    name, pipe, X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test
                )
                pipe = tuned_pipe
            except Exception as e:
                print(f"[Aviso] Falha ao tunar {name}: {e}. Usando parâmetros padrão.")
                pipe.fit(X_train, y_train)
                val_acc, test_acc, test_auc = evaluate_model(pipe, X_val, y_val, X_test, y_test)
        else:
            pipe.fit(X_train, y_train)
            val_acc, test_acc, test_auc = evaluate_model(pipe, X_val, y_val, X_test, y_test)

        results[name] = {
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc) if not np.isnan(test_auc) else np.nan,
        }

        print(
            f"Modelo={name} | Alvo={target_name.upper()} | "
            f"Val Accuracy={val_acc*100:.2f}% | Test Accuracy={test_acc*100:.2f}% | "
            f"Test ROC-AUC={(test_auc*100 if not np.isnan(test_auc) else np.nan):.2f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None
    return results, (best_name, best_acc, best_pipe)


def main() -> None:
    # Carregar dados
    log("Carregando dataset BRFSS 2015…")
    csv_path = os.path.join("data", "2015.csv")
    df = read_dataset(csv_path)

    log("Aplicando limpeza manual nas colunas selecionadas…")
    df = apply_manual_cleaning(df)

    # Ler metadados e construir mapa de ausentes
    log("Lendo 2015_formats.json (se disponível) para mapear códigos ausentes por coluna…")
    json_path = os.path.join("data", "2015_formats.json")
    fmt_json = load_formats_json(json_path)

    missing_map = build_missing_map(df, fmt_json)
    df = apply_missing_map(df, missing_map)
    log("Ausentes tratados respeitando a especificidade por coluna.")

    # Construção dos alvos
    log("Construindo os alvos (Binge e Fumante atual)…")
    y_binge, y_smoke, leakage_cols = derive_targets(df)

    # Seleção de features
    log("Selecionando features disponíveis e separando tipos…")
    cat_cols, num_cols = select_features(df, leakage_cols)
    if not cat_cols and not num_cols:
        raise RuntimeError("Nenhuma feature disponível após filtragem.")

    # Pré-processamento
    log("Montando pipeline de pré-processamento (imputação + one-hot + padronização)…")
    preprocess = build_preprocess(cat_cols, num_cols)

    # Preparar modelos
    models = get_models()
    if not _HAS_XGB:
        print("[Aviso] xgboost não instalado – pulando XGB.")
    if not _HAS_LGBM:
        print("[Aviso] lightgbm não instalado – pulando LGBM.")

    # Treinamento e avaliação por alvo
    log("Treinando e avaliando modelos para Binge…")
    binge_results, (binge_best_name, binge_best_acc, binge_best_pipe) = fit_and_evaluate(
        df, y_binge, preprocess, cat_cols, num_cols, target_name="Binge", models=models
    )

    log("Treinando e avaliando modelos para Fumante atual…")
    smoke_results, (smoke_best_name, smoke_best_acc, smoke_best_pipe) = fit_and_evaluate(
        df, y_smoke, preprocess, cat_cols, num_cols, target_name="Fumante atual", models=models
    )

    log("Treinamento e avaliação concluídos para todos os modelos.")

    # Salvar artefatos
    log("Salvando modelos e pipeline em models/…")
    os.makedirs("models", exist_ok=True)
    dump(preprocess, os.path.join("models", "feature_pipeline.joblib"))
    dump(binge_best_pipe, os.path.join("models", "alcohol_binge_model.joblib"))
    dump(smoke_best_pipe, os.path.join("models", "smoker_current_model.joblib"))

    # Relatório final amigável
    print("===== RESUMO =====")
    def _fmt_res(res: Dict[str, Dict[str, float]]) -> List[str]:
        lines = []
        for name in ["LogReg", "RF", "GB", "XGB", "LGBM"]:
            if name in res:
                v = res[name]
                lines.append(
                    f"  - {name}: Val {v['val_acc']*100:.2f}% | Test {v['test_acc']*100:.2f}%"
                )
        return lines

    print("Alvo: Binge")
    for line in _fmt_res(binge_results):
        print(line)
    print(
        f"  -> Melhor (por Test Accuracy): {binge_best_name} ({binge_best_acc*100:.2f}%)"
    )

    print("Alvo: Fumante atual")
    for line in _fmt_res(smoke_results):
        print(line)
    print(
        f"  -> Melhor (por Test Accuracy): {smoke_best_name} ({smoke_best_acc*100:.2f}%)"
    )

    print(
        "Observação: accuracy em %; dataset: BRFSS 2015; ausentes tratados por coluna conforme "
        "formatos/JSON e PDF."
    )


if __name__ == "__main__":
    main()
