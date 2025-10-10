# Ciência de Dados — Criação do Modelo

Este projeto treina e avalia dois modelos de classificação binária usando o BRFSS 2015:

- Binge drinking (consumo excessivo de álcool)
- Fumante atual

O pipeline inclui limpeza específica por coluna, tratamento de ausentes, derivação dos alvos, pré-processamento (categóricas e numéricas) e treino/validação/teste. Os modelos treinados são salvos como pipelines prontos para inferência.

## Requisitos

- Python 3.10+
- Instalar dependências:

```
pip install -r requirements.txt
```

## Dados

Coloque o arquivo do BRFSS 2015 em `data/2015.csv`.

As colunas necessárias são um subconjunto definido no código (`SELECTED_FEATURES`) e as colunas de alvo `_RFDRHV5` (binge) e `_SMOKER3` (fumante). O script valida a presença dessas colunas.

## Treinamento

Com modelos “full” (padrão):

```
python train_brfss_2015.py
```

Execução rápida (modelos mais leves):

```
python train_brfss_2015.py --quick
```

Amostragem de N linhas para acelerar:

```
python train_brfss_2015.py --sample 50000
```

Balancear o alvo de Binge (50/50) com teste externo:

```
python train_brfss_2015.py \
  --balanced-binge \
  --balanced-binge-size 25000 \
  --balanced-binge-train 40000 \
  --balanced-binge-val-frac 0.2 \
  --balanced-binge-holdout 0.1
```

Ao final, os modelos são salvos em:

- `models/alcohol_binge_model.joblib`
- `models/smoker_current_model.joblib`

## Avaliação (modelos salvos)

Avalia ambos os modelos salvos em conjunto, reutilizando a mesma limpeza e derivação de alvos do treino:

```
python scripts/evaluate_models.py --dataset data/2015.csv --models-dir models
```

Opções úteis:

- `--sample N` — avalia em uma amostra de N linhas
- `--only binge|smoke|both` — escolhe qual(is) modelo(s) avaliar (padrão: `both`)

Saída esperada: métricas por alvo (Accuracy, ROC-AUC, matriz de confusão, Precisão, Recall, Especificidade, F1).

## Estrutura

- `train_brfss_2015.py` — script principal de treinamento (gera os `.joblib`)
- `scripts/evaluate_models.py` — avaliação dos modelos salvos
- `data/2015.csv` — dataset de entrada (não versionado)
- `models/` — artefatos dos modelos (saída)
- `requirements.txt` — dependências do projeto

## Observações

- O pré-processamento no pipeline salvo inclui imputação, clipping IQR, padronização para numéricos e one‑hot para categóricos.
- A limpeza inclui regras por coluna e uma heurística segura de códigos de não resposta.
- O script lida com diferenças de versão do scikit‑learn para `OneHotEncoder`.
