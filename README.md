# Ciência de Dados — Tabagismo BRFSS 2015

Este projeto treina e avalia um modelo de classificação binária para identificar
fumantes atuais (`_SMOKER3`) no BRFSS 2015. O pipeline aplica limpeza específica
por coluna, trata códigos de não resposta, remove linhas com valores ausentes,
balanceia o conjunto 50/50 entre fumantes e não fumantes e treina diferentes
classificadores. O melhor modelo é salvo como um pipeline do scikit-learn pronto
para inferência, juntamente com métricas e importâncias de variáveis.

## Requisitos

- Python 3.10+
- Instalar as dependências:

```
pip install -r requirements.txt
```

## Dados

Coloque o arquivo BRFSS 2015 em `data/2015.csv`.

As colunas necessárias são o conjunto definido em `SELECTED_FEATURES` e a coluna
de alvo `_SMOKER3`. O script valida a presença dessas colunas antes de iniciar o
treinamento.

## Treinamento

Executa o pipeline completo:

```
python train_brfss_2015.py
```

Opções úteis:

- `--dataset PATH` — caminho alternativo para o CSV.
- `--sample N` — amostragem aleatória de N linhas para execuções rápidas.
- `--quick` — usa hiperparâmetros mais leves.
- `--balance-per-class N` — número máximo de registros por classe após o balanceamento 50/50.
- `--balance-holdout F` — fração mínima (0–0.5) reservada para teste externo após balancear.

Durante o treinamento:

1. Linhas com valores ausentes no alvo ou nas features são removidas.
2. O dataset é balanceado para garantir a mesma quantidade de fumantes e não fumantes.
3. Diversos classificadores são avaliados e o melhor (pela validação interna) é ajustado em treino+validação.
4. O desempenho final é reportado (teste interno e opcionalmente externo) com métricas, matriz de confusão e importâncias de variáveis (permutation importance).

O artefato treinado é salvo em:

- `models/smoker_current_model.joblib`

## Avaliação

A avaliação reutiliza as mesmas etapas de limpeza e preparação para medir o desempenho do modelo salvo:

```
python scripts/evaluate_models.py --dataset data/2015.csv --models-dir models
```

Argumentos:

- `--sample N` — avalia em uma amostra de N linhas.
- `--models-dir DIR` — diretório contendo o `.joblib` treinado.

O script imprime métricas (Accuracy, ROC-AUC, Precisão, Recall, Especificidade e F1) e a matriz de confusão.

## Estrutura

- `train_brfss_2015.py` — script de treinamento e persistência do modelo.
- `scripts/evaluate_models.py` — avaliação do modelo salvo.
- `model_metrics.py` — utilitários compartilhados para métricas.
- `data/2015.csv` — dataset de entrada (não versionado).
- `models/` — artefatos produzidos pelo treinamento.
- `requirements.txt` — dependências do projeto.

## Observações

- O pré-processamento inclui imputação, winsorização via IQR, padronização e one-hot encoding.
- Regras específicas por coluna removem códigos de não resposta antes do balanceamento.
- As importâncias de variáveis (permutation importance) ajudam a identificar quais fatores mais contribuem para o tabagismo atual.
