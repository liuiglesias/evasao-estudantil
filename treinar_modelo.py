import pandas as pd

from sklearn.model_selection import train_test_split

# carregar os dados tratados
df = pd.read_csv("data/students_tratados.csv")

# separar variáveis preditoras e variável-alvo
X = df.drop(columns=["Target", "TargetBinario"])

y = df["TargetBinario"]

# separar treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# verificar a divisão
print("=== DIVISÃO DOS DADOS ===")

print("Total de registros:", len(df))

print("Registros de treinamento:", len(X_treino))

print("Registros de teste:", len(X_teste))

print("\n=== DISTRIBUIÇÃO DAS CLASSES ===")

print("Treinamento:")
print(y_treino.value_counts())

print("\nTeste:")
print(y_teste.value_counts())

print("\nQuantidade de variáveis preditoras:", X.shape[1])

from sklearn.compose import ColumnTransformer #aplica tratamentos diferentes por tipo de variável
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder #padroniza variáveis numéricas; converte categorias em números
from sklearn.impute import SimpleImputer #preenche valores ausentes
from sklearn.linear_model import LogisticRegression #treina o classificador

# identificar variáveis categóricas e numéricas

colunas_categoricas = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

# Variáveis binárias representadas por números
categoricas_binarias = [
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "NecessidadesEspeciais",
    "International"
]

for coluna in categoricas_binarias:
    if coluna in X.columns and coluna not in colunas_categoricas:
        colunas_categoricas.append(coluna)

colunas_numericas = [
    coluna for coluna in X.columns
    if coluna not in colunas_categoricas
]

print("\n=== TIPOS DE VARIÁVEIS ===")
print("Numéricas:", len(colunas_numericas))
print("Categóricas:", len(colunas_categoricas))

print("\n=== COLUNAS CLASSIFICADAS COMO CATEGÓRICAS ===")
print(colunas_categoricas)

print("\n=== COLUNAS CLASSIFICADAS COMO NUMÉRICAS ===")
print(colunas_numericas)

# Pipeline numérico

pipeline_numerico = Pipeline([
    ("imputacao", SimpleImputer(strategy="median")),
    ("padronizacao", StandardScaler())
])

# Pipeline categórico

pipeline_categorico = Pipeline([
    ("imputacao", SimpleImputer(strategy="most_frequent")),
    ("codificacao", OneHotEncoder(handle_unknown="ignore"))
])

# pré-processamento

preprocessador = ColumnTransformer([
    ("numerico", pipeline_numerico, colunas_numericas),
    ("categorico", pipeline_categorico, colunas_categoricas)
])

# Pipeline completo de Machine Learning

modelo = Pipeline([
    ("preprocessamento", preprocessador),
    ("classificador", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

print("\n=== PIPELINE CRIADO COM SUCESSO ===")

# treinar o modelo

print("\n=== INICIANDO TREINAMENTO ===")

modelo.fit(X_treino, y_treino)

print("Modelo treinado com sucesso!")

# realizar previsões

previsoes = modelo.predict(X_teste)

print("\n=== PRIMEIRAS 10 PREVISÕES ===")
print(previsoes[:10])

print("\n=== PRIMEIROS 10 RESULTADOS REAIS ===")
print(y_teste.iloc[:10].values)

## avaliar o desempenho do modelo

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

## Probabilidade estimada de desistência (classe 1)
probabilidades = modelo.predict_proba(X_teste)[:, 1]

print("\n=== MÉTRICAS DO MODELO ===")

print("Acurácia:", round(accuracy_score(y_teste, previsoes), 4)) #Qual percentual total de previsões foi correto?

print("Precisão:", round(precision_score(y_teste, previsoes), 4)) #Dos estudantes previstos como desistentes, quantos realmente desistiram?

print("Recall:", round(recall_score(y_teste, previsoes), 4)) #Dos estudantes que realmente desistiram, quantos foram identificados? (+importante)

print("F1-score:", round(f1_score(y_teste, previsoes), 4)) #Qual o equilíbrio entre precisão e recall?

print("ROC-AUC:", round(roc_auc_score(y_teste, probabilidades), 4)) #Qual a capacidade do modelo de distinguir graduados de desistentes em diferentes limiares?

print("\n=== MATRIZ DE CONFUSÃO ===")
print(confusion_matrix(y_teste, previsoes))

print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(
    classification_report(
        y_teste,
        previsoes,
        target_names=["Graduado", "Desistente"]
    )
)

### validação cruzada estratificada com 5 folds

from sklearn.model_selection import StratifiedKFold, cross_val_score #evita que informações do grupo de validação sejam utilizadas antecipadamente na imputação, padronização ou codificação.
import numpy as np

print("\n=== VALIDAÇÃO CRUZADA ===")

validacao = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados_cv = cross_val_score(
    modelo,
    X_treino,
    y_treino,
    cv=validacao,
    scoring="roc_auc",
    n_jobs=1
)

print("ROC-AUC em cada fold:")
print(np.round(resultados_cv, 4))

print(
    "ROC-AUC médio:",
    round(resultados_cv.mean(), 4)
)

print(
    "Desvio-padrão:",
    round(resultados_cv.std(), 4)
)

#### Comparar desempenho entre treinamento, validação cruzada e teste

print("\n=== ANÁLISE DE OVERFITTING E UNDERFITTING ===")

#### Probabilidades calculadas no conjunto de treinamento
probabilidades_treino = modelo.predict_proba(X_treino)[:, 1]

#### ROC-AUC do treinamento
auc_treino = roc_auc_score(
    y_treino,
    probabilidades_treino
)

#### ROC-AUC médio da validação cruzada
auc_validacao = resultados_cv.mean()

#### ROC-AUC do conjunto de teste
auc_teste = roc_auc_score(
    y_teste,
    probabilidades
)

print("ROC-AUC treinamento:", round(auc_treino, 4))
print("ROC-AUC validação cruzada:", round(auc_validacao, 4))
print("ROC-AUC teste:", round(auc_teste, 4))

print(
    "Diferença treino - validação:",
    round(auc_treino - auc_validacao, 4)
)

print(
    "Diferença treino - teste:",
    round(auc_treino - auc_teste, 4)
)


##### gravação do pipeline treinado

import joblib #salva e carrega objetos
from pathlib import Path

##### criação a pasta de modelos, caso ainda não exista
pasta_modelos = Path("models")
pasta_modelos.mkdir(exist_ok=True)

##### caminho do arquivo
caminho_modelo = pasta_modelos / "modelo_evasao.joblib"

##### gravação do pipeline completo
joblib.dump(modelo, caminho_modelo)

print("\n=== MODELO SALVO ===")
print("Arquivo:", caminho_modelo)
print("Tamanho em bytes:", caminho_modelo.stat().st_size)