# Previsão de Evasão Estudantil

Projeto de conclusão da pós-graduação em Machine Learning Engineering, com classificação binária e aplicação web em Streamlit.

## Objetivo

Estimar a probabilidade de desistência de estudantes usando informações pessoais, financeiras e acadêmicas dos dois primeiros semestres.

Como utiliza resultados acadêmicos do segundo semestre, o modelo não representa uma previsão de evasão no momento da matrícula.

## Base e preparação

A base tratada contém 3.629 registros:

- 1.420 estudantes desistentes.
- 2.209 estudantes graduados.

Estudantes com situação “Matriculado” foram excluídos porque seu desfecho ainda é indefinido.

A variável-alvo foi codificada como:

- `Desistente = 1`
- `Graduado = 0`

Foram criadas taxas de aprovação por semestre, calculadas pela divisão entre disciplinas aprovadas e disciplinas inscritas. Quando o denominador é zero, a taxa fica ausente e recebe tratamento no pipeline.

Notas fora da escala de 0 a 20 foram marcadas como ausentes durante a preparação dos dados. Esses valores são imputados pela mediana dentro do pipeline. A mediana é calculada somente nos dados usados para ajustar o pipeline, inclusive em cada divisão da validação cruzada. Não foi realizada recuperação do separador decimal.

## Modelo e pipeline

Foi utilizada Regressão Logística, com:

- Imputação pela mediana e padronização das variáveis numéricas.
- Imputação pela categoria mais frequente e codificação One-Hot das variáveis categóricas.
- Tratamento de indicadores binários como variáveis categóricas.
- Pré-processamento e classificador reunidos em um único pipeline.

As colunas `Target` e `TargetBinario` são excluídas das variáveis preditoras.

## Avaliação

A divisão foi estratificada, com 80% dos registros para treinamento e 20% para teste, usando `random_state=42`.

A validação cruzada estratificada utiliza cinco divisões somente no conjunto de treinamento. O pré-processamento é ajustado dentro de cada divisão.

O código calcula acurácia, precisão, recall de desistência, F1-score, ROC-AUC e matriz de confusão no teste. Também compara ROC-AUC de treinamento, validação e teste para investigar overfitting e underfitting.

### Resultados no teste reservado

O conjunto de teste contém 726 estudantes. As métricas foram conferidas carregando o pipeline salvo, sem retreinamento, e reproduziram os resultados da avaliação original.

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,9215 |
| Precisão de desistência | 0,9127 |
| Recall de desistência | 0,8838 |
| F1-score de desistência | 0,8980 |
| ROC-AUC | 0,9627 |

### Matriz de confusão

| Classe real | Previsto Graduado | Previsto Desistente |
|---|---:|---:|
| Graduado | 418 | 24 |
| Desistente | 33 | 251 |

O modelo identificou 251 dos 284 desistentes do teste, correspondendo a recall de 88,38%. Deixou de identificar 33 desistentes e classificou 24 graduados como desistentes.

### Validação cruzada e capacidade de generalização

A validação cruzada estratificada em cinco divisões, realizada somente no treinamento, apresentou ROC-AUC médio de 0,9518 e desvio-padrão de 0,0095.

| Conjunto | ROC-AUC |
|---|---:|
| Treinamento | 0,9593 |
| Validação cruzada — média | 0,9518 |
| Teste reservado | 0,9627 |

A proximidade dos resultados não indica overfitting acentuado nesta avaliação. O desempenho elevado nos três conjuntos também não sugere underfitting acentuado. Essas observações não garantem generalização para outras instituições ou períodos.

A conferência do modelo salvo reproduziu o teste original; não constitui uma avaliação independente em novos dados.

Métricas calculadas sobre o CSV completo de previsões incluem registros usados no treinamento e não substituem a avaliação no teste reservado.

## Estrutura

- `explorar_dados.py`: exploração inicial.
- `verificar_notas.py`: inspeção das notas.
- `preparar_dados.py`: preparação dos dados e engenharia de atributos.
- `treinar_modelo.py`: treinamento, avaliação e salvamento do pipeline.
- `app.py`: aplicação Streamlit.
- `models/modelo_evasao.joblib`: pipeline treinado.
- `requirements.txt`: dependências.
- `.gitignore`: exclusões do repositório.

A pasta `data/` permanece fora do repositório. Para reproduzir o treinamento, é necessário obter a base autorizada e disponibilizar `data/students_tratados.csv`.

## Executar a aplicação no macOS

Na pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O arquivo `models/modelo_evasao.joblib` deve estar presente.

## Usar a aplicação

1. Envie um CSV tratado com as mesmas variáveis usadas no treinamento.
2. Utilize vírgula como delimitador e ponto como separador decimal.
3. Clique em **Realizar previsões**.
4. Confira as classificações e probabilidades.
5. Clique em **Baixar resultados**.

As colunas `Target` e `TargetBinario` não são necessárias para a previsão. Colunas adicionais são preservadas na exportação, mas não são usadas pelo modelo.

O arquivo exportado contém:

- `Previsao`: Graduado ou Desistente.
- `ProbabilidadeDesistencia`: probabilidade estimada da classe Desistente, entre 0 e 1.

O limiar de classificação é 0,5. As probabilidades são estimativas do modelo, não certezas.

## Limitações

- Uso de informações dos dois primeiros semestres.
- Exclusão da classe Matriculado.
- Presença de valores ausentes, tratados pelo pipeline.
- Notas fora da escala de 0 a 20 foram tratadas como ausentes. A imputação pela mediana não recupera as notas originais e pode afetar os resultados.
- Necessidade de validação em outras instituições e períodos antes de generalizar os resultados.

## Links Entregáveis

Repositório GitHub:
https://github.com/liuiglesias/evasao-estudantil

Aplicação Streamlit:
https://liuiglesias-evasao-estudantil-app-7qkke9.streamlit.app/
