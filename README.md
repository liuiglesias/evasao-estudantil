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

O tratamento de notas com valores inconsistentes envolveu uma hipótese de recuperação do separador decimal. Essa correção inferida é uma limitação da preparação dos dados e deve ser considerada na interpretação dos resultados.

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

Os resultados numéricos da execução serão registrados nesta documentação antes da entrega.

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
- Correção inferida de valores inconsistentes nas notas.
- Necessidade de validação em outras instituições e períodos antes de generalizar os resultados.

## Entrega

Os links do repositório, da aplicação publicada e do vídeo de apresentação serão reunidos em um arquivo `.txt`.