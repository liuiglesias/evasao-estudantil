import streamlit as st #criacao de interface
import pandas as pd
import joblib

from pathlib import Path

#configuração da página
st.set_page_config(
    page_title="Previsão de Evasão Estudantil",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Previsão de Evasão Estudantil")

st.write(
    "Aplicação de Machine Learning para estimar "
    "a probabilidade de desistência de estudantes."
)

st.info(
    "Modelo desenvolvido com Regressão Logística "
    "e informações acadêmicas dos dois primeiros semestres."
)

#carregando modelo
@st.cache_resource
def carregar_modelo():
    caminho = Path("models/modelo_evasao.joblib")
    return joblib.load(caminho)

modelo = carregar_modelo()

st.success("Modelo carregado com sucesso!")

#upload do arquivo
st.subheader("📂 Enviar dados dos estudantes")

arquivo = st.file_uploader(
    "Selecione um arquivo CSV",
    type=["csv"]
)

if arquivo is not None:

    df = pd.read_csv(arquivo)

    st.write("### Prévia dos dados")
    st.dataframe(df.head())

    if st.button("Realizar previsões"):

        colunas_esperadas = modelo.feature_names_in_.tolist()

        colunas_ausentes = [
            coluna
            for coluna in colunas_esperadas
            if coluna not in df.columns
        ]

        if colunas_ausentes:
            st.error(
                "O arquivo não contém todas as variáveis necessárias."
            )
            st.write("Colunas ausentes:", colunas_ausentes)

        else:

            dados = df[colunas_esperadas].copy()

            previsoes = modelo.predict(dados)

            probabilidades = modelo.predict_proba(dados)[:, 1]

            resultado = df.copy()

            resultado["Previsao"] = [
                "Desistente" if p == 1 else "Graduado"
                for p in previsoes
            ]

            resultado["ProbabilidadeDesistencia"] = probabilidades

            st.write("### Resultados das previsões")
            st.dataframe(resultado)

            csv = resultado.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "📥 Baixar resultados",
                data=csv,
                file_name="previsoes_evasao.csv",
                mime="text/csv"
            )