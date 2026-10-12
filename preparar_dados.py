import pandas as pd

# carregar a base original
df = pd.read_excel("data/StudentsPrepared.xlsx")

print("=== BASE ORIGINAL ===")
print("Registros:", len(df))

# remover os registros duplicados
df = df.drop_duplicates().copy()

print("\n=== APÓS REMOVER DUPLICADOS ===")
print("Registros:", len(df))

# selecionar estudantes com resultado definitivo (graduado ou desistente)
df = df[df["Target"].isin(["Graduado", "Desistente"])].copy()

print("\n=== APÓS SELEÇÃO DAS CLASSES ===")
print(df["Target"].value_counts())

# criar variável-alvo binária (0 = graduado, 1 = desistente)
df["TargetBinario"] = df["Target"].map({
    "Graduado": 0,
    "Desistente": 1
})

## Identificar notas fora da escala acadêmica esperada --add apos verificacao
colunas_notas = [
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares2SemestreGrau"
]

for coluna in colunas_notas:

    valores_invalidos = (
        (df[coluna] < 0) |
        (df[coluna] > 20)
    )

    print(
        f"\n{coluna}: "
        f"{valores_invalidos.sum()} valores inválidos"
    )

    ## Substituir valores inválidos por ausentes
    ## A imputação será realizada posteriormente no pipeline
    df.loc[valores_invalidos, coluna] = float("nan")

### Engenharia de atributos: taxas de aprovação

for semestre in [1, 2]:

    coluna_inscritos = (
        f"UnidadesCurriculares{semestre}SemestreInscrito"
    )

    coluna_aprovados = (
        f"UnidadesCurriculares{semestre}SemestreAprovado"
    )

    nova_coluna = f"TaxaAprovacao{semestre}Semestre"

    ### Evitar divisão por zero
    df[nova_coluna] = (
        df[coluna_aprovados] /
        df[coluna_inscritos].replace(0, float("nan"))
    )

    print(f"\nAtributo criado: {nova_coluna}")

    print("Valores ausentes:", df[nova_coluna].isna().sum())

    print("Mínimo:", df[nova_coluna].min())
    print("Máximo:", df[nova_coluna].max())

############# salvar a base tratada
df.to_csv(
    "data/students_tratados.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== TRATAMENTO CONCLUÍDO ===")
print("Registros finais:", len(df))
print("Variável-alvo:")
print(df["TargetBinario"].value_counts())
print("Arquivo salvo: data/students_tratados.csv")