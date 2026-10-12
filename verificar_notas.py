import pandas as pd

# carregar a base após a limpeza inicial
df = pd.read_csv("data/students_tratados.csv")

# colunas que representam notas acadêmicas
colunas_notas = [
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares2SemestreGrau"
]

print("=== ANÁLISE DAS NOTAS ACADÊMICAS ===")

for coluna in colunas_notas:

    print(f"\nColuna: {coluna}")

    print("Valor mínimo:", df[coluna].min())
    print("Valor máximo:", df[coluna].max())

    print("Notas acima de 20:", (df[coluna] > 20).sum())

    print("Notas negativas:", (df[coluna] < 0).sum())

    print("\nCinco maiores valores:")
    print(df[coluna].nlargest(5).to_string(index=False))