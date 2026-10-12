import pandas as pd #disponibiliza a biblioteca de manipulação de dados.
 
# ler planilha
df = pd.read_excel("data/StudentsPrepared.xlsx")

# Exibir informações iniciais
print("\n=== DIMENSÕES DA BASE ===")
print(df.shape) #informa a qtd de linhas e colunas

print("\n=== PRIMEIRAS 5 LINHAS ===")
print(df.head()) #apresenta os cinco primeiros registros

print("\n=== DISTRIBUIÇÃO DA VARIÁVEL TARGET ===")
print(df["Target"].value_counts()) #qtd de estudantes por categoria

# Verificar valores ausentes
print("\n=== VALORES AUSENTES ===")
print(df.isnull().sum()) #campos sem informação

# Verificar registros duplicados
print("\n=== REGISTROS DUPLICADOS ===")
print(df.duplicated().sum())