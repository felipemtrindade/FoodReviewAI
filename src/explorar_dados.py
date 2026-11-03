import pandas as pd

df = pd.read_csv("data/comentarios.csv")

print(df.head())
print("\nDimensões:", df.shape)
print("\nColunas:", df.columns.tolist())
print("\nValores ausentes:")
print(df.isnull().sum())
print("\nDistribuição das avaliações:")
print(df["avaliacao"].value_counts().sort_index())

print("\nIdiomas:")
print(df["idiomaTexto"].value_counts())

print("\nExemplos de avaliações:")
print(df[["textoOriginal", "avaliacao"]].dropna().head(10).to_string(index=False))
print("\nAvaliações em português:")
df_pt = df[df["idiomaTexto"] == "pt"]

print(df_pt["avaliacao"].value_counts().sort_index())

print("\nExemplos por nota:")

for nota in sorted(df_pt["avaliacao"].unique()):
    print(f"\nNOTA {nota}")

    exemplos = df_pt[
        (df_pt["avaliacao"] == nota) &
        (df_pt["textoOriginal"].notna())
    ]

    for texto in exemplos["textoOriginal"].head(3):
        print("-", texto[:200])
        print("\nDistribuição por plataforma e nota:")

tabela = pd.crosstab(
    df["IDPlataforma"],
    df["avaliacao"]
)

print(tabela)

print("\nExemplos das notas 0 e 1 por plataforma:")

for plataforma in sorted(df["IDPlataforma"].unique()):
    print(f"\nPLATAFORMA {plataforma}")

    exemplos = df[
        (df["IDPlataforma"] == plataforma) &
        (df["avaliacao"].isin([0, 1])) &
        (df["idiomaTexto"] == "pt") &
        (df["textoOriginal"].notna())
    ]

    for _, linha in exemplos.head(2).iterrows():
        print("Nota:", linha["avaliacao"])
        print("Texto:", linha["textoOriginal"][:150])