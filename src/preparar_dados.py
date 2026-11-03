import pandas as pd

df = pd.read_csv("data/comentarios.csv")

df = df[df["IDPlataforma"].isin([1, 2, 3])]
df = df[df["idiomaTexto"] == "pt"]
df = df[df["avaliacao"].isin([1, 2, 4, 5])]

df = df[["textoOriginal", "avaliacao"]].dropna()

df["textoOriginal"] = df["textoOriginal"].str.strip()
df = df[df["textoOriginal"].str.len() > 0]

df["sentimento"] = (df["avaliacao"] >= 4).astype(int)

df = df.drop_duplicates(subset="textoOriginal")

df = df.rename(columns={"textoOriginal": "texto"})

df.to_csv("data/avaliacoes_tratadas.csv", index=False)

print("Total de avaliações:", len(df))
print("\nDistribuição dos sentimentos:")
print(df["sentimento"].value_counts())