import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
datas = "Data/table.tsv"
# Charger la table
df = pd.read_csv(datas, sep="\t", comment="#", index_col=0)

# Transposer → samples x features
df = df.T

# Pseudo-count
df_pc = df + 1

# CLR transform
geom_mean = np.exp(np.log(df_pc).mean(axis=1))
df_clr = np.log(df_pc.div(geom_mean, axis=0))

# ACP
pca = PCA(n_components=2)
coords = pca.fit_transform(df_clr)

# Plot
plt.figure(figsize=(6,5))
plt.scatter(coords[:,0], coords[:,1], alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA on CLR-transformed ASV table")
plt.show()
