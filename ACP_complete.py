import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gmean
datas = "Data/table.tsv"
# =========================
# PARAMÈTRES À MODIFIER
# =========================
FEATURE_TABLE_TSV = datas   # chemin relatif si tu lances depuis ~/qiime2
METADATA_TSV = None  # ex: r"metadata.tsv" (optionnel)
METADATA_ID_COL = "sample-id"  # ou "SampleID" selon ton fichier
COLOR_BY = None  # ex: "treatment" (colonne metadata) si tu en as

# Filtrage simple (recommandé)
MIN_TOTAL_COUNT_PER_ASV = 10   # retire ASV ultra rares
MIN_PREVALENCE = 0.02          # au moins 2% des samples (0.02 * N)
PSEUDOCOUNT = 1.0              # pseudo-count pour CLR

# =========================
# 1) Charger la table ASV
# =========================
# QIIME2 export TSV: première ligne = "# Constructed from biom file"
# puis header avec "Feature ID"
asv = pd.read_csv(FEATURE_TABLE_TSV, sep="\t", comment="#", index_col=0)

# asv: rows = ASV, cols = samples
print("ASV table shape (features x samples):", asv.shape)

# =========================
# 2) Filtrages basiques
# =========================
# Filtre sur abondance totale
asv = asv.loc[asv.sum(axis=1) >= MIN_TOTAL_COUNT_PER_ASV]

# Filtre de prévalence (nombre de samples où ASV > 0)
prev = (asv > 0).sum(axis=1) / asv.shape[1]
asv = asv.loc[prev >= MIN_PREVALENCE]

print("After filtering shape:", asv.shape)

# =========================
# 3) CLR transform (compositional)
# =========================
# CLR s'applique par sample: CLR(x) = log(x/gmean(x))
# on ajoute un pseudocount pour éviter log(0)
asv_pc = asv + PSEUDOCOUNT

# transposer: samples x features
X = asv_pc.T

# geometric mean par ligne (par sample)
gm = X.apply(gmean, axis=1)
X_clr = np.log(X.div(gm, axis=0))

print("CLR matrix shape (samples x features):", X_clr.shape)

# =========================
# 4) PCA
# =========================
pca = PCA(n_components=10, random_state=0)
coords = pca.fit_transform(X_clr.values)

expl = pca.explained_variance_ratio_ * 100

# =========================
# 5) Charger metadata (optionnel)
# =========================
meta = None
if METADATA_TSV is not None:
    meta = pd.read_csv(METADATA_TSV, sep="\t")
    # rendre l'ID index
    meta = meta.set_index(METADATA_ID_COL)
    # aligner ordre des samples
    meta = meta.loc[X_clr.index]
    print("Metadata shape:", meta.shape)

# =========================
# 6) PLOTS
# =========================

def scatter_pc(pc_x, pc_y, title):
    plt.figure(figsize=(7,6))
    if meta is not None and COLOR_BY is not None:
        groups = meta[COLOR_BY].astype(str)
        for g in sorted(groups.unique()):
            mask = groups == g
            plt.scatter(coords[mask, pc_x], coords[mask, pc_y], label=g, alpha=0.75)
        plt.legend(title=COLOR_BY, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(coords[:, pc_x], coords[:, pc_y], alpha=0.75)

    plt.xlabel(f"PC{pc_x+1} ({expl[pc_x]:.1f}%)")
    plt.ylabel(f"PC{pc_y+1} ({expl[pc_y]:.1f}%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# (A) Scree plot
plt.figure(figsize=(7,4))
plt.plot(range(1, len(expl)+1), expl, marker="o")
plt.xticks(range(1, len(expl)+1))
plt.xlabel("Principal component")
plt.ylabel("Explained variance (%)")
plt.title("Scree plot (explained variance)")
plt.tight_layout()
plt.show()

# (B) PC1 vs PC2
scatter_pc(0, 1, "PCA on CLR-transformed ASV table (PC1 vs PC2)")

# (C) PC1 vs PC3
scatter_pc(0, 2, "PCA on CLR-transformed ASV table (PC1 vs PC3)")

# =========================
# 7) LOADINGS (taxa/features qui expliquent les axes)
# =========================
loadings = pd.DataFrame(
    pca.components_.T,
    index=X_clr.columns,  # ASV IDs
    columns=[f"PC{i}" for i in range(1, pca.n_components_+1)]
)

def top_loadings(pc="PC1", n=20):
    s = loadings[pc].abs().sort_values(ascending=False).head(n)
    return pd.DataFrame({
        "feature": s.index,
        "abs_loading": s.values,
        "signed_loading": loadings.loc[s.index, pc].values
    })

print("\nTop features driving PC1:")
print(top_loadings("PC1", 20).to_string(index=False))

print("\nTop features driving PC2:")
print(top_loadings("PC2", 20).to_string(index=False))

# (D) Barplot top loadings PC1
top_pc1 = top_loadings("PC1", 20)
plt.figure(figsize=(8,5))
plt.barh(top_pc1["feature"][::-1], top_pc1["signed_loading"][::-1])
plt.xlabel("Loading (signed)")
plt.title("Top 20 features contributing to PC1")
plt.tight_layout()
plt.show()

# (E) Barplot top loadings PC2
top_pc2 = top_loadings("PC2", 20)
plt.figure(figsize=(8,5))
plt.barh(top_pc2["feature"][::-1], top_pc2["signed_loading"][::-1])
plt.xlabel("Loading (signed)")
plt.title("Top 20 features contributing to PC2")
plt.tight_layout()
plt.show()

# =========================
# 8) BIPLOT léger (flèches pour top features)
# =========================
def biplot(pc_x=0, pc_y=1, n_arrows=10):
    plt.figure(figsize=(7,6))
    plt.scatter(coords[:, pc_x], coords[:, pc_y], alpha=0.6)

    # top features sur ces PCs
    pcx = f"PC{pc_x+1}"
    pcy = f"PC{pc_y+1}"
    scores = (loadings[pcx]**2 + loadings[pcy]**2).sort_values(ascending=False).head(n_arrows)

    # scale pour rendre visible
    scale_x = coords[:, pc_x].std()
    scale_y = coords[:, pc_y].std()

    for feat in scores.index:
        lx = loadings.loc[feat, pcx] * scale_x * 3
        ly = loadings.loc[feat, pcy] * scale_y * 3
        plt.arrow(0, 0, lx, ly, head_width=0.2, length_includes_head=True)
        plt.text(lx*1.05, ly*1.05, feat, fontsize=8)

    plt.xlabel(f"PC{pc_x+1} ({expl[pc_x]:.1f}%)")
    plt.ylabel(f"PC{pc_y+1} ({expl[pc_y]:.1f}%)")
    plt.title(f"Biplot (top {n_arrows} features)")
    plt.tight_layout()
    plt.show()

biplot(0, 1, n_arrows=10)

# =========================
# 9) Export des coordonnées PCA (utile pour rapport)
# =========================
pc_df = pd.DataFrame(coords[:, :5], index=X_clr.index, columns=[f"PC{i}" for i in range(1, 6)])
pc_df.to_csv("pca_coordinates_PC1-5.tsv", sep="\t")
print("\nSaved: pca_coordinates_PC1-5.tsv")
