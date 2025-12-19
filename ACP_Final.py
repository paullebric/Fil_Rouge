import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========
# Chemins
# ==========
DATA_DIR = "Data"
TABLE_TSV = f"{DATA_DIR}/table.tsv"
TAX_TSV   = f"{DATA_DIR}/taxonomy.tsv"
META_TSV  = f"{DATA_DIR}/sample-metadata.tsv"

# ==========
# Paramètres
# ==========
GROUP_COL = "env_material"   # Bulk soil / rhizosphere
RANK = "p__"                 # "p__" (phylum) ou "g__" (genus)
TOP_N = 15                   # barplot: top taxa
PSEUDOCOUNT = 1e-6           # CLR pseudocount

# -----------------------
# Fonctions taxonomie
# -----------------------
def get_rank(taxon_str: str, rank_prefix: str = "p__") -> str:
    if pd.isna(taxon_str) or taxon_str is None:
        return "Unassigned"
    parts = [p.strip() for p in str(taxon_str).split(";")]
    for p in parts:
        if p.startswith(rank_prefix):
            val = p.replace(rank_prefix, "").strip()
            return val if val else "Unassigned"
    return "Unassigned"

def pretty_asv_name(feature_id: str, taxon_str: str, rank_prefix: str = "p__") -> str:
    r = get_rank(taxon_str, rank_prefix)
    return f"{r} | {feature_id[:10]}"  # tronque l'ASV id pour lisibilité

# -----------------------
# 1) Lire table.tsv correctement
#   - saute la ligne "# Constructed from biom file"
#   - gère "#OTU ID" splitté en "#OTU" + "ID"
# -----------------------
table = pd.read_csv(TABLE_TSV, sep=r"\s+", engine="python", header=0, skiprows=1)

if len(table.columns) > 1 and table.columns[0] == "#OTU" and table.columns[1] == "ID":
    table = table.rename(columns={"#OTU": "feature_id"}).drop(columns=["ID"])
else:
    table = table.rename(columns={table.columns[0]: "feature_id"})

table = table.set_index("feature_id")
table.columns = table.columns.str.strip()
table = table.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)  # counts

# -----------------------
# 2) Lire taxonomie + metadata
# -----------------------
tax = pd.read_csv(TAX_TSV, sep="\t", dtype=str)
tax = tax.rename(columns={"Feature ID": "feature_id", "Taxon": "taxon"}).set_index("feature_id")
tax["taxon"] = tax["taxon"].fillna("Unassigned")

meta = pd.read_csv(META_TSV, sep="\t", dtype=str)
meta = meta.rename(columns={"#SampleID": "sample_id"}).set_index("sample_id")
meta.index = meta.index.str.strip()

# -----------------------
# 3) Aligner samples
# -----------------------
common_samples = table.columns.intersection(meta.index)
print("Samples communs table/metadata =", len(common_samples))
if len(common_samples) == 0:
    raise ValueError("Toujours aucun SampleID commun. Vérifie que tu utilises la bonne metadata (Run SRR...).")

table = table[common_samples]
meta = meta.loc[common_samples].copy()

# Aligner taxonomie (certains ASV peuvent manquer)
tax = tax.reindex(table.index)
tax["taxon"] = tax["taxon"].fillna("Unassigned")

# -----------------------
# 4) (Recommandé) filtrer mitochondries/chloroplastes
# -----------------------
mask_bad = tax["taxon"].str.contains("Mitochondria", na=False) | tax["taxon"].str.contains("Chloroplast", na=False)
if mask_bad.any():
    print("Filtrage mito/chloro:", int(mask_bad.sum()), "ASV supprimées")
    table = table.loc[~mask_bad]
    tax = tax.loc[~mask_bad]

# Comptage précis des catégories problématiques
mask_mito = tax["taxon"].str.contains("Mitochondria", case=False, na=False)
mask_chloro = tax["taxon"].str.contains("Chloroplast", case=False, na=False)
mask_euk = tax["taxon"].str.contains("Eukaryota", na=False)
mask_archaea = tax["taxon"].str.contains("Archaea", na=False)
mask_bacteria = tax["taxon"].str.contains("Bacteria", na=False)
# =======================
# FILTRAGE FINAL "CLEAN"
# =======================

mask_bacteria = tax["taxon"].str.contains("Bacteria", na=False)

table_clean = table.loc[mask_bacteria]
tax_clean = tax.loc[mask_bacteria]
reads_per_sample = table_clean.sum(axis=0)

# filtrage samples
good_samples = reads_per_sample[reads_per_sample >= 1000].index

table_filt_samples = table_clean[good_samples]
metadata_filt = meta.loc[good_samples]

print("Samples conservés :", table_filt_samples.shape[1])
reads_stats = table_filt_samples.sum(axis=0)

print(reads_stats.describe())
import numpy as np

table_rel = table_filt_samples.div(table_filt_samples.sum(axis=0), axis=1)
table_log = np.log10(table_rel + 1e-6)


X = table_log.T  # samples x ASV

pca = PCA(n_components=2)
coords = pca.fit_transform(X)


pca_df = pd.DataFrame(
    coords,
    index=X.index,
    columns=["PC1", "PC2"]
)

pca_df = pca_df.join(metadata_filt)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="env_material",
    alpha=0.8
)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("ACP – communautés bactériennes")
plt.legend(title="Environnement")
plt.tight_layout()
plt.show()

# =======================
# BARPLOT PHYLUM (FIX)
# =======================

tax_final = tax_clean.copy()

# extraire phylum
tax_final["phylum"] = tax_final["taxon"].str.extract(r"p__([^;]+)")
tax_final["phylum"] = tax_final["phylum"].fillna("Unassigned")

# table phylum
table_phylum = table_filt_samples.groupby(tax_final["phylum"]).sum()

# abondances relatives
table_phylum_rel = table_phylum.div(table_phylum.sum(axis=0), axis=1)
# retirer phylum non assigné
table_phylum_rel = table_phylum_rel.drop(index="Unassigned", errors="ignore")

# top 10 phyla
top_phyla = (
    table_phylum_rel
    .mean(axis=1)
    .sort_values(ascending=False)
    .head(10)
    .index
)

# ⚠️ NE GARDER QUE LES PHYLA
bar_df = table_phylum_rel.loc[top_phyla].T
bar_df["env_material"] = metadata_filt["env_material"]

# melt uniquement les phyla
bar_df_melt = bar_df.reset_index().melt(
    id_vars=["index", "env_material"],
    value_vars=top_phyla,
    var_name="Phylum",
    value_name="Abundance"
)

plt.figure(figsize=(10,5))
sns.barplot(
    data=bar_df_melt,
    x="Phylum",
    y="Abundance",
    hue="env_material",
    estimator="mean",
    errorbar=None
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Abondance relative moyenne")
plt.title("Top 10 phyla bactériens")
plt.tight_layout()
plt.show()


print(
    f"ACP réalisée sur {table_filt_samples.shape[1]} samples, "
    f"{table_filt_samples.shape[0]} ASV bactériennes "
    f"(≥1000 reads/sample, mito/chloro exclus)"
)

print(type(tax))
print(tax.shape)
tax.head()
print(tax.head(10))
# combien de taxons contiennent p__ ?
(tax_clean["taxon"].str.contains("p__", na=False).mean()*100)

# afficher 20 taxons au hasard qui n'ont PAS p__
tax_clean.loc[~tax_clean["taxon"].str.contains("p__", na=False), "taxon"].sample(20, random_state=0)
 