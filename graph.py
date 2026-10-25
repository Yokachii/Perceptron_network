import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Paramètres ----------------
FICHIER   = "./data_out/data2/resultats_courbes.csv"
DATASET   = None      # ex : "./datasets/spirale_dure.csv" ; None = le premier du fichier
NORM      = 1
RATIO     = 0.8
TAUX      = None      # ex : [0.05, 0.1, 0.3] ; None = tous
NEURONES  = None      # ex : [16, 32] ; None = tous
COUCHES   = None      # ex : [1, 2] ; None = tous
LISSAGE   = 5         # moyenne glissante sur 5 points (= 50 époques) ; 1 = pas de lissage
REFERENCE = None      # ex : 0.98 pour tracer une ligne de référence ; None = aucune

# ---------------- Chargement et filtres ----------------
df = pd.read_csv(FICHIER)

if DATASET is None:
    DATASET = df["dataset"].iloc[0]

masque = (df["dataset"] == DATASET) & (df["normalized"] == NORM) & (df["ratio"] == RATIO)
if TAUX     is not None: masque &= df["taux_initial"].isin(TAUX)
if NEURONES is not None: masque &= df["neurones"].isin(NEURONES)
if COUCHES  is not None: masque &= df["couches"].isin(COUCHES)

sub = df[masque].sort_values(["run", "epoque"]).copy()

if sub.empty:
    print("Aucune donnée trouvée pour ces filtres.")
    raise SystemExit

# Lissage par run (centré, pour ne pas décaler la courbe)
sub["acc_lisse"] = sub.groupby("run")["acc"].transform(
    lambda s: s.rolling(LISSAGE, min_periods=1, center=True).mean()
)

# Étiquette discrète pour le taux (légende lisible, ordre numérique)
sub["taux_lbl"] = sub["taux_initial"].map(lambda v: f"{v:g}")
ordre_taux = [f"{v:g}" for v in sorted(sub["taux_initial"].unique())]

nb_seeds = sub["seed"].nunique()
titre = f"{DATASET.split('/')[-1]} (ratio = {RATIO}, norm = {NORM})"

def decorer(g, titre_fig):
    g.set(ylim=(0.45, 1.01))
    g.set_axis_labels("Époque", "Accuracy (test)")
    g.set_titles(row_template="{row_name} couche(s) cachée(s)",
                 col_template="{col_name} neurones")
    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", alpha=0.6)
        if REFERENCE is not None:
            ax.axhline(REFERENCE, color="gray", linestyle=":", linewidth=1.5)
    g.figure.suptitle(titre_fig, y=1.02)

# ---------------- Figure 1 : moyenne ± écart-type sur les graines ----------------
g1 = sns.relplot(
    data=sub, x="epoque", y="acc_lisse",
    hue="taux_lbl", hue_order=ordre_taux, palette="viridis",
    col="neurones", row="couches",
    kind="line", errorbar="sd",          # ci="sd" si seaborn < 0.12
    linewidth=2.2, height=3.6, aspect=1.4,
    facet_kws={"margin_titles": True},
)
g1._legend.set_title("Taux initial")
decorer(g1, f"Accuracy moyenne ± écart-type sur {nb_seeds} graine(s) : {titre}")
g1.savefig("courbes_acc_moyenne.png", dpi=150, bbox_inches="tight")

# ---------------- Figure 2 : un trait par run (une couleur par taux) ----------------
g2 = sns.relplot(
    data=sub, x="epoque", y="acc_lisse",
    hue="taux_lbl", hue_order=ordre_taux, palette="viridis",
    units="run", estimator=None,
    col="neurones", row="couches",
    kind="line", linewidth=1.0, alpha=0.8, height=3.6, aspect=1.4,
    facet_kws={"margin_titles": True},
)
g2._legend.set_title("Taux initial")
decorer(g2, f"Chaque run séparément : {titre}")
g2.savefig("courbes_acc_par_run.png", dpi=150, bbox_inches="tight")

plt.show()