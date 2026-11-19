"""Génère notebooks/02_testbench_results.ipynb (sans sorties).

    uv run python notebooks/build_02_testbench.py
    uv run python -m src.testbench.run                    # (et --mode realtime) produit les résultats lus ici
    uv run jupyter nbconvert --to notebook --execute notebooks/02_testbench_results.ipynb \
        --output-dir data/outputs --output 02_testbench_results_executed.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("md", s))  # noqa: E731
code = lambda s: cells.append(("code", s))  # noqa: E731

md("""# 02 · Banc de test des prévisions : résultats des baselines

**Question posée par la partie 2 :** une prévision est-elle meilleure que « le prix reste le même », et sa
fourchette P10–P90 contient-elle vraiment la réalité 8 fois sur 10 ?

**Méthode.** Origines glissantes mensuelles de janv. 2020 à aujourd'hui ; à chaque origine, chaque modèle ne voit
que les données connues à cette date (garde-fou testé automatiquement), prévoit 3/6/9 mois, et on compare avec la
réalité. Deux modes : *submission* (l'origine = dernier mois publié, comme le CSV) et *realtime* (on retire en plus les
délais de publication réels, ~2 mois pour Eurostat). Toutes les erreurs sont en log (≈ %).

> ⚠️ NDA : notebook commité sans sorties ; la version exécutée est dans `data/outputs/` (gitignoré).""")

code("""import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src import data as d
from src.testbench import metrics
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")
FIG = d.OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
R = {m: d.OUT / "testbench" / m for m in ["submission", "realtime"]}
fc = pd.read_parquet(R["submission"] / "forecasts.parquet")
scored = metrics.score_rows(fc)
table = pd.read_csv(R["submission"] / "scores.csv")
MODELS = ["naive", "naive_gauss", "naive_vol", "drift_conf", "ar1_conf", "ma12_conf"]
print(fc.shape, "prévisions ;", scored.shape[0], "avec réalité connue ;", fc.origin.nunique(), "origines")""")

md("""## 1. Vue d'ensemble (mode submission)
`rel_MAE` < 1 : meilleur que le naïf. `series_sig_better_5pct` : séries où l'amélioration est significative (Diebold-Mariano).
`coverage` cible = 0,80 ; `below`/`above` cible = 0,10 chacun.""")
code("""ov = pd.read_csv(R["submission"] / "overview.csv").set_index(["model", "horizon"]).loc[MODELS]
ov""")

md("## 2. Qui bat le naïf, série par série ? (erreur relative du P50)")
code("""fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
for ax, h in zip(axes, [3, 6, 9]):
    piv = table[table.horizon == h].pivot(index="series", columns="model", values="rel_MAE")[MODELS[2:]]
    sig = table[table.horizon == h].pivot(index="series", columns="model", values="DM_p")[MODELS[2:]]
    ann = piv.round(2).astype(str) + np.where((sig < 0.05) & (piv < 1), "*", "")
    sns.heatmap(piv, annot=ann, fmt="", cmap="RdBu_r", center=1, vmin=0.6, vmax=1.4, ax=ax, cbar=h == 9)
    ax.set_title(f"h = {h} mois"); ax.set_xlabel("")
fig.suptitle("MAE du modèle / MAE du naïf (bleu = mieux ; * = significatif à 5 %)"); save(fig, "tb_relmae_heatmap")""")
md("""**Lecture.**
- Le **momentum (AR1)** bat le naïf sur les **indices Eurostat** (aluminium S08, acier S09, galvanisé S10), surtout à 3 mois :
  ces indices sont lissés et suivent le marché avec retard, donc leur variation récente se prolonge.
- Sur les **prix de marché** (LME, minerai) **rien ne bat le naïf** : cohérent avec un marché efficient.
- Le « retour à la moyenne » (ma12) est toujours pire → les prix ne reviennent pas vers leur moyenne à 3–9 mois.
- Avec ~80 origines qui se chevauchent, la significativité est rare : il faut le dire honnêtement.""")

md("## 3. La fourchette P10–P90 est-elle fiable ?")
code("""reg = metrics.summarize(scored, by=("model", "regime", "horizon")).reset_index()
reg = reg[reg.model.isin(["naive", "naive_gauss", "naive_vol", "ar1_conf"])]
g = sns.catplot(data=reg, x="horizon", y="coverage", hue="model", col="regime", kind="bar", height=4, aspect=1.2)
for ax in g.axes.flat: ax.axhline(0.8, color="k", ls="--", lw=1)
g.fig.suptitle("Couverture P10–P90 (cible 80 %) : période calme vs choc 2021-22", y=1.03); save(g.fig, "tb_coverage_regime")
reg.pivot_table(index=["model", "horizon"], columns="regime", values=["coverage", "below", "above"]).round(2)""")
md("""**Lecture.**
- En période calme, toutes les fourchettes sont correctes (≈ 80 %+).
- **Pendant le choc 2021-22, les fourchettes classiques s'effondrent (~45 % de couverture)** et les ratés sont
  surtout **à la hausse** : exactement le cas où la marge d'un devis à prix fixe est détruite.
- La version **`naive_vol`** (bandes qui s'élargissent dès que la volatilité monte) remonte la couverture du choc à ~70 %+
  et atteint ~76–77 % sur l'ensemble 2020-26 (crise incluse), avec la meilleure perte quantile → **c'est la référence
  d'incertitude à battre.**
- Reste un biais : trop de dépassements au-dessus du P90 (~15–19 %) et trop peu sous le P10 (~4–6 %). Pour Schneider,
  c'est le côté qui compte : le **P90 est encore un peu optimiste** → la marge de sécurité doit en tenir compte.""")

md("## 4. Exemple : cuivre et acier, fourchettes à 6 mois dans le temps")
code("""fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, s in zip(axes, ["D01", "S09"]):
    for m, c in [("naive", "tab:grey"), ("naive_vol", "tab:blue")]:
        x = fc[(fc.model == m) & (fc.series == s) & (fc.horizon == 6)].set_index("target_month").sort_index()
        ax.fill_between(x.index, x.p10, x.p90, alpha=.25, color=c, label=f"{m} P10–P90")
    truth = fc[(fc.model == "naive") & (fc.series == s) & (fc.horizon == 6)].set_index("target_month").y_true
    ax.plot(truth.index, truth, "k", lw=1.5, label="réalité")
    ax.set_title(f"{s} : fourchette prévue 6 mois avant vs réalité"); ax.legend(loc="upper left")
save(fig, "tb_fan_examples")""")

md("## 5. Indices de coût BOM (lignes `bom_index` du CSV)")
code("""bom = table[table.series.isin(["RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV_IEC"]) & table.model.isin(["naive", "naive_vol", "ar1_conf"])]
bom.set_index(["series", "model", "horizon"])[["n", "MAE_pct", "rel_MAE", "DM_p", "coverage", "below", "above", "width_pct"]]""")
md("""**Lecture.** Erreur typique du naïf sur le coût matières : quelques % à 3 mois, davantage à 9 mois — c'est l'ordre de
grandeur de la « marge à risque » d'un devis à prix fixe (partie 4). Les poids viennent de `config/assumptions.yaml`
(BOM officielle xlsx_v1, pondération en valeur, base = origine = 100) ; epoxy et électronique pas encore inclus.""")

md("## 6. Robustesse : mode realtime (délais de publication)")
code("""rt = pd.read_csv(R["realtime"] / "overview.csv").set_index(["model", "horizon"]).loc[MODELS]
pd.concat({"submission": ov[["rel_MAE_median", "series_beating_naive", "coverage_mean"]],
           "realtime": rt[["rel_MAE_median", "series_beating_naive", "coverage_mean"]]}, axis=1)""")
md("""**Lecture.** Avec les vrais délais (les indices Eurostat ont ~2 mois de retard), une prévision « à 3 mois » est en fait
à ~5 mois de la dernière donnée : l'avance du momentum se réduit. Les conclusions restent les mêmes.""")

md("""## 6bis. Choix du modèle sur période gelée (2020-23 → 2024-26)
Le meilleur modèle par série × horizon est choisi uniquement sur les prévisions dont la cible est avant 2024, puis noté
sur les origines 2024-26. C'est le score honnête à montrer au jury.""")
code("""sel = pd.read_csv(R["submission"] / "holdout_selection.csv").set_index(["series", "horizon"])
display(sel.groupby("horizon")[["test_rel_MAE", "test_rel_QL", "test_coverage"]].median())
sel""")
code("""pd.read_csv(R["submission"] / "compute_footprint.csv")  # temps de calcul, kWh, gCO2e (estimation)""")

md("""## 7. Ce que ça implique
1. **Référence à battre** pour la partie 3 : `naive` pour le P50, `naive_vol` pour les fourchettes.
2. **Indices Eurostat** : il y a du signal (momentum, drivers avancés) → c'est là que les modèles de la partie 3 peuvent gagner.
3. **Prix de marché** : ne pas promettre mieux que le naïf ; mettre l'effort dans la **fourchette** (régimes de volatilité).
4. **Pour la partie 4** : dimensionner la marge de sécurité sur un P90 **ajusté à la volatilité du moment**, pas sur l'historique moyen.
5. CSV filet de sécurité (`naive_vol`) déjà produit et validé : `data/outputs/testbench/draft_forecast_202609_naive_vol.csv`.""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("02_testbench_results.ipynb")
nbf.write(nb, out)
print("écrit :", out)
