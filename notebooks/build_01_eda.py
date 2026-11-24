"""Génère notebooks/01_eda_julia.ipynb (sans sorties) à partir de cellules Python.

Exécution :  uv run python notebooks/build_01_eda.py
Puis      :  uv run jupyter nbconvert --to notebook --execute notebooks/01_eda_julia.ipynb \
                 --output-dir data/outputs --output 01_eda_julia_executed.ipynb
La version exécutée (qui contient des données) reste dans data/outputs/, gitignoré.
"""

from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("md", s))  # noqa: E731
code = lambda s: cells.append(("code", s))  # noqa: E731

md("""# 01 · Exploration des données (Julia)

Objectif : comprendre le panel matières + énergie, la BOM des deux produits, et le **nouveau benchmark de prix de vente RMU**
(illisible pour l'équipe le 17/11, débloqué le 18/11), pour préparer la prévision 3/6/9 mois et la stratégie SF6-free.

> ⚠️ Données sous NDA. Ce notebook est commité **sans sorties** ; la version exécutée est dans `data/outputs/` (gitignoré).""")

code("""import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from statsmodels.tsa.stattools import adfuller
from src import data as d

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid", context="notebook")
FIG = d.OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")

rm, en = d.load_rm_wide(), d.load_energy_wide()
cat_rm, cat_en = d.load_catalogs()
print("RM :", rm.shape, rm.index.min().date(), "→", rm.index.max().date())
print("Énergie :", en.shape)""")

md("## 1. Couverture et données manquantes")
code("""def coverage(df, cat):
    rows = []
    for c in df.columns:
        s = df[c]
        obs = s.dropna()
        rows.append({"series": c,
                     "name": cat["series_name"].get(c, "") if "series_name" in cat else "",
                     "unit": cat["unit"].get(c, ""),
                     "first": obs.index.min().strftime("%Y-%m") if len(obs) else None,
                     "last": obs.index.max().strftime("%Y-%m") if len(obs) else None,
                     "n_obs": len(obs),
                     "gaps_inside": int(s.loc[obs.index.min():obs.index.max()].isna().sum()) if len(obs) else None})
    return pd.DataFrame(rows).set_index("series")

cov = pd.concat([coverage(rm, cat_rm), coverage(en, cat_en)])
cov""")
code("""fig, ax = plt.subplots(figsize=(12, 9))
allp = pd.concat([rm, en], axis=1)
sns.heatmap(allp.notna().T, cbar=False, cmap=["#e5e5e5", "#2b6cb0"], ax=ax,
            xticklabels=12, yticklabels=True)
ax.set_xticklabels([allp.index[int(t.get_text())].strftime("%Y-%m") if t.get_text().isdigit() else t.get_text()
                    for t in ax.get_xticklabels()], rotation=45)
ax.set_title("Mois observés (bleu) vs manquants (gris)"); save(fig, "01_coverage")""")
md("""**Lecture.** Aucun trou à l'intérieur des historiques ; seuls les 1–3 derniers mois manquent (délais de publication).
Les L01–L08 (CRU/MEPS/ICIS sous licence) sont vides volontairement. → À l'origine 2026-09, les PPI Eurostat
ne sont en réalité connus que jusqu'à ~2026-07 : il faut **nowcaster** les derniers mois, pas les imputer.""")

md("## 2. Dynamique des prix (base 100 = nov. 2016)")
code("""T = d.RM_TARGETS
reb = rm[T] / rm[T].iloc[0] * 100
fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
for ax, s in zip(axes.flat, T):
    ax.plot(reb.index, reb[s], lw=1.4)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), color="grey", alpha=.12)
    ax.set_title(f"{s} · {cat_rm.loc[s, 'material']} · {cat_rm.loc[s, 'region']}", fontsize=9)
for ax in axes.flat[len(T):]: ax.axis("off")
fig.suptitle("Prix et indices matières, base nov. 2016 = 100 (gris : choc 2021-22)"); fig.tight_layout(); save(fig, "02_rebased")""")
code("""r = d.dlog(rm[T])
stats = pd.DataFrame({
    "vol_annual_%": r.std() * np.sqrt(12) * 100,
    "yoy_last_%": (rm[T].ffill().iloc[-1] / rm[T].ffill().shift(12).iloc[-1] - 1) * 100,
    "peak_month": rm[T].idxmax().dt.strftime("%Y-%m"),
    "last_vs_peak_%": (rm[T].ffill().iloc[-1] / rm[T].max() - 1) * 100,
    "AC1_dlog": r.apply(lambda s: s.autocorr(1)),
}).round(2).sort_values("vol_annual_%", ascending=False)
stats""")
md("""**Lecture.** Les prix cotés (LME, minerai) sont 2 à 4× plus volatils que les PPI Eurostat/BLS, et ces derniers ont une
**forte autocorrélation des variations (momentum)** → les PPI sont structurellement plus prévisibles que les prix de marché.""")

md("## 3. Stationnarité et saisonnalité")
code("""def adf_p(s):
    s = s.dropna()
    return adfuller(s, autolag="AIC")[1] if len(s) > 24 else np.nan
st = pd.DataFrame({"ADF_p_log_level": np.log(rm[T]).apply(adf_p),
                   "ADF_p_dlog": r.apply(adf_p)})
# Saisonnalité : part de variance des variations expliquée par le mois calendaire
st["seasonal_R2"] = [r[s].groupby(r.index.month).transform("mean").var() / r[s].var() for s in T]
st.round(3)""")
md("""**Lecture.** Niveaux en log non stationnaires, variations stationnaires → modéliser les **log-variations à h mois**.
Saisonnalité faible (R² mensuel bas) : pas de terme saisonnier nécessaire, à reconfirmer série par série.""")

md("## 4. Qui bouge avant qui ? (lead-lag sur variations mensuelles)")
code("""drivers = {"Brent EUR (ED01)": en["ED01"], "TTF EUR/MWh (ED02)": en["ED02"], "Power DE (E09)": en["E09"],
           "Power HU (E11)": en["E11"], "Energy PPI EU (E13)": en["E13"], "Iron ore (S04)": rm["S04"],
           "Nickel (S03)": rm["S03"], "Zinc (S05)": rm["S05"], "Copper EUR (D01)": rm["D01"], "Alu EUR (D02)": rm["D02"]}
followers = ["S07", "S08", "S09", "S10", "S11", "S12", "S13"]
rows = []
for dn, ds in drivers.items():
    x = d.dlog(ds)
    for f in followers:
        y = d.dlog(rm[f])
        cors = {k: y.corr(x.shift(k)) for k in range(0, 7)}
        k_best = max(cors, key=lambda k: abs(cors[k]))
        rows.append({"driver": dn, "follower": f, "best_lag_m": k_best, "corr": cors[k_best], **{f"k{k}": v for k, v in cors.items()}})
ll = pd.DataFrame(rows)
piv = ll.pivot(index="driver", columns="follower", values="corr")
lag = ll.pivot(index="driver", columns="follower", values="best_lag_m")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(piv, annot=piv.round(2).astype(str) + "\\n(" + lag.astype(int).astype(str) + "m)", fmt="", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Corrélation max des log-variations mensuelles (driver décalé de k mois → suiveur)"); save(fig, "04_leadlag")
ll.sort_values("corr", key=abs, ascending=False).head(15).round(2)""")
md("""**Lecture.** Les meilleures relations avec k ≥ 1 sont exploitables comme features (connues au moment de la prévision).
Attention au k = 0 : utile pour le *nowcast* des mois non publiés, mais pas comme feature future.""")

md("## 5. Exposition des deux produits (BOM × prix)")
code("""bom = pd.concat([d.load_bom("csv", allow_deprecated=True), d.load_bom("root")])
bom.pivot_table(index=["product", "material_family"], columns="bom_version", values="mass_kg").round(1)""")
md("""Deux versions de BOM en conflit (data.zip vs xlsx racine) ; RM6 : la part non étiquetée (~45 %) est supposée acier carbone.
Les matières suivies par indice sont converties en EUR/kg via un **prix d'ancrage 2021** (`src/data.py`, hypothèse à
remplacer par les niveaux CRU/MEPS).""")
code("""costs = {v: d.bom_cost(rm.ffill(), d.load_bom(v, allow_deprecated=True)) for v in ["csv", "root"]}
tot = pd.concat({v: c.groupby(["month", "product"]).cost_eur.sum().unstack() for v, c in costs.items()}, axis=1)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, p in zip(axes, ["RM6", "AirSeT"]):
    c = costs["csv"].query("product == @p").pivot(index="month", columns="material_family", values="cost_eur")
    ax.stackplot(c.index, c.T.values, labels=c.columns, alpha=.85)
    ax.plot(tot.index, tot[("root", p)], "k--", lw=1, label="total (BOM racine)")
    ax.set_title(f"{p} : coût matières pricé (EUR/unité, BOM data.zip)"); ax.legend(fontsize=7, loc="upper left")
save(fig, "05_bom_cost")
last = costs["csv"][costs["csv"].month == rm.ffill().index[-2]]
share = last.pivot(index="material_family", columns="product", values="cost_eur")
(share / share.sum() * 100).round(1).rename(columns=lambda c: c + " %coût")""")
code("""# Contribution de chaque matière à la variance des variations mensuelles du coût (où est le risque ?)
def var_contrib(p, v="csv"):
    c = costs[v].query("product == @p").pivot(index="month", columns="material_family", values="cost_eur")
    dc = c.diff().dropna(); tot_d = dc.sum(axis=1)
    return (dc.apply(lambda s: s.cov(tot_d)) / tot_d.var() * 100).round(1)
pd.DataFrame({p: var_contrib(p) for p in ["RM6", "AirSeT"]})""")
md("""**Lecture.** RM6 = produit « cuivre » ; AirSeT = produit « acier ». Le risque matières des deux offres ne vient pas des
mêmes marchés → la règle de devis ETO doit être **spécifique au produit** (et l'écart de coût AirSeT − RM6 se resserre quand le cuivre monte).""")

md("## 6. 🆕 Benchmark des prix de vente RMU (2017–2026, synthétique)")
code("""bm = d.load_rmu_price_benchmark()
p50 = bm.pivot(index="Year", columns="Brand", values="P50")
fig, ax = plt.subplots(figsize=(11, 5))
for b in p50.columns:
    ax.plot(p50.index, p50[b], marker="o", lw=2.5 if b in ("Schneider Electric", "Eaton") else 1.2, label=b)
ax.axvline(2025.5, color="red", ls=":", label="marqueur SF6 (fichier)")
ax.set_title("Prix de vente P50 d'un RMU 3 fonctions par marque (EUR, synthétique)"); ax.legend(); save(fig, "06_rmu_prices")
yoy = bm.pivot(index="Year", columns="Brand", values="YoY P50").loc[2026] * 100
pd.DataFrame({"P50 2025": p50.loc[2025], "P50 2026": p50.loc[2026], "hausse 2026 %": yoy.round(1),
              "écart vs SE 2026 %": ((p50.loc[2026] / p50.loc[2026, "Schneider Electric"] - 1) * 100).round(1)})""")
code("""# Largeur de bande (P90-P10)/P50 : dispersion des prix de marché par marque
band = bm.assign(band_pct=(bm.P90 - bm.P10) / bm.P50 * 100).pivot(index="Year", columns="Brand", values="band_pct").round(1)
band""")
code("""# Poids des matières pricées dans le prix de vente Schneider (ordre de grandeur)
annual_cost = tot["csv"].groupby(tot.index.year).mean()
se = p50["Schneider Electric"]
pd.DataFrame({"coût matières RM6 (EUR)": annual_cost["RM6"], "prix P50 SE (EUR)": se,
              "part matières %": annual_cost["RM6"] / se * 100}).dropna().round(1)""")
md("""**Lecture (à valider avec Schneider : quel produit, quelle région ?).** *(chiffres : voir les sorties ci-dessus,
non reproduits ici car le fichier est sous NDA)*
- En **2026**, les fabricants **avec SF6** augmentent fortement leurs prix, l'offre **sans SF6** (Eaton) beaucoup moins.
  Le fichier place un marqueur SF6 mi-2025 : cohérent avec l'entrée en vigueur des restrictions UE sur le SF6 en MT.
- Conséquence : en 2026 l'offre sans SF6 devient **plus compétitive** que le RM6 → menace directe sur le RM6,
  et argument fort pour pousser **RM AirSeT** (sans SF6) avant que les concurrents ne le fassent hors UE.
- Les matières pricées ne pèsent qu'une **petite part du prix de vente** (hors « Other », main d'œuvre, frais) : c'est
  l'ordre de grandeur de la marge de sécurité à intégrer aux devis ETO (à recouper avec le should-cost).""")

md("## 7. Epoxy : contrôle de cohérence")
code("""ep = d.load_epoxy()
cols = ep.columns.tolist(); y = cols[-1]; X = cols[:-1]
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(ep[X], ep[y]); res = ep[y] - lr.predict(ep[X])
print("R² recette résine ~ intrants :", round(lr.score(ep[X], ep[y]), 3))
# Une régression ne voit pas l'erreur si la résine est CALCULÉE à partir de l'intrant faux :
# on compare plutôt chaque valeur à la moyenne de ses deux voisins.
neigh = (ep.shift(1) + ep.shift(-1)) / 2
jump = (ep / neigh - 1) * 100
flags = jump[(jump.abs() > 40).any(axis=1)].round(0)
flags""")
md("""**Lecture.** La résine époxy est quasiment une recette linéaire de ses intrants (R² ≈ 0,98). Les mois signalés s'écartent de
plus de 40 % de leurs deux voisins : erreurs de saisie probables (épichlorhydrine mai 2026, et la résine qui en découle)
→ à corriger avant toute prévision.""")

md("""## 8. Synthèse pour l'équipe
1. **Panel propre** (aucun trou interne) ; le vrai sujet est le *nowcast* des 1–3 derniers mois (délais Eurostat/BLS).
2. **Deux familles de cibles** : PPI lissés et prévisibles (momentum + drivers avancés) vs prix de marché ≈ marche aléatoire.
3. **Drivers avancés** utilisables comme features (énergie → plastiques/alu, minerai/nickel/zinc → aciers).
4. **Produits exposés différemment** : RM6 → cuivre, AirSeT → aciers. La règle de devis doit être par produit.
5. **🆕 Prix de vente** : saut 2026 des offres SF6, bien plus fort que celui de l'offre sans SF6 → pivot de la stratégie AirSeT.
6. **À trancher** : version de BOM, prix d'ancrage (CRU), produit/région exacts du benchmark de prix.""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("01_eda_julia.ipynb")
nbf.write(nb, out)
print("écrit :", out)
