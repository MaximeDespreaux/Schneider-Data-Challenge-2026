# Banc de test des prévisions (partie 2)

> Ce document ne contient aucune donnée : il décrit le code. Les résultats chiffrés sont dans `data/outputs/` (gitignoré).

## Version durcie (backtest-hardening, nov. 2026) : à utiliser pour tout résultat montré au jury

```
uv run python -m src.testbench.pipeline --dev        # fenêtre DEV : origines 2010-01 → 2021-12, cibles < 2022-01
uv run python -m src.testbench.pipeline --canaries   # tests anti-fuite sur données réelles -> data/outputs/canaries/<commit>.json
uv run python -m src.testbench.pipeline --graded     # UNE seule fois : commit tagué, arbre propre, canaris OK
uv run python -m src.testbench.pipeline --submit     # CSV origine 2026-09 (wf_blend), à partir du run gradé
uv run python -m src.business.run_margin_replay      # Goal A : quelle politique de devis protège la marge ?
```

- **Ce qui est connu à chaque origine** : `config/availability.yaml` est la seule table des délais (mode `realtime`).
  En mode `submission`, chaque entrée est masquée du même nombre de mois qu'aujourd'hui (bord irrégulier du live).
  Les moyennes mensuelles de prix quotidiens (VIX, change, LME…) ont un délai de 1 en `realtime` (mois incomplet
  à mi-mois), contre 0 auparavant.
- **Indices BOM causaux** (`targets.make_bom_deriver`) : poids en coût (prix × masse, BOM v2) connus à l'origine,
  acier régional par usine (RM6 → MEPS France CR), composantes prolongées jusqu'à l'origine (indice = 100),
  historique chaîné quand une composante démarre plus tard. Plastiques / électronique : niveau EUR/kg de config.
- **Choix du modèle walk-forward** (`meta.py`) : à chaque origine, seulement les prévisions déjà réalisées ;
  `wf_best`, `wf_blend` (modèle soumis), `wf_median`. Plus de choix sur 2024-26 (déjà regardé).
- **Statistiques** (`stats.py`) : DM ordonné (HAC + HLN), Clark-West, DM groupé par famille + Holm, IC bootstrap
  par blocs, MDE par placebos, Kupiec / Christoffersen / Ljung-Box sur origines non chevauchantes.
- **Canaris** (`canaries.py`) : fuite plantée, cible permutée, identité de troncature, décalage d'un mois.
- **Protocole** (`protocol.py`) : pré-enregistrement `docs/backtest/preregistration.md`, garde « un seul run gradé »,
  `manifest.json` (hash du code, de la config et des données) dans chaque dossier de sortie.

Les commandes historiques ci-dessous (`src.testbench.run`) restent disponibles pour l'exploration.

## Ce que fait le banc (version d'origine)
On rejoue l'histoire mois par mois. À chaque **origine** (janv. 2020 → aujourd'hui), chaque modèle ne reçoit que ce
qui était connu à cette date, prévoit les mois **origine + 3, + 6 et + 9** (P10 / P50 / P90), puis on compare
avec la réalité. Tous les scores sont comparés à la référence **`naive`** : « le prix reste celui d'aujourd'hui ».

```
uv run python -m src.testbench.run                    # mode submission, origines 2020-01 → fin
uv run python -m src.testbench.run --mode realtime     # + délais de publication réels (Eurostat ≈ 2 mois)
uv run python -m src.testbench.run --start 2017-11     # échantillon complet
uv run pytest -q tests                                 # tests (données synthétiques, tournent en CI)
```
Sorties dans `data/outputs/testbench/<mode>/` : `forecasts.parquet`, `scores.csv`, `overview.csv`, `regimes.csv`,
`holdout_selection.csv` (choix du modèle sur 2020-23, vérifié sur 2024-26), `compute_footprint.csv` (temps, kWh, gCO2e).

## Choisir le modèle final sans tricher
`metrics.holdout_selection(scored, split="2024-01-01")` : pour chaque série × horizon, choisit le meilleur modèle en ne
regardant que les prévisions dont la cible est **avant** 2024, puis le note sur les origines **à partir de** 2024.
C'est ce score-là (pas celui du backtest complet) qu'il faut montrer au jury pour le modèle retenu.

## Calcul et CO2
`compute_footprint(fc.attrs["runtime_s"])` : temps CPU par modèle → kWh (portable ~30 W) → gCO2e (~0,25 kg/kWh, mix UE).
Hypothèses volontairement simples, à citer comme ordre de grandeur.

## Les deux modes
| Mode | À l'origine t, le modèle connaît… | Usage |
|---|---|---|
| `submission` | toutes les séries jusqu'à t inclus | même définition que le CSV (`origin_month` = dernier mois utilisé) |
| `realtime` | chaque série jusqu'à t − délai de publication | test de robustesse « vraie vie » (un acheteur en novembre n'a l'indice Eurostat que jusqu'en septembre) |

Les délais sont dans `config/availability.yaml` (une ligne par série, test qui vérifie que chaque colonne du panel en a une).

## Brancher un modèle (partie 3)
Un modèle est un objet avec un attribut `name` et une méthode :

```python
def forecast(self, y: pd.Series, steps: list[int], exog: pd.DataFrame | None = None) -> pd.DataFrame:
    # y     : historique de la série cible, mensuel, sans NaN, se termine à la dernière observation connue
    # steps : nb de mois après la dernière observation (≥ horizon si la série a du retard de publication)
    # exog  : tout le panel connu à l'origine (matières + énergie + indices BOM), déjà tronqué → pas de fuite possible
    # retour: DataFrame index = steps, colonnes p10, p50, p90 (en niveau), p10 ≤ p50 ≤ p90
```

Si le modèle ne donne qu'une valeur centrale, écrire `point(y, h) -> float` et l'envelopper :
`Conformal(MonModele())` → P10/P90 calculés à partir de ses propres erreurs passées (rapide uniquement :
le modèle est rejoué ~48 fois par appel). Pour un modèle lent (ex. Chronos-2), renvoyer directement ses quantiles.

Ensuite :
```python
from src.testbench.run import build_panel, DEFAULT_TARGETS
from src.testbench.backtest import run_backtest, default_origins
from src.testbench.models import Naive, NaiveVolScaled
from src.testbench import metrics

panel = build_panel()
fc = run_backtest(panel, DEFAULT_TARGETS, [Naive(), NaiveVolScaled(), MonModele()],
                  origins=default_origins(panel, "2020-01"))
table = metrics.summarize(metrics.score_rows(fc))
metrics.overview(table)
```
**Toujours inclure `Naive()`** : c'est la référence des colonnes `rel_*` et du test de Diebold-Mariano.

## Les scores (tous en log, ≈ %)
| Score | Définition | Bon si |
|---|---|---|
| `MAE_pct` | erreur absolue moyenne du P50 | bas |
| `rel_MAE` | MAE modèle / MAE naïf, sur les mêmes prévisions | < 1 |
| `DM_p` | p-value Diebold-Mariano vs naïf (variance HAC h−1 retards, correction HLN) | < 0,05 pour dire « significatif » |
| `QL`, `rel_QL` | perte quantile (pinball) moyenne P10/P50/P90, et relative au naïf | bas / < 1 |
| `coverage` | part des réalités entre P10 et P90 | ≈ 0,80 |
| `below` / `above` | part sous P10 / au-dessus de P90 | ≈ 0,10 chacun |
| `width_pct` | largeur moyenne P10–P90 | le plus étroit possible **à couverture égale** |
| `regime` | « choc 2021-22 » si l'origine ou la cible tombe en 2021-22, sinon « calme » | — |

## Baselines fournies
| Nom | Idée |
|---|---|
| `naive` | prix d'aujourd'hui ; bandes = quantiles des variations passées à h mois |
| `naive_gauss` | idem, bandes gaussiennes (contre-exemple : sous-estime les queues) |
| `naive_vol` | naïf + bandes remises à l'échelle de la volatilité récente (EWMA, demi-vie 3 mois) |
| `drift_conf` | naïf + tendance des 60 derniers mois, bandes conformal |
| `ar1_conf` | momentum : AR(1) sur les variations mensuelles, bandes conformal |
| `ma12_conf` | retour vers la moyenne 12 mois, bandes conformal |

## Cibles
Séries ajoutées (acier MEPS/CRU L01-L03, epoxy EP01, électronique EL01-EL03), BOM v2 et indice BOM : voir [data_fixes.md](data_fixes.md).
Les 14 séries matières du panel (`src/data.py::RM_TARGETS`) + 2 indices de coût BOM (`src/testbench/targets.py`) :
indice à quantités fixes, pondéré en valeur, reconstruit à chaque origine (base = origine = 100), BOM officielle
`xlsx_v1` (`RM_Prices_XHEC_Challenge.xlsx` ; le CSV `bom_materials.csv` est obsolète). Paramètres lus dans
`config/assumptions.yaml`, **local et gitignoré** car il contient des prix MEPS/CRU sous licence : copier
`config/assumptions.example.yaml` vers `config/assumptions.yaml` et remplir les valeurs `FILL_LOCALLY`.

## Format du CSV
`src/testbench/submission.py` : `from_forecasts(...)` convertit une sortie du banc au format Schneider,
`validate_submission(df)` liste les problèmes (colonnes, enums, `target_month = origin + h`, `p10 ≤ p50 ≤ p90`, doublons).

## Limites connues
- La « réalité » est la valeur révisée d'aujourd'hui, pas la première publication (Eurostat/BLS révisent 2–3 mois).
- ~80 origines qui se chevauchent : la significativité statistique est difficile à obtenir, surtout à 9 mois.
- La demi-vie de `naive_vol` a été choisie parmi {3, 6, 12} sur ce même backtest (léger biais de sélection).
- Testé et rejeté : `naive_vol` sans recentrage (garder l'asymétrie historique) ne corrige pas le P90 trop optimiste.

## Intégration continue
`.github/workflows/tests.yml` lance `uv run pytest` à chaque push. Les tests n'utilisent que des données synthétiques.

Modules métier (pricing objectif A, objectif B, décarbonation) : voir [business.md](business.md).

## Modélisation complémentaire (24/11)
- **Modèle ML** : `src/testbench/ml_models.py::LGBMQuantile` (LightGBM quantile 10/50/90 %, dynamique propre de la
  série). Scoré sur la vérification 2024-26 de `run_champion` à côté du champion. **Règle fixée avant les résultats**
  (`run_champion.ml_replaces_champion`, même règle que champion v1/v2) : il remplace le champion seulement si sa rel_QL
  médiane est plus basse ET sa couverture médiane ≥ celle du champion − 0,02. Si c'est le cas, ses prévisions sont
  écrites sous le nom `champion` pour que le business suive automatiquement. Décision : `champion_vs_lgbm_q.csv`.
- **Ljung-Box** sur les erreurs du P50 (origines non chevauchantes) : `metrics.ljung_box_table`, sortie
  `champion_ljungbox.csv`. Kupiec et ETS amorti sont dans le pipeline renforcé (`src/testbench/pipeline.py`).
- **KPSS + STL** (stationnarité / saisonnalité) : section 3 du notebook d'exploration `notebooks/01_eda_julia.ipynb`.
- **Source des prévisions du business** (`src/business/forecasts.py`) : modèle de `config/business.yaml`
  (`champion_wf`, passage noté) → à défaut le **champion** (`champion_holdout.parquet`) → à défaut `naive_vol`. Le cache
  est invalidé automatiquement dès qu'un fichier source plus récent existe.
