# Données publiques externes et modèle « champion »

> Aucune donnée ici : ce document décrit les sources et le code. Les fichiers téléchargés et les résultats restent
> dans `data/` (gitignoré).

## Confidentialité
Le script de téléchargement n'**envoie** que des identifiants publics (codes de séries FRED / Eurostat, adresse du
fichier de la Fed de New York). Aucune donnée du challenge ne sort de la machine.

```
uv run python -m src.external.fetch_public     # télécharge dans data/raw/external/
uv run python -m src.external.screen           # criblage des signaux (cibles < 2024 uniquement)
```

## Sources (gratuites)
| Source | Séries | Depuis | Usage | Licence |
|---|---|---|---|---|
| FRED / FMI | cuivre, alu, nickel, zinc, minerai, Brent, gaz Europe, indices métaux et toutes matières | 1992 | historique long des cibles + signaux | libre avec citation |
| FRED / BLS | PPI US : toutes matières, matières brutes, biens intermédiaires, fer et acier, ferraille, non-ferreux, résines, fret maritime, produits sidérurgiques | 1913–1988 | historique long (S12, S13) + signaux « chaîne des prix » | domaine public (BLS) |
| FRED / Fed | production industrielle, métaux primaires, taux d'utilisation, mises en chantier, commandes, dollar large, taux 10 ans, inflation anticipée | 1919–2006 | signaux de demande / monnaie | domaine public |
| FRED / CBOE, EPU | VIX, OVX, incertitude politique mondiale | 1990–2007 | régime de nervosité | conditions des fournisseurs (citation) |
| NY Fed | Global Supply Chain Pressure Index | 1998 | tension des chaînes d'appro | libre avec citation |
| Eurostat `sts_inppd_m` | mêmes indices que S07–S11 | 2000 | historique long | CC BY 4.0 |
| Fichier LME fourni | prix comptant vs 3 mois, stocks | 2016 | structure à terme | données du challenge (NDA) |

Contrôle : sur la période commune, les sources longues coïncident avec le panel du challenge (0 % d'écart pour
métaux et Eurostat ; < 1,5 % pour les PPI US, révisés).

## Signaux testés (`src/external/features.py`)
Dynamique propre, autres matières, énergie, signaux externes, et des signaux « créatifs » :
`breadth3` (part des matières en hausse), `overext36` (distance au plus haut 3 ans), `real_dev120` (prix réel vs
moyenne 10 ans), `twin_gap` (indice Eurostat vs prix de marché jumeau), `upstream3` (matières brutes vs biens
intermédiaires US), `scrap_spread3` (ferraille vs acier), structure à terme et stocks LME, régime de volatilité.

## Protocole sans triche
1. **Criblage** (`screen.py`) : uniquement des cibles **antérieures à 2024**. Un signal n'est retenu que si sa
   corrélation avec la variation future a le **même signe sur 2003-2012 et 2013-2023** et dépasse 2/√(fenêtres
   indépendantes).
2. **Règle du modèle final fixée avant vérification** (`models.py::Champion`) : régression ridge prudente sur les
   signaux retenus si au moins 3 signaux stables existent pour (série, horizon), sinon naïf ; fourchette `naive_vol`.
3. **Vérification sur 2024-2026**, jamais utilisée pour choisir quoi que ce soit.
4. **Audit de perturbation** : tout le futur de toutes les colonnes (y compris externes) remplacé par du bruit →
   prévisions identiques, en modes submission et realtime (délais de publication des signaux dans
   `config/availability.yaml`).

## Limites
- Les signaux sont surtout américains/mondiaux ; peu de séries chinoises gratuites et propres.
- La structure à terme LME ne commence qu'en 2016 : trop courte pour être validée seule.
- Sur les prix LME à 3 mois, la règle « ≥ 3 signaux » active des signaux faibles qui n'aident pas en 2024-26.
