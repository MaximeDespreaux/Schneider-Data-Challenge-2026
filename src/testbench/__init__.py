"""Banc de test des prévisions (partie 2 du team split).

Idée : rejouer l'histoire mois par mois. À chaque date d'origine, on ne donne au modèle que ce qui
était connu à cette date, il prévoit 3/6/9 mois plus tard (P10/P50/P90), on compare avec la réalité,
on avance d'un mois et on recommence. Tous les modèles sont comparés à « le prix reste le même ».

Modules
- vintage   : ce qui est connu à une date donnée (délais de publication) — garde-fou anti-fuite
- models    : interface commune + baselines + wrapper d'intervalles conformal
- backtest  : moteur d'origines glissantes
- metrics   : MAE, MAE relative au naïf, pinball, couverture P10–P90, test de Diebold-Mariano
- submission: validateur du format CSV de Schneider
- run       : point d'entrée en ligne de commande
"""
