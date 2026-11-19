"""Panel enrichi : historique long + signaux externes + signaux « créatifs ».

Tout est ajouté comme des COLONNES du panel mensuel. Le banc de test tronque le panel à chaque origine, donc
aucun signal ne peut voir le futur (vérifié par le test de perturbation, src/testbench/audit.py).

1. Historique long (`extend_history`) : les mêmes séries que le panel, prolongées dans le passé à partir des
   sources publiques identiques (FMI via FRED depuis 1992, Eurostat depuis 2000, BLS). Raccord par ratio au premier
   mois commun ; sur la période commune les sources coïncident (écart 0 % pour métaux et Eurostat).
   -> plus de crises dans l'historique (2008, 2011, 2015) pour calibrer les fourchettes et entraîner les modèles.
2. Signaux externes publics (préfixe X_) : chaîne des prix US, demande, nervosité, dollar, supply chain.
3. Signaux LME fournis (préfixe X_) : structure à terme (prix 3 mois vs comptant) et stocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import openpyxl

from src import data as d

EXT = d.RAW / "external"

# série du panel -> (fichier, colonne) de la source longue identique
LONG_SOURCES = {
    "S01": ("fred", "PCOPPUSDM"), "S02": ("fred", "PALUMUSDM"), "S03": ("fred", "PNICKUSDM"),
    "S04": ("fred", "PIORECRUSDM"), "S05": ("fred", "PZINCUSDM"), "S06": ("fred", "EXUSEU"),
    "S12": ("fred", "WPU1017"), "S13": ("fred", "PCU325211325211"),
    "S07": ("eu", "S07"), "S08": ("eu", "S08"), "S09": ("eu", "S09"), "S10": ("eu", "S10"), "S11": ("eu", "S11"),
    "E01": ("fred", "POILBREUSDM"), "E05": ("fred", "PNGASEUUSDM"),
}

EXTERNAL = {  # colonne FRED -> nom du signal
    "PPIACO": "X_US_PPI_ALL", "WPSID62": "X_US_PPI_CRUDE", "WPSID61": "X_US_PPI_INTERMED", "WPU101": "X_US_PPI_STEEL",
    "WPU1012": "X_US_STEEL_SCRAP", "WPU102": "X_US_PPI_NONFERROUS", "PCU483111483111": "X_FREIGHT",
    "INDPRO": "X_US_INDPRO", "IPG331S": "X_US_IP_METALS", "TCU": "X_US_CAPUTIL", "HOUST": "X_US_HOUSING",
    "AMTMNO": "X_US_NEW_ORDERS", "VIXCLS": "X_VIX", "OVXCLS": "X_OVX", "GEPUCURRENT": "X_GLOBAL_EPU",
    "DTWEXBGS": "X_USD_BROAD", "T10YIE": "X_US_BREAKEVEN", "DGS10": "X_US_10Y", "PMETAINDEXM": "X_IMF_METALS",
    "PALLFNFINDEXM": "X_IMF_ALLCOMMOD",
}

def _fred() -> pd.DataFrame:
    return pd.read_csv(EXT / "fred_monthly.csv", index_col=0, parse_dates=True)


def _eu() -> pd.DataFrame:
    return pd.read_csv(EXT / "eurostat_ppi_long.csv", index_col=0, parse_dates=True)


def load_lme() -> pd.DataFrame:
    """Structure à terme et stocks LME (fichier fourni). Mois complets uniquement."""
    wb = openpyxl.load_workbook(d.RAW / "LME_Copper_Aluminium_EU_Monthly_2016_2026.xlsx", read_only=True, data_only=True)
    rows = list(wb["Monthly Prices"].iter_rows(min_row=6, values_only=True))
    df = pd.DataFrame(rows, columns=["month", "status", "cu_cash", "cu_3m", "al_cash", "al_3m", "fx", "a", "b", "c", "e",
                                     "cu_spread", "al_spread", "cu_stocks", "al_stocks", "fxcov"])
    df = df[df["status"].astype(str).str.startswith("Completed")]
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    df = df.set_index("month")
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "X_CU_TERM": np.log(num("cu_3m") / num("cu_cash")),   # > 0 : contango (abondance) ; < 0 : backwardation (tension)
        "X_AL_TERM": np.log(num("al_3m") / num("al_cash")),
        "X_CU_STOCKS": num("cu_stocks"), "X_AL_STOCKS": num("al_stocks"),
    })


def load_external() -> pd.DataFrame:
    f = _fred()
    x = f[[c for c in EXTERNAL if c in f]].rename(columns=EXTERNAL)
    # GSCPI optionnel : le fetcher enregistre un .xls (format OLE2) que rien ne convertit en gscpi.csv
    if (EXT / "gscpi.csv").exists():
        g = pd.read_csv(EXT / "gscpi.csv", index_col=0, parse_dates=True)["GSCPI"].rename("X_GSCPI")
        x = x.join(g, how="outer")
    try:
        x = x.join(load_lme(), how="outer")
    except FileNotFoundError:
        pass
    return x


def extend_history(panel: pd.DataFrame, start: str = "1992-01-01") -> pd.DataFrame:
    """Prolonge les séries du panel dans le passé avec leur source publique identique (raccord par ratio)."""
    f, eu = _fred(), _eu()
    idx = pd.date_range(start, panel.index.max(), freq="MS")
    out = panel.reindex(idx)
    for col, (src, code) in LONG_SOURCES.items():
        if col not in out:
            continue
        longs = (f if src == "fred" else eu)[code].reindex(idx)
        first = panel[col].first_valid_index()
        if first is None or pd.isna(longs.get(first)):
            continue
        ratio = panel.at[first, col] / longs.at[first]
        before = idx < first
        out.loc[before, col] = longs[before] * ratio
    # séries dérivées en EUR (définies seulement là où le taux EUR/USD existe, i.e. depuis 1999)
    before = out.index < panel.index.min()
    out.loc[before, "D01"] = out.loc[before, "S01"] / out.loc[before, "S06"]
    out.loc[before, "D02"] = out.loc[before, "S02"] / out.loc[before, "S06"]
    if "ED01" in out:
        out.loc[before, "ED01"] = out.loc[before, "E01"] / out.loc[before, "S06"]
    if "ED02" in out:
        out.loc[before, "ED02"] = out.loc[before, "E05"] * 3.412 / out.loc[before, "S06"]
    return out


def build_panel(long: bool = True, external: bool = True) -> pd.DataFrame:
    """Panel public (sans données sous licence). Délègue au constructeur unique du banc de test."""
    from src.testbench.panel import build_full_panel
    return build_full_panel(use_licensed=False, long=long, external=external)
