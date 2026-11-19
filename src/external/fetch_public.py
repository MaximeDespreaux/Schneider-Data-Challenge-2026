"""Téléchargement de données PUBLIQUES et gratuites pouvant aider la prévision.

Règle de confidentialité : ce script n'ENVOIE que des identifiants publics (codes de séries FRED/Eurostat,
adresses de fichiers officiels). Aucune donnée du challenge ne sort de la machine. Les fichiers téléchargés
sont écrits dans data/raw/external/ (gitignoré).

    uv run python -m src.external.fetch_public

Sources et licences :
- FRED (Federal Reserve Bank of St. Louis) : séries IMF, BLS, Fed, CBOE, EPU — usage libre avec citation
  (certaines séries tierces ont leurs propres conditions, cf. la page de chaque série).
- NY Fed, Global Supply Chain Pressure Index : libre avec citation.
- Eurostat (sts_inppd_m) : CC BY 4.0.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from src import data as d

EXT = d.RAW / "external"
UA = {"User-Agent": "Mozilla/5.0 (research script)"}

# code FRED -> description (toutes mensuelles ou agrégées au mois)
FRED = {
    # --- historique long des mêmes prix que le panel (FMI, depuis 1990) -> calibrage des fourchettes
    "PCOPPUSDM": "Cuivre LME USD/t (FMI)", "PALUMUSDM": "Aluminium USD/t (FMI)", "PNICKUSDM": "Nickel USD/t (FMI)",
    "PZINCUSDM": "Zinc USD/t (FMI)", "PIORECRUSDM": "Minerai de fer USD/t (FMI)", "POILBREUSDM": "Brent USD/bbl (FMI)",
    "PNGASEUUSDM": "Gaz Europe USD/MMBtu (FMI)", "PMETAINDEXM": "Indice métaux FMI", "PALLFNFINDEXM": "Indice toutes matières FMI",
    "EXUSEU": "USD par EUR (mensuel)",
    # --- chaîne des prix US : de l'amont (matières brutes) vers l'aval
    "PPIACO": "PPI US toutes matières", "WPSID62": "PPI US matières brutes pour transformation",
    "WPSID61": "PPI US biens intermédiaires", "WPU101": "PPI US fer et acier", "WPU1012": "PPI US ferraille d'acier",
    "WPU102": "PPI US métaux non ferreux", "PCU325211325211": "PPI US résines plastiques", "WPU1017": "PPI US produits sidérurgiques",
    "PCU483111483111": "PPI US fret maritime",
    # --- demande / cycle industriel
    "INDPRO": "Production industrielle US", "IPG331S": "Production US métaux primaires", "TCU": "Taux d'utilisation US",
    "HOUST": "Mises en chantier US", "AMTMNO": "Nouvelles commandes manufacturières US",
    # --- régime de marché / nervosité / monnaie
    "VIXCLS": "VIX (volatilité actions)", "OVXCLS": "OVX (volatilité pétrole)", "GEPUCURRENT": "Incertitude politique mondiale",
    "DTWEXBGS": "Dollar US large (indice)", "T10YIE": "Inflation anticipée US 10 ans", "DGS10": "Taux 10 ans US",
}

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m"
# mêmes indices que S07-S11 du panel, mais depuis 2000 (historique plus long)
EUROSTAT_SERIES = {"S07": ("C2444", "EU27_2020"), "S08": ("C2442", "EU27_2020"), "S09": ("C241", "EU27_2020"),
                   "S10": ("C241", "DE"), "S11": ("C2016", "EU27_2020")}


def fetch_fred(code: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", code]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[code], errors="coerce")
    s.index = df["date"]
    return s.resample("MS").mean()  # séries quotidiennes -> moyenne mensuelle


def fetch_eurostat(nace: str, geo: str) -> pd.Series:
    params = {"nace_r2": nace, "geo": geo, "indic_bt": "PRC_PRR_DOM", "s_adj": "NSA", "unit": "I21", "freq": "M",
              "sinceTimePeriod": "2000-01"}
    r = requests.get(EUROSTAT, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    js = r.json()
    idx = js["dimension"]["time"]["category"]["index"]
    vals = js.get("value", {})
    pos_to_time = {v: k for k, v in idx.items()}
    s = pd.Series({pd.Timestamp(pos_to_time[int(k)] + "-01"): float(v) for k, v in vals.items()}).sort_index()
    return s


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    out, log = {}, []
    for code, desc in FRED.items():
        try:
            out[code] = fetch_fred(code)
            log.append((code, desc, "ok", out[code].first_valid_index(), out[code].last_valid_index()))
        except Exception as e:  # noqa: BLE001
            log.append((code, desc, f"échec: {type(e).__name__}", None, None))
        time.sleep(0.3)
    fred = pd.DataFrame(out)
    fred.to_csv(EXT / "fred_monthly.csv")

    try:
        r = requests.get(GSCPI_URL, headers=UA, timeout=60)
        r.raise_for_status()
        (EXT / "gscpi_data.xlsx").write_bytes(r.content)
        log.append(("GSCPI", "Pression chaînes d'appro (NY Fed)", "ok", None, None))
    except Exception as e:  # noqa: BLE001
        log.append(("GSCPI", "Pression chaînes d'appro (NY Fed)", f"échec: {type(e).__name__}", None, None))

    eu = {}
    for sid, (nace, geo) in EUROSTAT_SERIES.items():
        try:
            eu[sid] = fetch_eurostat(nace, geo)
            log.append((f"EUROSTAT_{sid}", f"{nace} {geo}", "ok", eu[sid].first_valid_index(), eu[sid].last_valid_index()))
        except Exception as e:  # noqa: BLE001
            log.append((f"EUROSTAT_{sid}", f"{nace} {geo}", f"échec: {type(e).__name__}: {str(e)[:80]}", None, None))
    if eu:
        pd.DataFrame(eu).to_csv(EXT / "eurostat_ppi_long.csv")

    rep = pd.DataFrame(log, columns=["code", "description", "statut", "début", "fin"])
    rep.to_csv(EXT / "download_log.csv", index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
