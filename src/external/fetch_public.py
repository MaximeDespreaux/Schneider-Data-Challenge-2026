"""Téléchargement de données PUBLIQUES et gratuites pouvant aider la prévision.

Règle de confidentialité : ce script n'ENVOIE que des identifiants publics (codes de séries FRED/Eurostat,
adresses de fichiers officiels). Aucune donnée du challenge ne sort de la machine. Les fichiers téléchargés
sont écrits dans data/raw/external/ (gitignoré).

    uv run python -m src.external.fetch_public                     # toutes les sources
    uv run python -m src.external.fetch_public --only eurostat     # une seule source (fred | eurostat | gscpi)
    uv run python -m src.external.fetch_public --force             # re-télécharge même si le fichier existe

Sources et licences :
- FRED (Federal Reserve Bank of St. Louis) : séries IMF, BLS, Fed, CBOE, EPU — usage libre avec citation
  (certaines séries tierces ont leurs propres conditions, cf. la page de chaque série).
- NY Fed, Global Supply Chain Pressure Index : libre avec citation.
- Eurostat (sts_inppd_m) : CC BY 4.0.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from openpyxl.utils.exceptions import InvalidFileException

from src import data as d

EXT = d.RAW / "external"
# Pas d'en-tête User-Agent personnalisé : FRED ne répond pas (timeout) à un UA « navigateur » (Chrome testé) ni à
# l'ancien « Mozilla/5.0 (research script) », mais répond tout de suite avec l'UA par défaut de requests.
UA: dict[str, str] = {}
TIMEOUT_S = 20
RETRIES = 2          # tentatives supplémentaires après un échec
RETRY_PAUSE_S = 2
OUTPUTS = {"fred": "fred_monthly.csv", "eurostat": "eurostat_ppi_long.csv", "gscpi": "gscpi.csv"}


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET avec timeout de 20 s et 2 nouvelles tentatives (pause de 2 s)."""
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_PAUSE_S)
    raise RuntimeError("unreachable")

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
    # --- historique long des séries énergie fournies (E06, E07, E08, E22 ; ED03 / ED04 en sont dérivées)
    "PCOALAUUSDM": "Charbon Australie USD/t (FMI)", "PNGASUSUSDM": "Gaz naturel US USD/MMBtu (FMI)",
    "PNGASJPUSDM": "GNL Japon USD/MMBtu (FMI)", "PNRGINDEXM": "Indice énergie FMI",
    "VIXCLS": "VIX (volatilité actions)", "OVXCLS": "OVX (volatilité pétrole)", "GEPUCURRENT": "Incertitude politique mondiale",
    "DTWEXBGS": "Dollar US large (indice)", "T10YIE": "Inflation anticipée US 10 ans", "DGS10": "Taux 10 ans US",
}

GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
# Le « .xlsx » ci-dessus est en fait un ancien .xls (OLE2), illisible par openpyxl. Repli : le CSV qui alimente le
# graphique interactif de la NY Fed (une colonne par millésime de publication ; la dernière = série actuelle).
GSCPI_VINTAGES_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/data/gscpi/gscpi_interactive_data.csv"

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_inppd_m"
# mêmes indices que S07-S11 du panel, mais depuis 2000 (historique plus long)
EUROSTAT_SERIES = {"S07": ("C2444", "EU27_2020"), "S08": ("C2442", "EU27_2020"), "S09": ("C241", "EU27_2020"),
                   "S10": ("C241", "DE"), "S11": ("C2016", "EU27_2020"),
                   # NACE C26.1 = composants et cartes électroniques ; HU = usine AirSeT, FR = usine RM6
                   "EL01": ("C261", "EU27_2020"), "EL02": ("C261", "HU"), "EL03": ("C261", "FR"),
                   # NACE D35 = électricité, gaz, vapeur : historique long de E14 (DE) et E15 (FR)
                   "E14": ("D35", "DE"), "E15": ("D35", "FR")}


def fetch_fred(code: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    r = _get(url)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", code]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[code], errors="coerce")
    s.index = df["date"]
    return s.resample("MS").mean()  # séries quotidiennes -> moyenne mensuelle


def fetch_eurostat(nace: str, geo: str) -> pd.Series:
    params = {"nace_r2": nace, "geo": geo, "indic_bt": "PRC_PRR_DOM", "s_adj": "NSA", "unit": "I21", "freq": "M",
              "sinceTimePeriod": "2000-01"}
    r = _get(EUROSTAT, params=params)
    js = r.json()
    idx = js["dimension"]["time"]["category"]["index"]
    vals = js.get("value", {})
    pos_to_time = {v: k for k, v in idx.items()}
    s = pd.Series({pd.Timestamp(pos_to_time[int(k)] + "-01"): float(v) for k, v in vals.items()}).sort_index()
    return s


def gscpi_to_csv(xlsx: Path, csv: Path) -> pd.Series:
    """Convertit le classeur NY Fed en ``gscpi.csv`` (colonne GSCPI, index = début de mois).

    La mise en page est lue à l'exécution : on retient la feuille dont la 1re colonne contient des dates et la
    2e des nombres (en général « GSCPI Monthly Data »), celle qui a le plus de lignes valides.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    best: dict = {}
    try:
        names = sorted(wb.sheetnames, key=lambda n: n != "GSCPI Monthly Data")
        for name in names:
            found = {}
            for r in wb[name].iter_rows(values_only=True):
                if not r or len(r) < 2:
                    continue
                if isinstance(r[0], datetime):
                    dt = pd.Timestamp(r[0])
                elif isinstance(r[0], str):
                    dt = pd.to_datetime(r[0], errors="coerce")
                else:
                    continue
                if pd.isna(dt) or isinstance(r[1], bool) or not isinstance(r[1], (int, float)):
                    continue
                found[dt.to_period("M").to_timestamp()] = float(r[1])
            if len(found) > len(best):
                best = found
    finally:
        wb.close()
    if len(best) < 12:
        raise ValueError(f"{xlsx.name} : aucune feuille avec des dates en colonne A et des nombres en colonne B")
    s = pd.Series(best, name="GSCPI").sort_index()
    s.index.name = "month"
    s.to_frame().to_csv(csv)
    return s


def _fetch_fred_all(log: list) -> None:
    out = {}
    for code, desc in FRED.items():
        try:
            out[code] = fetch_fred(code)
            log.append((code, desc, "ok", out[code].first_valid_index(), out[code].last_valid_index()))
        except Exception as e:  # noqa: BLE001
            log.append((code, desc, f"échec: {type(e).__name__}", None, None))
        time.sleep(0.3)
    if out:
        pd.DataFrame(out).to_csv(EXT / OUTPUTS["fred"])


def _fetch_gscpi(log: list) -> None:
    try:
        r = _get(GSCPI_URL)
        (EXT / "gscpi_data.xlsx").write_bytes(r.content)
        try:
            g = gscpi_to_csv(EXT / "gscpi_data.xlsx", EXT / OUTPUTS["gscpi"])
            how = "ok (xlsx)"
        except (zipfile.BadZipFile, InvalidFileException):  # fichier .xls déguisé en .xlsx
            g = gscpi_vintages_to_csv(_get(GSCPI_VINTAGES_URL).text, EXT / OUTPUTS["gscpi"])
            how = "ok (csv interactif, dernier millésime)"
        log.append(("GSCPI", "Pression chaînes d'appro (NY Fed)", how, g.first_valid_index(), g.last_valid_index()))
    except Exception as e:  # noqa: BLE001
        log.append(("GSCPI", "Pression chaînes d'appro (NY Fed)", f"échec: {type(e).__name__}", None, None))


def _fetch_eurostat_all(log: list) -> None:
    eu = {}
    for sid, (nace, geo) in EUROSTAT_SERIES.items():
        try:
            eu[sid] = fetch_eurostat(nace, geo)
            log.append((f"EUROSTAT_{sid}", f"{nace} {geo}", "ok", eu[sid].first_valid_index(), eu[sid].last_valid_index()))
        except Exception as e:  # noqa: BLE001
            log.append((f"EUROSTAT_{sid}", f"{nace} {geo}", f"échec: {type(e).__name__}: {str(e)[:80]}", None, None))
    if eu:
        pd.DataFrame(eu).to_csv(EXT / OUTPUTS["eurostat"])


def gscpi_vintages_to_csv(text: str, csv: Path) -> pd.Series:
    """CSV interactif NY Fed -> ``gscpi.csv`` : on garde le millésime le plus récent (colonnes = dates Excel)."""
    df = pd.read_csv(io.StringIO(text), na_values=["#N/A"])
    vintages = [c for c in df.columns[1:] if str(c).strip().isdigit()]
    if not vintages:
        raise ValueError("CSV GSCPI : aucune colonne de millésime trouvée")
    latest = max(vintages, key=lambda c: int(c))
    s = pd.Series(pd.to_numeric(df[latest], errors="coerce").to_numpy(), name="GSCPI",
                  index=pd.to_datetime(df.iloc[:, 0], format="%d-%b-%Y").dt.to_period("M").dt.to_timestamp())
    s = s.dropna().groupby(level=0).mean().sort_index()
    if len(s) < 12:
        raise ValueError("CSV GSCPI : moins de 12 mois de données")
    s.index.name = "month"
    s.to_frame().to_csv(csv)
    return s


EXPECTED_COLUMNS = {"fred": lambda: set(FRED), "eurostat": lambda: set(EUROSTAT_SERIES), "gscpi": lambda: {"GSCPI"}}


def _complete(name: str) -> bool:
    """Le fichier de sortie existe ET contient toutes les séries configurées (sinon on re-télécharge la source)."""
    path = EXT / OUTPUTS[name]
    if not path.exists():
        return False
    cols = set(pd.read_csv(path, nrows=0).columns)
    missing = EXPECTED_COLUMNS[name]() - cols
    if missing:
        print(f"{name} : {path.name} existe mais il manque {sorted(missing)} -> re-téléchargement")
    return not missing


def main(argv=None):
    ap = argparse.ArgumentParser(description="Télécharge les données publiques dans data/raw/external/")
    ap.add_argument("--only", default="all", choices=["all", "fred", "eurostat", "gscpi"])
    ap.add_argument("--force", action="store_true", help="re-télécharger même si le fichier de sortie existe")
    args = ap.parse_args(argv)

    EXT.mkdir(parents=True, exist_ok=True)
    log = []
    steps = {"fred": _fetch_fred_all, "gscpi": _fetch_gscpi, "eurostat": _fetch_eurostat_all}
    for name, fn in steps.items():
        if args.only not in ("all", name):
            continue
        if not args.force and _complete(name):
            print(f"{name} : {EXT / OUTPUTS[name]} existe déjà avec toutes les séries -> ignoré "
                  "(--force pour re-télécharger)")
            continue
        fn(log)
    if not log:
        return
    rep = pd.DataFrame(log, columns=["code", "description", "statut", "début", "fin"])
    path = EXT / "download_log.csv"
    if path.exists():  # garder les lignes des sources non relancées
        old = pd.read_csv(path)
        rep = pd.concat([old[~old["code"].isin(rep["code"])], rep], ignore_index=True)
    rep.to_csv(path, index=False)
    print(rep[rep["code"].isin([r[0] for r in log])].to_string(index=False))


if __name__ == "__main__":
    main()
