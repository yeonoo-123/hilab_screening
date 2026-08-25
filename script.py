#!/usr/bin/env python3
"""
Firm size and clinical-trial phase participation, 2016-2026.

One script, three stages:

    1. trials   AACT -> data/interim/trials.csv
                One row per (trial, sponsor). Industry-sponsored
                interventional studies started in the analysis window,
                with a normalized phase and a normalized sponsor key.

    2. firms    SEC EDGAR -> data/interim/firm_year.csv
                One row per (CIK, fiscal year) for firms in the pharma
                SIC universe, with as-reported annual revenue and a
                firm_group of BIG_PHARMA / BIOTECH / OTHER.

    3. exhibit  data/reports/
                Phase distribution of lead-sponsor trials, Big Pharma vs
                Biotech, plus a sample-construction table.

Run everything:      python pipeline.py
Rebuild one stage:   python pipeline.py --stage firms --force

Measurement choices that matter are collected in SETTINGS below and
restated in README.md. Two rules are enforced throughout:

  * Entity resolution is exact-key only. Sponsor and SEC names are
    normalized to a canonical key and joined on equality; nothing is
    merged on fuzzy similarity, because false positives there are
    invisible in the output.
  * Nothing is imputed. A sponsor with no SEC match, no SIC code, or no
    revenue in the trial's start year is excluded from the exhibit and
    counted in the sample-construction table, never guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import time
import unicodedata
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

# =====================================================================
# SETTINGS
# =====================================================================

# SEC requires a contact address; requests without one get a 403.
# Set it here or export SEC_USER_AGENT="Jane Doe, Univ of X, j@x.edu".
SEC_USER_AGENT = "Yeonoo Jeong, UCSB, yeonoo@ucsb.edu"

START_YEAR, END_YEAR = 2016, 2026

# Classification on nominal (as-reported) global annual revenue.
BIG_PHARMA_REVENUE_FLOOR_USD = 10_000_000_000
BIOTECH_REVENUE_CAP_USD = 500_000_000

# SEC SIC codes scoping the firm universe:
#   2833 Medicinal Chemicals & Botanical Products
#   2834 Pharmaceutical Preparations
#   2836 Biological Products (No Diagnostic Substances)
#   8731 Commercial Physical & Biological Research
PHARMA_SIC_CODES = {"2833", "2834", "2836", "8731"}

# Combination designations (PHASE1/PHASE2) -> "earlier" or "later".
COMBO_PHASE = "earlier"
INCLUDE_PHASE4 = True

# Optional explicit folder holding the AACT export; None = auto-discover.
AACT_DIR: str | None = None

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_RATE_LIMIT_PER_SEC = 8

# =====================================================================
# Paths -- anchored on the repository, not the shell's directory
# =====================================================================


def _find_root(start: Path) -> Path:
    """Nearest ancestor containing data/, so the script runs from anywhere."""
    for d in [start, *start.parents]:
        if (d / "data").is_dir():
            return d
    return start


ROOT = _find_root(Path(__file__).resolve().parent)
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
REPORTS = DATA / "reports"
MANIFEST = RAW / "MANIFEST.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")

NOT_IN_SOURCE = ".n"   # source has no value for this field
UNMATCHED = ".u"       # value present but could not be resolved


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Parquet if an engine is installed, else the .csv sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except ImportError:
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


def read_table(path: Path) -> pd.DataFrame:
    """Read whichever of .parquet / .csv exists."""
    pq, csv = path.with_suffix(".parquet"), path.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(
        f"Neither {pq.name} nor {csv.name} exists in {path.parent}. "
        f"Run the earlier stage first (python pipeline.py)."
    )


def table_exists(path: Path) -> bool:
    return path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists()


# =====================================================================
# Fetch with an on-disk cache and a provenance manifest
# =====================================================================


def _user_agent() -> str:
    import os
    ua = SEC_USER_AGENT or os.environ.get("SEC_USER_AGENT", "")
    if not ua.strip():
        sys.exit(
            "SEC_USER_AGENT is not set. SEC blocks requests without a contact "
            "address. Set SEC_USER_AGENT at the top of pipeline.py, or export "
            'SEC_USER_AGENT="Your Name, Institution, you@example.edu".'
        )
    return ua


def fetch_json(url: str, retries: int = 3, quiet_404: bool = False) -> dict | None:
    """
    GET `url`, cache the body under data/raw/ keyed by URL hash, and append
    one provenance line to MANIFEST.jsonl.

    The cache is what makes a rerun cheap and a result reproducible: AACT
    ships a new snapshot monthly and SEC filings are restated, so the
    manifest (URL, timestamp, SHA-256) is the thing to cite in a data
    availability statement.
    """
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".json")
    if dest.exists():
        try:
            return json.loads(dest.read_text())
        except json.JSONDecodeError:
            dest.unlink()

    headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}
    backoff = 1.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                if not quiet_404:
                    log.debug("404 %s", url)
                return None
            if r.status_code == 403 and "sec.gov" in url:
                sys.exit("SEC returned 403 -- set a real contact address in SEC_USER_AGENT.")
            r.raise_for_status()
            dest.write_bytes(r.content)
            with MANIFEST.open("a") as fh:
                fh.write(json.dumps({
                    "url": url,
                    "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
                    "http_status": r.status_code,
                    "sha256": hashlib.sha256(r.content).hexdigest(),
                    "n_bytes": len(r.content),
                }) + "\n")
            time.sleep(1.0 / SEC_RATE_LIMIT_PER_SEC)
            return json.loads(r.content)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed (%d/%d) %s: %s", attempt, retries, url, exc)
            time.sleep(backoff)
            backoff *= 2
    return None


# =====================================================================
# Entity resolution
# =====================================================================

_LEGAL_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "lp", "llp", "plc", "sa", "s a", "ag", "nv",
    "n v", "bv", "b v", "gmbh", "kgaa", "kg", "oy", "ab", "as", "a s",
    "aps", "spa", "s p a", "srl", "s r l", "pty", "pte", "kk", "k k",
    "co ltd", "holdings", "holding", "group", "trust",
]

# Descriptors carrying no identity information: dropping them collapses
# "Novartis Pharmaceuticals Corporation" -> "novartis". Deliberately does
# NOT include "labs/laboratories" (Abbott Laboratories vs Abbott) or
# "health" (Bayer HealthCare is a distinct filer) -- those tokens do
# distinguish real entities in this industry.
_NOISE_TOKENS = {
    "pharmaceutical", "pharmaceuticals", "pharma", "pharmac", "pharmaceutica",
    "therapeutics", "therapeutic", "biosciences", "bioscience", "biopharma",
    "biopharmaceutical", "biopharmaceuticals", "biotechnology", "biotech",
    "sciences", "science", "medicines", "medicine", "medical",
    "international", "worldwide", "global", "usa", "us", "america",
    "american", "north", "research", "development", "randd",
    "the", "of", "and",
}

_SEP = re.compile(r"[^a-z0-9]+")


def normalize_firm(name: str | None) -> str:
    """
    Canonical key for a firm name, used for both AACT sponsors and SEC
    registrants. Strips accents, parentheticals, legal form, and industry
    descriptors; never merges on fuzzy similarity ("Alexion" vs "Alexza").
    """
    if not name or not isinstance(name, str):
        return UNMATCHED
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = _SEP.sub(" ", s).strip()

    toks = s.split()
    changed = True
    while changed and toks:
        changed = False
        for k in (3, 2, 1):
            if len(toks) >= k and " ".join(toks[-k:]) in _LEGAL_SUFFIXES:
                toks = toks[:-k]
                changed = True
                break

    key = " ".join(t for t in toks if t not in _NOISE_TOKENS).strip()
    return key or UNMATCHED


_PHASE_MAP = {
    "EARLY_PHASE1": "PHASE1", "EARLY PHASE 1": "PHASE1",
    "PHASE1": "PHASE1", "PHASE 1": "PHASE1",
    "PHASE2": "PHASE2", "PHASE 2": "PHASE2",
    "PHASE3": "PHASE3", "PHASE 3": "PHASE3",
    "PHASE4": "PHASE4", "PHASE 4": "PHASE4",
}
_PHASE_ORDER = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}


def normalize_phase(raw: str | None) -> str:
    """
    Map an AACT phase string onto the analysis ladder, always returning a
    single phase. Combination designations (PHASE1/PHASE2) resolve to the
    EARLIER or LATER phase per COMBO_PHASE; unusable values return
    NOT_IN_SOURCE and are flagged, not silently dropped.
    """
    if not raw or not isinstance(raw, str):
        return NOT_IN_SOURCE
    s = raw.strip().upper().replace("PHASE", "PHASE ").replace("  ", " ").replace("PHASE ", "PHASE")
    if s in ("NA", "N/A", "NOT APPLICABLE", ""):
        return NOT_IN_SOURCE

    mapped = [_PHASE_MAP[p.strip()] for p in re.split(r"[/,]", s)
              if p.strip() in _PHASE_MAP]
    if not mapped:
        return NOT_IN_SOURCE
    pick = min if COMBO_PHASE == "earlier" else max
    return pick(mapped, key=lambda m: _PHASE_ORDER[m])


# =====================================================================
# Stage 1 -- trials
# =====================================================================

AACT_SUFFIXES = (".txt", ".csv", ".tsv")
STUDY_COLS = ["nct_id", "phase", "overall_status", "study_type", "start_date",
              "completion_date", "enrollment", "why_stopped"]


def _sniff_sep(sample: str) -> str:
    """AACT ships pipe-delimited files; the web UI also hands out CSV/TSV.
    Guess from the header, because the wrong separator does not raise --
    it yields a one-column frame and a KeyError three steps later."""
    head = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {sep: head.count(sep) for sep in ("|", "\t", ",", ";")}
    best = max(counts, key=counts.get)
    return best if counts[best] else "|"


def _read_aact(table: str) -> pd.DataFrame:
    """Find and read one AACT table from disk. Accepts loose .txt/.csv/.tsv
    files or any .zip, in AACT_DIR, any data/ subfolder, or the repo root."""
    cand: list[Path] = []
    if AACT_DIR:
        cand.append(Path(AACT_DIR).expanduser())
    cand += [d for d in sorted(DATA.glob("*")) if d.is_dir()] + [DATA, ROOT]

    inventory: set[str] = set()
    for d in cand:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in AACT_SUFFIXES:
                inventory.add(p.stem.lower())
                if p.stem.lower() == table:
                    sample = p.open("r", encoding="utf-8", errors="replace").read(8192)
                    df = pd.read_csv(p, sep=_sniff_sep(sample), low_memory=False,
                                     on_bad_lines="warn", encoding_errors="replace")
                    log.info("AACT %-14s %8d rows  %s", table, len(df), p)
                    return df
            elif p.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(p) as z:
                        for m in z.namelist():
                            mp = Path(m)
                            if mp.suffix.lower() not in AACT_SUFFIXES:
                                continue
                            inventory.add(mp.stem.lower())
                            if mp.stem.lower() == table:
                                with z.open(m) as fh:
                                    raw = fh.read()
                                df = pd.read_csv(
                                    io.BytesIO(raw),
                                    sep=_sniff_sep(raw[:8192].decode("utf-8", "replace")),
                                    low_memory=False, on_bad_lines="warn",
                                    encoding_errors="replace")
                                log.info("AACT %-14s %8d rows  %s::%s",
                                         table, len(df), p.name, m)
                                return df
                except zipfile.BadZipFile:
                    continue

    raise FileNotFoundError(
        f"AACT table '{table}' not found.\n"
        f"  Searched: {', '.join(str(d) for d in cand if d.is_dir())}\n"
        f"  Found instead: {', '.join(sorted(inventory)) or '(nothing)'}\n"
        f"  Put studies.txt and sponsors.txt (or the AACT export .zip) in "
        f"{DATA}/AACT, or set AACT_DIR at the top of pipeline.py."
    )


def _select(df: pd.DataFrame, cols: list[str], table: str) -> pd.DataFrame:
    """Select the columns we need, tolerating AACT schema drift."""
    have = [c for c in cols if c in df.columns]
    missing = set(cols) - set(have)
    if missing:
        log.warning("AACT %s is missing column(s) %s -- continuing without them",
                    table, sorted(missing))
    return df[have].copy()


def build_trials() -> pd.DataFrame:
    """One row per (trial, sponsor) for industry interventional studies."""
    studies = _select(_read_aact("studies"), STUDY_COLS, "studies")
    sponsors = _select(_read_aact("sponsors"),
                       ["nct_id", "name", "agency_class", "lead_or_collaborator"],
                       "sponsors")

    s = studies[studies["study_type"].astype(str).str.upper()
                .str.contains("INTERVENTION", na=False)].copy()
    n_interventional = len(s)

    s["phase_norm"] = s["phase"].map(normalize_phase)
    ladder = {"PHASE1", "PHASE2", "PHASE3"} | ({"PHASE4"} if INCLUDE_PHASE4 else set())
    s["phase_missing"] = ~s["phase_norm"].isin(ladder)

    for c in ("start_date", "completion_date"):
        if c in s.columns:
            s[c] = pd.to_datetime(s[c], errors="coerce")
    s["start_year"] = s["start_date"].dt.year
    s["date_missing"] = s["start_year"].isna()
    s = s[s["start_year"].between(START_YEAR, END_YEAR) | s["date_missing"]]

    st = s["overall_status"].astype(str).str.upper()
    s["trial_completed"] = st.eq("COMPLETED")
    s["trial_stopped_early"] = st.isin({"TERMINATED", "WITHDRAWN", "SUSPENDED"})
    s["trial_ongoing"] = st.isin({"RECRUITING", "ACTIVE_NOT_RECRUITING",
                                  "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"})

    sp = sponsors.rename(columns={"name": "sponsor_name_raw"})
    sp["sponsor_role"] = sp["lead_or_collaborator"].astype(str).str.lower()
    sp["firm_key"] = sp["sponsor_name_raw"].map(normalize_firm)
    sp["is_industry"] = sp["agency_class"].astype(str).str.upper().eq("INDUSTRY")

    # An industry trial is one with at least one industry sponsor in any role.
    s = s[s["nct_id"].isin(set(sp.loc[sp["is_industry"], "nct_id"]))]
    sp = sp[sp["nct_id"].isin(set(s["nct_id"]))]

    agg = sp.groupby("nct_id").agg(
        n_sponsors=("sponsor_name_raw", "size"),
        n_industry_sponsors=("is_industry", "sum"),
    ).reset_index()
    agg["has_industry_collaborator"] = agg["n_industry_sponsors"] > 1

    out = (s.merge(agg, on="nct_id", how="left")
             .merge(sp.drop(columns=["lead_or_collaborator"]), on="nct_id", how="left"))
    out["source_dataset"] = "AACT"

    log.info("interventional studies: %d; industry-sponsored in %d-%d: %d trials, "
             "%d trial-sponsor rows", n_interventional, START_YEAR, END_YEAR,
             out["nct_id"].nunique(), len(out))
    log.info("phase unusable: %.1f%% of trials; start date missing: %.1f%%",
             100 * s["phase_missing"].mean(), 100 * s["date_missing"].mean())
    return out


# =====================================================================
# Stage 2 -- firms
# =====================================================================

# XBRL revenue tags in priority order. Firms tag revenue inconsistently
# and many switched tag when ASC 606 took effect in 2018, so all tags are
# collected and the highest-priority one available is chosen PER YEAR --
# taking only the first tag that appears anywhere would truncate the
# panel for every firm that switched.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]


def build_crosswalk(sponsor_names) -> pd.DataFrame:
    """
    Join AACT sponsor strings to SEC CIKs on the normalized key.

    Exact-key matching only. Keys resolving to several CIKs are flagged
    AMBIGUOUS rather than arbitrarily assigned. The unmatched residual is
    mostly private firms, foreign filers without a US listing, and
    academic sponsors -- a substantively meaningful group, reported in the
    sample-construction table rather than resolved away.
    """
    idx = fetch_json(SEC_COMPANY_TICKERS_URL)
    if not idx:
        sys.exit("could not fetch the SEC company index")
    sec = pd.DataFrame(idx.values() if isinstance(idx, dict) else idx)
    sec = sec.rename(columns={"cik_str": "cik", "title": "sec_name"})
    sec["cik"] = sec["cik"].astype(int)
    sec["firm_key"] = sec["sec_name"].map(normalize_firm)
    sec = sec[sec["firm_key"] != UNMATCHED]

    dupes = sec.groupby("firm_key")["cik"].nunique()
    ambiguous = set(dupes[dupes > 1].index)
    sec = sec.drop_duplicates("firm_key")

    obs = pd.DataFrame({"sponsor_name_raw": sorted({n for n in sponsor_names if n})})
    obs["firm_key"] = obs["sponsor_name_raw"].map(normalize_firm)
    out = obs.merge(sec[["firm_key", "cik", "ticker", "sec_name"]],
                    on="firm_key", how="left")
    out["cik"] = out["cik"].astype("Int64")
    out["match_status"] = "matched"
    out.loc[out["cik"].isna(), "match_status"] = UNMATCHED
    out.loc[out["firm_key"].isin(ambiguous), "match_status"] = "ambiguous"

    log.info("crosswalk: %d sponsor strings, %.1f%% matched to a CIK",
             len(out), 100 * out["match_status"].eq("matched").mean())
    return out


def fetch_sic(cik: int) -> str:
    """CIK -> 4-digit SIC. Only the submissions endpoint carries it;
    companyfacts does not."""
    sub = fetch_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if sub and sub.get("sic"):
        return str(sub["sic"]).zfill(4)
    return NOT_IN_SOURCE


def annual_revenue(facts: dict) -> tuple[dict[int, float], dict[int, str]]:
    """
    Annual revenue by fiscal year from XBRL company facts.

    Two things the obvious implementation gets wrong:

      * The `fy`/`fp` fields describe the FILING, not the observation. A
        10-K carries three years of comparative figures, all stamped with
        the filing's fy -- keying on `fy` silently relabels prior-year
        revenue as current-year. The period is taken from start/end.
      * Only durations of roughly a year are annual revenue; quarterly and
        multi-year cumulative facts share the same tag.

    Fiscal years ending before June are labelled to the prior calendar
    year, the usual convention. Where a year appears more than once
    (restatements) the latest-filed value wins.
    """
    us = (facts.get("facts") or {}).get("us-gaap") or {}
    by_year: dict[int, tuple[int, str, float]] = {}   # year -> (tag rank, filed, val)

    for rank, tag in enumerate(REVENUE_TAGS):
        node = us.get(tag)
        if not node:
            continue
        for unit, observations in (node.get("units") or {}).items():
            if not unit.startswith("USD"):
                continue
            for o in observations:
                if str(o.get("form", "")).split("/")[0] != "10-K":
                    continue
                start, end, val = o.get("start"), o.get("end"), o.get("val")
                if not start or not end or val is None:
                    continue
                try:
                    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
                except ValueError:
                    continue
                if not 340 <= (d1 - d0).days <= 400:
                    continue
                year = d1.year if d1.month >= 6 else d1.year - 1
                filed = str(o.get("filed", ""))
                prev = by_year.get(year)
                # a higher-priority tag wins; within one tag, the later filing
                # wins, so restatements supersede the original figure
                if prev is None or rank < prev[0] or (rank == prev[0] and filed >= prev[1]):
                    by_year[year] = (rank, filed, float(val))

    values = {y: v for y, (_, _, v) in by_year.items()}
    tags = {y: REVENUE_TAGS[r] for y, (r, _, _) in by_year.items()}
    return values, tags


def build_firm_years(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crosswalk -> SIC filter -> revenue -> firm_group, one row per CIK-year."""
    xwalk = build_crosswalk(pd.unique(trials["sponsor_name_raw"].dropna().astype(str)))
    matched = xwalk[xwalk["match_status"].eq("matched")]
    ciks = sorted({int(c) for c in matched["cik"].dropna().unique()})

    # SIC first: it is one small request per firm and it removes most of
    # the CIKs before the much larger companyfacts pull.
    log.info("resolving SIC for %d registrants", len(ciks))
    sic = {}
    for i, cik in enumerate(ciks, 1):
        sic[cik] = fetch_sic(cik)
        if i % 250 == 0:
            log.info("  SIC %d/%d", i, len(ciks))
    in_universe = [c for c in ciks if sic.get(c) in PHARMA_SIC_CODES]
    log.info("in pharma SIC universe: %d of %d registrants", len(in_universe), len(ciks))
    if not in_universe:
        sys.exit("no firms in the SIC universe -- check SEC_USER_AGENT and network access")

    log.info("pulling XBRL revenue for %d registrants", len(in_universe))
    rows: list[dict] = []
    for i, cik in enumerate(in_universe, 1):
        facts = fetch_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        if not facts:
            continue
        values, tags = annual_revenue(facts)
        for year, val in values.items():
            if START_YEAR - 1 <= year <= END_YEAR:
                rows.append({"cik": cik, "year": year, "revenue_usd": val,
                             "revenue_tag": tags[year], "sic": sic[cik],
                             "sec_entity_name": facts.get("entityName")})
        if i % 100 == 0:
            log.info("  revenue %d/%d", i, len(in_universe))

    fy = pd.DataFrame(rows)
    if fy.empty:
        sys.exit("no SEC financials retrieved -- check SEC_USER_AGENT and network access")

    fy["firm_group"] = "OTHER"
    fy.loc[fy["revenue_usd"] >= BIG_PHARMA_REVENUE_FLOOR_USD, "firm_group"] = "BIG_PHARMA"
    fy.loc[fy["revenue_usd"] < BIOTECH_REVENUE_CAP_USD, "firm_group"] = "BIOTECH"

    fy = fy.merge(matched[["cik", "firm_key"]].drop_duplicates("cik").astype({"cik": int}),
                  on="cik", how="left")

    counts = ", ".join(f"{k}={v}" for k, v in fy["firm_group"].value_counts().items())
    log.info("firm-years: %d for %d firms (%s)", len(fy), fy["cik"].nunique(), counts)
    return fy, xwalk


# =====================================================================
# Stage 3 -- exhibit
# =====================================================================

FIRM_GROUPS = ["BIG_PHARMA", "BIOTECH"]
GROUP_LABELS = {"BIG_PHARMA": "Big Pharma", "BIOTECH": "Biotech"}
GROUP_COLORS = {"BIG_PHARMA": "#4C72B0", "BIOTECH": "#DD8452"}
PHASE_LABELS = {"PHASE1": "Phase 1", "PHASE2": "Phase 2",
                "PHASE3": "Phase 3", "PHASE4": "Phase 4"}


def build_exhibit(trials: pd.DataFrame, firm_year: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase distribution of lead-sponsor trials by firm group, with the
    sample-construction table that documents every exclusion.

    Each trial is attributed to its lead sponsor's classification IN THE
    TRIAL'S START YEAR: firm_group is time-varying, so a firm crossing the
    $10B line mid-window is Biotech before and Big Pharma after.
    """
    steps: list[dict] = []

    lead = trials[trials["sponsor_role"].eq("lead")].copy()
    steps.append({"step": "industry trials, lead sponsor identified",
                  "n_trials": lead["nct_id"].nunique()})

    lead = lead[~lead["phase_missing"]]
    steps.append({"step": "with a usable phase", "n_trials": lead["nct_id"].nunique()})

    lead = lead.dropna(subset=["start_year"])
    lead["start_year"] = lead["start_year"].astype(int)
    steps.append({"step": "with a start date", "n_trials": lead["nct_id"].nunique()})

    fy = firm_year.dropna(subset=["firm_key"])[["firm_key", "year", "firm_group",
                                                "revenue_usd"]]
    lead = lead.merge(fy, left_on=["firm_key", "start_year"],
                      right_on=["firm_key", "year"], how="left")
    lead = lead[lead["firm_group"].notna()]
    steps.append({"step": "lead sponsor has SEC revenue in the start year "
                          "and is in the SIC universe",
                  "n_trials": lead["nct_id"].nunique()})

    lead = lead[lead["firm_group"].isin(FIRM_GROUPS)]
    steps.append({"step": "lead sponsor is BIG_PHARMA or BIOTECH (excludes OTHER)",
                  "n_trials": lead["nct_id"].nunique()})

    sample = pd.DataFrame(steps)
    sample["pct_of_start"] = (100 * sample["n_trials"] / sample["n_trials"].iloc[0]).round(1)

    counts = (lead.groupby(["phase_norm", "firm_group"])["nct_id"]
              .nunique().rename("n_trials").reset_index())
    if not counts.empty:
        counts["pct_of_group"] = 100 * counts["n_trials"] / counts.groupby(
            "firm_group")["n_trials"].transform("sum")
        order = [p for p in PHASE_LABELS if p in set(counts["phase_norm"])]
        counts["phase_norm"] = pd.Categorical(counts["phase_norm"], order, ordered=True)
        counts = counts.sort_values(["phase_norm", "firm_group"])
    return counts, sample


def plot_exhibit(counts: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = (counts.pivot(index="phase_norm", columns="firm_group",
                          values="pct_of_group")
             .reindex(columns=[g for g in FIRM_GROUPS if g in set(counts["firm_group"])])
             .fillna(0))
    totals = counts.groupby("firm_group")["n_trials"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    for i, group in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        xs = [x + offset for x in range(len(pivot.index))]
        ax.bar(xs, pivot[group], width, color=GROUP_COLORS.get(group),
               label=f"{GROUP_LABELS.get(group, group)} (n={totals.get(group, 0):,})")
        for x, v in zip(xs, pivot[group]):
            ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in pivot.index])
    ax.set_ylabel("Share of the firm group's lead-sponsor trials (%)")
    ax.set_title(f"Clinical-trial phase distribution by firm size, "
                 f"{START_YEAR}\u2013{END_YEAR}")
    ax.set_ylim(0, min(100, pivot.to_numpy().max() * 1.25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# =====================================================================
# Entry point
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", choices=["trials", "firms", "exhibit", "all"],
                    default="all")
    ap.add_argument("--force", action="store_true",
                    help="rebuild a stage even if its output already exists")
    args = ap.parse_args()

    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    trials_path, firms_path = INTERIM / "trials", INTERIM / "firm_year"

    if args.stage in ("trials", "all"):
        if args.force or not table_exists(trials_path):
            log.info("wrote %s", write_table(build_trials(), trials_path))
        else:
            log.info("trials table exists; skipping (use --force to rebuild)")

    if args.stage in ("firms", "all"):
        if args.force or not table_exists(firms_path):
            fy, xwalk = build_firm_years(read_table(trials_path))
            log.info("wrote %s", write_table(xwalk, INTERIM / "firm_crosswalk"))
            log.info("wrote %s", write_table(fy, firms_path))
        else:
            log.info("firm_year table exists; skipping (use --force to rebuild)")

    if args.stage in ("exhibit", "all"):
        counts, sample = build_exhibit(read_table(trials_path), read_table(firms_path))
        sample.to_csv(REPORTS / "sample_construction.csv", index=False)
        log.info("sample construction:\n%s", sample.to_string(index=False))
        if counts.empty:
            log.warning("no lead-sponsor trials matched a BIG_PHARMA/BIOTECH "
                        "firm-year -- nothing to plot")
            return
        counts.to_csv(REPORTS / "phase_by_firm_group.csv", index=False)
        plot_exhibit(counts, REPORTS / "phase_by_firm_group.png")
        log.info("wrote %s", REPORTS / "phase_by_firm_group.png")


if __name__ == "__main__":
    main()
