"""
SHADE – LinkedIn Intelligence Agent
Cleaned & wrapped in FastAPI.

Endpoints:
    POST /run       – Full B2C or B2B pipeline
    GET  /health    – Liveness check
"""

from __future__ import annotations

import os
import re
import uuid
import json
import time
import math
import random
import logging
import asyncio
import requests

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote, quote

from psycopg2 import connect
from psycopg2.extras import RealDictCursor

from cost_tracker import get_cost_tracker

import aiohttp
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# =============================================================================
# Configuration
# =============================================================================

HARDCODE = True

GOOGLE_API_KEY_HARDCODE = "AIzaSyBGJGT2qvuhEHb3TnyhFGvSb4H2L8oddqE"
GOOGLE_CSE_ID_HARDCODE  = "17ec2b63245fa48f8"

BRIGHT_DATA_API_KEY_HARDCODE    = "b303229c-f60a-43dc-b8c6-d5d7bdecb9a1"
BRIGHT_DATA_DATASET_ID_HARDCODE = "gd_l1viktl72bvl7bjuj0"

AZURE_OPENAI_API_KEY_HARDCODE       = "2be1544b3dc14327b60a870fe8b94f35"
AZURE_OPENAI_ENDPOINT_HARDCODE      = "https://notedai.openai.azure.com"
AZURE_OPENAI_DEPLOYMENT_ID_HARDCODE = "gpt-4o"
AZURE_OPENAI_API_VERSION_HARDCODE   = "2024-06-01"

SPECTRE_DB_URL_HARDCODE = (
    "postgresql://monsteradmin:M0nsteradmin"
    "@monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require"
)

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SHADE - %(levelname)s - %(message)s",
)
logger = logging.getLogger("shade")

tracker = get_cost_tracker()

# =============================================================================
# Per-Run Cost Tracking
# =============================================================================

# Pricing constants (USD) — adjust as your contracts change
GPT4O_INPUT_COST_PER_1K   = 0.0025    # $2.50 / 1M input tokens
GPT4O_OUTPUT_COST_PER_1K  = 0.0100    # $10.00 / 1M output tokens
BRIGHT_DATA_COST_PER_ROW  = 0.010     # ~$0.01 per successful LinkedIn record
GOOGLE_CSE_COST_PER_QUERY = 0.005     # $5 / 1,000 queries


class RunCostTracker:
    """Accumulates costs for a single SHADE run."""

    def __init__(self):
        self.gpt_prompt_tokens: int = 0
        self.gpt_completion_tokens: int = 0
        self.gpt_calls: int = 0
        self.bright_data_rows: int = 0
        self.google_queries: int = 0

    def track_gpt(self, prompt_tokens: int, completion_tokens: int):
        self.gpt_prompt_tokens += prompt_tokens
        self.gpt_completion_tokens += completion_tokens
        self.gpt_calls += 1

    def track_bright(self, rows: int):
        self.bright_data_rows += rows

    def track_google(self, queries: int = 1):
        self.google_queries += queries

    def summary(self) -> Dict[str, Any]:
        gpt_input_cost  = (self.gpt_prompt_tokens / 1000) * GPT4O_INPUT_COST_PER_1K
        gpt_output_cost = (self.gpt_completion_tokens / 1000) * GPT4O_OUTPUT_COST_PER_1K
        gpt_total       = gpt_input_cost + gpt_output_cost
        bright_cost     = self.bright_data_rows * BRIGHT_DATA_COST_PER_ROW
        google_cost     = self.google_queries * GOOGLE_CSE_COST_PER_QUERY
        total           = gpt_total + bright_cost + google_cost

        return {
            "total_cost_usd": round(total, 6),
            "gpt": {
                "calls": self.gpt_calls,
                "prompt_tokens": self.gpt_prompt_tokens,
                "completion_tokens": self.gpt_completion_tokens,
                "total_tokens": self.gpt_prompt_tokens + self.gpt_completion_tokens,
                "input_cost_usd": round(gpt_input_cost, 6),
                "output_cost_usd": round(gpt_output_cost, 6),
                "total_cost_usd": round(gpt_total, 6),
            },
            "bright_data": {
                "profiles_scraped": self.bright_data_rows,
                "cost_usd": round(bright_cost, 6),
            },
            "google_cse": {
                "queries": self.google_queries,
                "cost_usd": round(google_cost, 6),
            },
        }


# Module-level per-run tracker (reset at the start of each run)
_run_costs: Optional[RunCostTracker] = None


def _rc() -> Optional[RunCostTracker]:
    """Shorthand to get the current run cost tracker."""
    return _run_costs


# =============================================================================
# Helpers
# =============================================================================


def _cfg(env_key: str, hardcode_val: str) -> str:
    """Return hardcoded value when HARDCODE is True, else read env."""
    if HARDCODE:
        return hardcode_val
    return os.getenv(env_key, "")


def get_db_connection():
    db_url = _cfg("SPECTRE_DB_URL", SPECTRE_DB_URL_HARDCODE)
    if not db_url:
        raise RuntimeError("SPECTRE_DB_URL is not set in environment")
    conn = connect(db_url, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_") or "company"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_name(n: str) -> str:
    return re.sub(r"[^\w\s]", "", (n or "").lower()).strip()


def _name_token_set(n: str) -> set:
    return set(_normalize_name(n).split())


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _guess_name_from_linkedin_url(url: str) -> str:
    try:
        parts = [x for x in urlparse(url).path.split("/") if x]
        slug = parts[1] if len(parts) >= 2 and parts[0] == "in" else (parts[0] if parts else "")
        slug = unquote(slug).split("?")[0].strip("-_/")
        slug = re.sub(r"-\d+$", "", slug)
        name = slug.replace("-", " ").replace("_", " ").strip()
        return " ".join(w.capitalize() for w in name.split()) if name else "(seed from URL)"
    except Exception:
        return "(seed from URL)"


def _normalize_education(edu_raw: Optional[Any]) -> List[Dict[str, Any]]:
    if not edu_raw:
        return []
    if isinstance(edu_raw, list):
        return [e for e in edu_raw if isinstance(e, dict)]
    return [edu_raw] if isinstance(edu_raw, dict) else []


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RawEmployee:
    name: str
    linkedin_url: str
    snippet: str
    company: str
    is_seed: bool = False


@dataclass
class CompanyData:
    name: str
    website: str = ""
    description: str = ""
    industry: str = ""
    headquarters: str = ""
    founded_year: str = ""
    employee_estimate: str = ""
    revenue_estimate: str = ""
    funding_info: str = ""
    tech_stack: List[str] = field(default_factory=list)
    social_links: Dict[str, str] = field(default_factory=dict)
    recent_news: List[Dict[str, str]] = field(default_factory=list)
    financial_data: Dict[str, Any] = field(default_factory=dict)
    business_model: str = ""
    key_products: str = ""
    market_position: str = ""
    competitive_analysis: str = ""
    growth_metrics: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()


# =============================================================================
# Google CSE Employee Finder
# =============================================================================


class GoogleCSEEmployeeFinder:
    def __init__(self):
        self.api_key = _cfg("GOOGLE_API_KEY", GOOGLE_API_KEY_HARDCODE)
        self.cse_id  = _cfg("GOOGLE_CSE_ID", GOOGLE_CSE_ID_HARDCODE)
        if not self.api_key or not self.cse_id:
            raise ValueError("Google CSE keys missing.")
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        })

    # ---- public ----

    def find_employees(self, company_name: str, max_results: int = 50) -> List[RawEmployee]:
        if max_results <= 0:
            return []
        query = f'site:linkedin.com/in "{company_name}"'
        logger.info(f"🔎 Discovery query: {query} (limit={max_results})")
        out: List[RawEmployee] = []
        start_index = 1
        for _ in range(math.ceil(max_results / 10)):
            if len(out) >= max_results:
                break
            items = self._search_page(query, start_index)
            if not items:
                break
            for it in items:
                if len(out) >= max_results:
                    break
                emp = self._extract_employee(it, company_name, skip_c_suite=True)
                if emp:
                    out.append(emp)
            start_index += 10
            time.sleep(random.uniform(0.5, 1.0))
        logger.info(f"🎯 Discovery found {len(out)} employees")
        return out

    def find_employee_by_name_improved(self, company_name: str, full_name: str) -> Optional[RawEmployee]:
        full_name = (full_name or "").strip()
        if not full_name:
            return None
        clean = _normalize_name(full_name)
        parts = clean.split()
        variants = [
            f'site:linkedin.com/in "{full_name}" "{company_name}"',
            f'site:linkedin.com/in "{full_name}" {company_name}',
            f'"{full_name}" "{company_name}" site:linkedin.com',
            f'"{full_name}" {company_name} linkedin',
        ]
        if len(parts) >= 2:
            fn, ln = parts[0], parts[-1]
            variants += [
                f'site:linkedin.com/in "{fn} {ln}" "{company_name}"',
                f'site:linkedin.com/in {quote(fn)} {quote(ln)} "{company_name}"',
            ]
        for vi, q in enumerate(_dedupe_preserve_order(variants), 1):
            for start in (1, 11, 21):
                logger.info(f"🔎 Seed search v{vi}, start={start}: {q}")
                for it in self._search_page(q, start) or []:
                    emp = self._extract_employee(it, company_name, skip_c_suite=False)
                    if emp and self._name_matches(emp.name, full_name):
                        emp.is_seed = True
                        logger.info(f"✅ SEED MATCH: {emp.name} ({emp.linkedin_url})")
                        return emp
                time.sleep(0.5)
        logger.warning(f"❌ Seed not found: {full_name}")
        return None

    # ---- private ----

    def _name_matches(self, found_name: str, target_name: str) -> bool:
        found, target = _name_token_set(found_name), _name_token_set(target_name)
        if not target:
            return False
        if found == target:
            return True
        return len(found & target) >= min(2, len(target)) if len(target) > 1 else len(found & target) >= 1

    def _search_page(self, query: str, start_index: int) -> List[Dict[str, Any]]:
        base_params = {"key": self.api_key, "cx": self.cse_id, "q": query, "start": start_index, "num": 10}
        masked_params = {**base_params, "fields": "items(title,link,snippet),searchInformation(totalResults)"}

        for attempt in range(3):
            try:
                params = masked_params if attempt == 0 else base_params
                r = self.session.get(self.base_url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()

                if isinstance(data, dict) and "error" in data:
                    err_msg = str(data["error"])
                    if "Invalid field selection" in err_msg and "fields" in params:
                        data = self.session.get(self.base_url, params=base_params, timeout=30).json()
                        if "error" in data:
                            return []
                    else:
                        return []

                items = (data or {}).get("items", [])
                if items:
                    tracker.track_google_query(num_items=len(items))
                    if _rc():
                        _rc().track_google()
                return items

            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                body = getattr(e.response, "text", "") or ""
                if status == 400 and "Invalid field selection" in body and attempt == 0:
                    continue
                if status and status != 429 and 400 <= status < 500:
                    return []
                time.sleep(0.5 * (attempt + 1))
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return []

    def _extract_employee(self, result: Dict[str, Any], company_name: str, skip_c_suite: bool = True) -> Optional[RawEmployee]:
        try:
            link = result.get("link", "")
            if "linkedin.com/in/" not in link:
                return None
            title = result.get("title", "") or ""
            name = self._extract_name_from_title(title)
            if not name:
                return None
            if skip_c_suite:
                t = title.upper()
                if any(x in t for x in (" CEO ", "CFO", "CTO", "COO", "CMO", "CIO", "CHRO", "CXO", "CPO ", " CDO ")):
                    return None
            snippet = self._clean_snippet(result.get("snippet", "") or "")
            return RawEmployee(name=name, linkedin_url=link, snippet=snippet, company=company_name)
        except Exception:
            return None

    def _extract_name_from_title(self, title: str) -> str:
        if not title:
            return ""
        parts = re.split(r"[|\-–—]", title)
        if parts:
            candidate = parts[0].strip()
            if candidate and "linkedin" not in candidate.lower():
                return candidate
        return title.replace("LinkedIn", "").strip()

    def _clean_snippet(self, s: str) -> str:
        if not s:
            return ""
        for j in [
            "LinkedIn is the world's largest professional network",
            "View the profiles of professionals named",
            "View the profiles of people named",
            "There are ", " professionals named", "on LinkedIn.",
        ]:
            s = s.replace(j, "")
        return re.sub(r"\s+", " ", s).strip()


# =============================================================================
# Bright Data Scraper (Dataset Filter – cached, fast)
# =============================================================================


class BrightDataScraper:
    def __init__(self):
        self.api_key = _cfg("BRIGHT_DATA_API_KEY", BRIGHT_DATA_API_KEY_HARDCODE)
        self.dataset_id = _cfg("BRIGHT_DATA_DATASET_ID", BRIGHT_DATA_DATASET_ID_HARDCODE)
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def scrape_profiles_one_shot(self, urls: List[str], timeout_sec: int = 300) -> List[Dict[str, Any]]:
        if not (self.api_key and self.dataset_id):
            logger.warning("Bright Data not configured")
            return []
        urls = _dedupe_preserve_order([u for u in urls if u])[:100]
        if not urls:
            return []
        logger.info(f"Processing {len(urls)} URLs with Bright Data...")
        results: List[Dict[str, Any]] = []
        for url in urls:
            result = self._fetch_profile_for_url(url)
            if result.get("profiles"):
                results.extend(result["profiles"])
        logger.info(f"Bright Data collected {len(results)} total profiles")
        tracker.track_bright_data_rows(num_rows=len(results))
        if _rc():
            _rc().track_bright(len(results))
        return results

    def _extract_slug(self, url: str) -> Optional[str]:
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        parts = [p for p in urlparse(url).path.split("/") if p]
        return parts[-1] if parts else None

    def _fetch_profile_for_url(self, url: str) -> dict:
        overall_start = time.time()
        slug = self._extract_slug(url)
        if not slug:
            return {"url": url, "slug": None, "profiles": [], "error": "slug_extraction_failed", "runtime": 0.0}

        logger.info(f"Starting lookup for {url} (slug: {slug})")

        # Trigger filter job
        try:
            resp = requests.post(
                "https://api.brightdata.com/datasets/filter",
                headers=self.headers,
                json={"dataset_id": self.dataset_id, "records_limit": 1000,
                      "filter": {"name": "url", "operator": "includes", "value": slug}},
                timeout=30,
            )
            if resp.status_code != 200:
                return {"url": url, "slug": slug, "profiles": [], "error": f"trigger_error_{resp.status_code}", "runtime": time.time() - overall_start}
            snapshot_id = resp.json().get("snapshot_id")
        except Exception as e:
            return {"url": url, "slug": slug, "profiles": [], "error": str(e), "runtime": time.time() - overall_start}

        # Poll for ready
        status_url = f"https://api.brightdata.com/datasets/snapshots/{snapshot_id}"
        job_start = time.time()
        while time.time() - job_start < 300:
            try:
                sr = requests.get(status_url, headers=self.headers, timeout=15)
                if sr.status_code == 200:
                    st = sr.json().get("status")
                    if st == "ready":
                        break
                    if st == "failed":
                        return {"url": url, "slug": slug, "profiles": [], "error": "snapshot_failed", "runtime": time.time() - overall_start}
            except Exception:
                pass
            time.sleep(5)
        else:
            return {"url": url, "slug": slug, "profiles": [], "error": "snapshot_timeout", "runtime": time.time() - overall_start}

        # Download
        dl_url = f"https://api.brightdata.com/datasets/snapshots/{snapshot_id}/download?format=json"
        while True:
            try:
                dr = requests.get(dl_url, headers=self.headers, timeout=30)
                if dr.status_code == 200:
                    break
                if dr.status_code == 202:
                    time.sleep(5)
                    continue
                return {"url": url, "slug": slug, "profiles": [], "error": f"download_error_{dr.status_code}", "runtime": time.time() - overall_start}
            except Exception as e:
                return {"url": url, "slug": slug, "profiles": [], "error": str(e), "runtime": time.time() - overall_start}

        try:
            profiles = dr.json()
        except Exception:
            return {"url": url, "slug": slug, "profiles": [], "error": "json_parse_error", "runtime": time.time() - overall_start}

        return {"url": url, "slug": slug, "profiles": profiles, "error": None, "runtime": time.time() - overall_start}


# =============================================================================
# Bright Data Trigger Scraper (fresh scrape, slower)
# =============================================================================


class BrightDataTriggerScraper:
    def __init__(self):
        self.api_key = _cfg("BRIGHT_DATA_API_KEY", BRIGHT_DATA_API_KEY_HARDCODE)
        self.dataset_id = _cfg("BRIGHT_DATA_DATASET_ID", BRIGHT_DATA_DATASET_ID_HARDCODE)
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def is_configured(self) -> bool:
        return bool(self.api_key and self.dataset_id)

    def scrape_profiles_trigger(self, urls: List[str], timeout_sec: int = 300) -> List[Dict[str, Any]]:
        if not self.is_configured():
            return []
        urls = _dedupe_preserve_order([u for u in urls if u])[:100]
        if not urls:
            return []
        logger.info(f"🔄 TRIGGER: Fresh scrape for {len(urls)} URLs...")
        snap = self._trigger(urls)
        if not snap:
            return []
        if not self._wait_ready(snap, timeout_sec, interval=10):
            return []
        profiles = self._fetch(snap)
        logger.info(f"✅ TRIGGER: Got {len(profiles)} fresh profiles")
        return profiles

    def _trigger(self, urls: List[str]) -> Optional[str]:
        try:
            r = requests.post(
                f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={self.dataset_id}&include_errors=true",
                headers=self.headers, json=[{"url": u} for u in urls], timeout=30,
            )
            if r.ok:
                js = r.json()
                return js.get("snapshot_id") or js.get("snapshot") or js.get("id")
        except Exception as e:
            logger.error(f"TRIGGER exception: {e}")
        return None

    def _wait_ready(self, snap: str, timeout: int, interval: int) -> bool:
        elapsed = 0
        while elapsed <= timeout:
            try:
                r = requests.get(
                    f"https://api.brightdata.com/datasets/v3/progress/{snap}",
                    headers=self.headers, timeout=15,
                )
                if r.ok:
                    st = (r.json().get("status") or r.json().get("state") or "").lower()
                    if st == "ready":
                        return True
                    if st == "error":
                        return False
            except Exception:
                pass
            time.sleep(interval)
            elapsed += interval
        return False

    def _fetch(self, snap: str) -> List[Dict[str, Any]]:
        try:
            r = requests.get(
                f"https://api.brightdata.com/datasets/v3/snapshot/{snap}",
                headers=self.headers, timeout=120,
            )
            if not r.ok:
                return []
            data = []
            for line in r.text.splitlines():
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
            tracker.track_bright_data_rows(num_rows=len(data))
            if _rc():
                _rc().track_bright(len(data))
            return data
        except Exception:
            return []


# =============================================================================
# Parallel scrape (dataset + trigger)
# =============================================================================


async def scrape_profiles_parallel(urls: List[str], timeout_sec: int = 300) -> List[Dict[str, Any]]:
    urls = _dedupe_preserve_order([u for u in urls if u])
    if not urls:
        return []
    ds = BrightDataScraper()
    ts = BrightDataTriggerScraper()
    results = await asyncio.gather(
        asyncio.to_thread(ds.scrape_profiles_one_shot, urls, timeout_sec),
        asyncio.to_thread(ts.scrape_profiles_trigger, urls, timeout_sec),
        return_exceptions=True,
    )
    dataset_profiles = results[0] if isinstance(results[0], list) else []
    trigger_profiles = results[1] if isinstance(results[1], list) else []

    def _url_key(p: Dict[str, Any]) -> str:
        u = p.get("url") or p.get("profile_url") or p.get("linkedin_url") or p.get("input_url") or ""
        return u.lower().split("?")[0].rstrip("/") if u else ""

    merged: Dict[str, Dict[str, Any]] = {}
    for p in dataset_profiles:
        k = _url_key(p)
        if k:
            merged[k] = p
    for p in trigger_profiles:
        k = _url_key(p)
        if k:
            merged[k] = p  # fresher data wins
    return list(merged.values())


# =============================================================================
# Bright-profile cleaner (deterministic, no GPT)
# =============================================================================


def clean_bright_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    current_company = raw.get("current_company") or {}

    raw_profile: Dict[str, Any] = {
        "id": raw.get("id", ""),
        "url": raw.get("url", ""),
        "input_url": raw.get("input_url", ""),
        "linkedin_id": raw.get("linkedin_id", raw.get("id", "")),
        "linkedin_num_id": raw.get("linkedin_num_id", ""),
        "first_name": raw.get("first_name", ""),
        "last_name": raw.get("last_name", ""),
        "name": raw.get("name", ""),
        "position": raw.get("position", ""),
        "about": raw.get("about", ""),
        "avatar": raw.get("avatar", ""),
        "banner_image": raw.get("banner_image", ""),
        "default_avatar": bool(raw.get("default_avatar", False)),
        "location": raw.get("location", ""),
        "city": raw.get("city", ""),
        "country_code": raw.get("country_code", ""),
        "followers": int(raw.get("followers", 0) or 0),
        "connections": int(raw.get("connections", 0) or 0),
        "current_company": {
            "link": current_company.get("link", ""),
            "name": current_company.get("name", ""),
            "location": current_company.get("location"),
            "company_id": current_company.get("company_id", ""),
        },
        "current_company_name": raw.get("current_company_name", ""),
        "current_company_company_id": raw.get("current_company_company_id", ""),
        "posts": raw.get("posts"),
        "courses": raw.get("courses"),
        "patents": raw.get("patents"),
        "projects": [
            {"title": p.get("title", ""), "start_date": p.get("start_date", ""),
             "end_date": p.get("end_date", ""), "description": p.get("description", "")}
            for p in (raw.get("projects") or [])
        ],
        "bio_links": raw.get("bio_links") or [],
        "education": [
            {"url": edu.get("url", ""), "title": edu.get("title", ""),
             "start_year": edu.get("start_year", ""), "end_year": edu.get("end_year", ""),
             "description": edu.get("description"),
             "description_html": edu.get("description_html"),
             "institute_logo_url": edu.get("institute_logo_url")}
            for edu in _normalize_education(raw.get("education"))
        ],
        "educations_details": raw.get("educations_details", ""),
        "experience": [
            {"url": ex.get("url", ""), "title": ex.get("title", ""),
             "company": ex.get("company", ""), "company_id": ex.get("company_id", ""),
             "location": ex.get("location", ""), "start_date": ex.get("start_date", ""),
             "end_date": ex.get("end_date", ""), "description": ex.get("description", ""),
             "description_html": ex.get("description_html", ""),
             "company_logo_url": ex.get("company_logo_url", "")}
            for ex in (raw.get("experience") or [])
        ],
        "languages": [
            {"title": lang.get("title", ""), "subtitle": lang.get("subtitle", "")}
            for lang in (raw.get("languages") or [])
        ],
        "certifications": [
            {"meta": c.get("meta", ""), "title": c.get("title", ""),
             "subtitle": c.get("subtitle", ""), "credential_id": c.get("credential_id", ""),
             "credential_url": c.get("credential_url", "")}
            for c in (raw.get("certifications") or [])
        ],
        "activity": [
            {"id": a.get("id", ""), "img": a.get("img"), "link": a.get("link", ""),
             "title": a.get("title", ""), "interaction": a.get("interaction", "")}
            for a in (raw.get("activity") or [])
        ],
        "publications": raw.get("publications"),
        "organizations": raw.get("organizations"),
        "honors_and_awards": raw.get("honors_and_awards"),
        "volunteer_experience": raw.get("volunteer_experience"),
        "people_also_viewed": raw.get("people_also_viewed"),
        "similar_profiles": raw.get("similar_profiles") or [],
        "recommendations": raw.get("recommendations"),
        "recommendations_count": raw.get("recommendations_count"),
        "memorialized_account": bool(raw.get("memorialized_account", False)),
    }

    # ---- vitals ----
    name = raw.get("name", "")
    title = raw.get("position", "")
    company_name = raw.get("current_company_name") or current_company.get("name", "")

    vitals_experience: List[Dict[str, Any]] = []
    for ex in (raw.get("experience") or []):
        for pos in (ex.get("positions") or [ex]):
            start = pos.get("start_date") or ex.get("start_date") or ""
            end = pos.get("end_date") or ex.get("end_date") or ""
            vitals_experience.append({
                "title": pos.get("title", "") or ex.get("title", ""),
                "company": ex.get("company", ""),
                "company_id": ex.get("company_id", ""),
                "location": pos.get("location", "") or ex.get("location", ""),
                "description": pos.get("description", "") or ex.get("description", ""),
                "description_html": pos.get("description_html", "") or ex.get("description_html", ""),
                "start_date": start, "end_date": end,
                "is_current": end in ("Present", "present", "", None),
                "duration_months": 0,
            })

    vitals: Dict[str, Any] = {
        "employee_id": "",
        "name": name, "company": company_name, "title": title, "position": title,
        "current_company": {"name": company_name, "title": title},
        "experience": vitals_experience,
        "certifications": [{"title": c.get("title", "")} for c in (raw.get("certifications") or [])],
        "honors_and_awards": (
            [{"title": h.get("title", "")} for h in raw.get("honors_and_awards")]
            if isinstance(raw.get("honors_and_awards"), list) else []
        ),
        "languages": [
            {"name": lang.get("title", ""), "proficiency": lang.get("subtitle", "")}
            for lang in (raw.get("languages") or [])
        ],
        "education": [
            {"title": edu.get("title", ""), "degree": "", "field": ""}
            for edu in _normalize_education(raw.get("education"))
        ],
        "about": raw.get("about", ""),
        "courses": [],
        "activity": [
            {"title": a.get("title", ""), "interaction": a.get("interaction", "")}
            for a in (raw.get("activity") or [])
        ],
        "posts": [],
        "recommendations": [
            {"text": rec, "type": "", "giver": ""}
            for rec in (raw.get("recommendations") or []) if isinstance(rec, str)
        ],
        "linkedin_skills": [], "top_skills": [],
        "match_info": {"target_employee": name, "similarity_score": "", "match_rationale": ""},
        "city": raw.get("city", ""), "country_code": raw.get("country_code", ""),
        "location": raw.get("location", ""),
        "followers": int(raw.get("followers", 0) or 0),
        "connections": int(raw.get("connections", 0) or 0),
    }
    return {"raw_profile": raw_profile, "vitals": vitals}


# =============================================================================
# Company Research
# =============================================================================


class CompanyReportGenerator:
    def __init__(self):
        self.azure_api_key       = _cfg("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY_HARDCODE)
        self.azure_endpoint      = _cfg("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT_HARDCODE)
        self.azure_deployment_id = _cfg("AZURE_OPENAI_DEPLOYMENT_ID", AZURE_OPENAI_DEPLOYMENT_ID_HARDCODE)
        self.azure_api_version   = _cfg("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION_HARDCODE)
        self.google_api_key      = _cfg("GOOGLE_API_KEY", GOOGLE_API_KEY_HARDCODE)
        self.google_cx           = _cfg("GOOGLE_CSE_ID", GOOGLE_CSE_ID_HARDCODE)
        self.max_links   = 5
        self.timeout     = 20
        self.delay       = 0.5
        self.concurrency = 6
        self.ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

    async def generate_company_report(self, company_name: str) -> CompanyData:
        logger.info(f"🏢 Company research for: {company_name}")
        queries = [
            f'"{company_name}" official website',
            f'"{company_name}" press release',
            f'"{company_name}" about',
            f'"{company_name}" headquarters location',
            f'"{company_name}" revenue ARR',
            f'"{company_name}" funding crunchbase',
            f'"{company_name}" LinkedIn',
        ]
        urls: List[str] = []
        tasks = [asyncio.create_task(self._google_search(q)) for q in queries]
        for r in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(r, list):
                urls.extend(r)
        urls = _dedupe_preserve_order(urls)[:20]
        if not urls:
            return CompanyData(name=company_name)

        sem = asyncio.Semaphore(self.concurrency)
        pages: List[Dict[str, Any]] = []

        async def worker(u: str):
            async with sem:
                data = await self._fetch_page(u)
                await asyncio.sleep(self.delay)
                if data:
                    pages.append(data)

        await asyncio.gather(*[asyncio.create_task(worker(u)) for u in urls])
        if not pages:
            return CompanyData(name=company_name)

        cd = CompanyData(name=company_name, website=pages[0]["url"])
        for p in pages:
            for k, v in (p.get("social_links") or {}).items():
                cd.social_links.setdefault(k, v)
            info = p.get("company_info") or {}
            if info.get("revenue_estimate") and not cd.revenue_estimate:
                cd.revenue_estimate = info["revenue_estimate"]
            if info.get("employee_estimate") and not cd.employee_estimate:
                cd.employee_estimate = info["employee_estimate"]
            if info.get("founded_year") and not cd.founded_year:
                cd.founded_year = info["founded_year"]
            if info.get("headquarters") and not cd.headquarters:
                cd.headquarters = info["headquarters"]

        cd.description = f"Automated web scan from {len(pages)} sources. Social: {', '.join(cd.social_links.keys()) or 'none'}."

        if self.azure_api_key and self.azure_endpoint:
            try:
                chunk = "\n\n".join(
                    [f"Title: {p['title']}\nDesc: {p['description']}\nContent: {p['content'][:2000]}" for p in pages[:3]]
                )[:8000]
                ai_sum = await self._azure_summary(
                    f"Provide a concise factual profile of {company_name}: industry, products, business model, position, notable metrics.\n\nContext:\n{chunk}"
                )
                if ai_sum:
                    cd.description = ai_sum
                    cd.business_model = ai_sum
            except Exception as e:
                logger.warning(f"Azure summary failed: {e}")
        return cd

    async def _google_search(self, q: str) -> List[str]:
        if not (self.google_api_key and self.google_cx):
            return []
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"q": q, "key": self.google_api_key, "cx": self.google_cx, "num": self.max_links},
                    timeout=self.timeout,
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    items = data.get("items", [])[:self.max_links]
                    tracker.track_google_query(num_items=len(items))
                    if _rc():
                        _rc().track_google()
                    return [it["link"] for it in items if it.get("link")]
        except Exception:
            return []

    async def _fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, headers={"User-Agent": self.ua}, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            meta = soup.find("meta", attrs={"name": "description"})
            desc = meta.get("content", "").strip() if meta else ""
            content = ""
            for css in ["main", "article", ".content", ".post-content", "#content", ".main-content", "body"]:
                node = soup.select_one(css)
                if node:
                    content = node.get_text(" ", strip=True)
                    break
            if not content:
                content = soup.get_text(" ", strip=True)
            content = re.sub(r"\s+", " ", content)[:15000]
            return {
                "url": url, "title": title, "description": desc, "content": content,
                "social_links": self._extract_social(soup),
                "company_info": self._extract_info(content),
            }
        except Exception:
            return None

    def _extract_social(self, soup: BeautifulSoup) -> Dict[str, str]:
        out: Dict[str, str] = {}
        mapping = {
            "linkedin.com": "linkedin", "twitter.com": "twitter", "x.com": "twitter",
            "facebook.com": "facebook", "instagram.com": "instagram",
            "youtube.com": "youtube", "crunchbase.com": "crunchbase", "github.com": "github",
        }
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            for domain, key in mapping.items():
                if domain in href and key not in out:
                    out[key] = a["href"]
                    break
        return out

    def _extract_info(self, content: str) -> Dict[str, str]:
        info: Dict[str, str] = {}
        rev = re.search(r"\$\s?(\d+(?:\.\d+)?)\s*(billion|million|b|m)\b.*?(revenue|arr|sales)", content, re.I)
        if rev:
            info["revenue_estimate"] = rev.group(0)
        emp = re.search(r"(\d{2,3}(?:,\d{3})?)\s*(employees|people|team)", content, re.I)
        if emp:
            info["employee_estimate"] = emp.group(0)
        fy = re.search(r"(founded|established|launched)\s+(in\s+)?(19|20)\d{2}", content, re.I)
        if fy:
            info["founded_year"] = re.search(r"(19|20)\d{2}", fy.group(0)).group(0)
        hq = re.search(r"(headquarters|based)\s+(in|at)\s+([A-Z][A-Za-z]+(?:,\s*[A-Z][A-Za-z]+)?)", content, re.I)
        if hq:
            info["headquarters"] = hq.group(3)
        return info

    async def _azure_summary(self, user_content: str) -> str:
        if not (self.azure_api_key and self.azure_endpoint and self.azure_deployment_id):
            return ""
        url = (
            f"{self.azure_endpoint}/openai/deployments/"
            f"{self.azure_deployment_id}/chat/completions"
            f"?api-version={self.azure_api_version}"
        )
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    url,
                    headers={"Content-Type": "application/json", "api-key": self.azure_api_key},
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a precise, fact-focused summarizer. Given company information, produce a concise, factual overview. Do NOT hallucinate."},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.2, "top_p": 0.95, "max_tokens": 600,
                    },
                    timeout=30,
                ) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    usage = (data or {}).get("usage", {})
                    if usage.get("prompt_tokens") or usage.get("completion_tokens"):
                        pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                        tracker.track_gpt_tokens(pt, ct)
                        if _rc():
                            _rc().track_gpt(pt, ct)
                    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            return ""


# =============================================================================
# Comprehensive Data Manager (DB writer)
# =============================================================================


class ComprehensiveDataManager:

    @dataclass
    class StructuredBrightFields:
        current_title: str = ""
        current_company: str = ""
        first_name: str = ""
        last_name: str = ""
        location: str = ""
        city: str = ""
        country_code: str = ""
        profile_url: str = ""
        linkedin_profile_id: str = ""
        current_company_company_id: str = ""

    def __init__(self):
        self.azure_api_key       = _cfg("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY_HARDCODE)
        self.azure_endpoint      = _cfg("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT_HARDCODE)
        self.azure_deployment_id = _cfg("AZURE_OPENAI_DEPLOYMENT_ID", AZURE_OPENAI_DEPLOYMENT_ID_HARDCODE)
        self.azure_api_version   = _cfg("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION_HARDCODE)

    # ---- URL / matching helpers ----

    def _normalize_linkedin_url(self, url: str) -> str:
        if not url:
            return ""
        url = url.strip()
        if "linkedin.com" not in url:
            return url
        try:
            parts = [x for x in urlparse(url).path.split("/") if x]
            slug = parts[1] if len(parts) >= 2 and parts[0] == "in" else (parts[0] if parts else "")
            slug = unquote(slug).split("?")[0].split("#")[0].strip("/")
            slug = re.sub(r"-\d+$", "", slug)
            return f"linkedin.com/in/{slug.lower()}" if slug else "linkedin.com"
        except Exception:
            return url

    def _extract_slug(self, url: str) -> str:
        norm = self._normalize_linkedin_url(url)
        parts = norm.split("/")
        return (parts[-1] or (parts[-2] if len(parts) >= 2 else "")).lower() if parts else ""

    def _fuzzy_match_profile(self, e: RawEmployee, profile_list: List[Dict[str, Any]], norm_url: str) -> Optional[Dict[str, Any]]:
        target_slug = self._extract_slug(norm_url)
        if target_slug:
            for p in profile_list:
                u = p.get("url") or p.get("profile_url") or p.get("linkedin_url") or p.get("link") or ""
                if "linkedin.com" in u and self._extract_slug(u) == target_slug:
                    return p
        target_tokens = _name_token_set(e.name)
        best, best_score = None, 0
        for p in profile_list:
            cand = p.get("full_name") or p.get("name") or p.get("profile_name") or ""
            score = len(target_tokens & _name_token_set(cand))
            if score > best_score:
                best_score, best = score, p
        return best if best and best_score >= 2 else None

    def _summarize_profile(self, detail: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
        full_name = detail.get("full_name") or detail.get("name") or detail.get("profile_name") or fallback_name
        headline = detail.get("occupation") or detail.get("headline") or detail.get("current_position") or ""
        location = detail.get("location") or detail.get("locationName") or detail.get("geo") or "Not available"
        positions = detail.get("positions") or detail.get("experience") or []
        exp_years = sum(_safe_int(p.get("duration_years") or p.get("durationYears") or 0) for p in positions) if isinstance(positions, list) else 0
        skills_raw = detail.get("skills") or []
        skills = skills_raw.get("values") if isinstance(skills_raw, dict) else (skills_raw if isinstance(skills_raw, list) else [])
        education = detail.get("education") or detail.get("educations") or []
        connections = detail.get("connections") or detail.get("connectionsCount") or detail.get("num_connections") or "Not available"
        return {
            "full_name": full_name,
            "current_position": headline or "Not available",
            "location": location or "Not available",
            "experience_years": exp_years,
            "skills_count": len(skills),
            "education_count": len(education) if isinstance(education, list) else 0,
            "connections": connections,
        }

    # ---- DB helpers ----

    def _pick_enum_value(self, cur, enum_type: str, preferred_labels: list, context: str) -> str:
        cur.execute("""
            SELECT e.enumlabel AS label FROM pg_type t
            JOIN pg_enum e ON t.oid = e.enumtypid
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE n.nspname = 'spectre' AND t.typname = %s
            ORDER BY e.enumsortorder;
        """, (enum_type,))
        allowed = [r["label"] for r in cur.fetchall()]
        if not allowed:
            raise RuntimeError(f"{context}: spectre.{enum_type} has no enum values.")
        for label in preferred_labels:
            if label in allowed:
                return label
        return allowed[0]

    def _upsert_company(self, cur, block: Dict[str, Any]) -> uuid.UUID:
        name = (block.get("name") or "").strip() or "Unknown Company"
        new_id = uuid.uuid4()
        shade_meta = {
            "source": "shade",
            "description": block.get("description", ""),
            "industry": block.get("industry", ""),
            "headquarters": block.get("headquarters", ""),
            "employee_estimate": block.get("employee_estimate", ""),
            "revenue_estimate": block.get("revenue_estimate", ""),
            "timestamp": block.get("timestamp") or _now_iso(),
        }
        cur.execute("""
            INSERT INTO spectre.companies (
                company_id, name, domain, linkedin_url, metadata_json, raw_json,
                description, industry, business_model
            ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
            ON CONFLICT (name) DO UPDATE SET
                domain=COALESCE(EXCLUDED.domain, spectre.companies.domain),
                linkedin_url=COALESCE(EXCLUDED.linkedin_url, spectre.companies.linkedin_url),
                metadata_json=COALESCE(spectre.companies.metadata_json,'{}'::jsonb)||COALESCE(EXCLUDED.metadata_json,'{}'::jsonb),
                raw_json=COALESCE(EXCLUDED.raw_json, spectre.companies.raw_json),
                description=COALESCE(EXCLUDED.description, spectre.companies.description),
                industry=COALESCE(EXCLUDED.industry, spectre.companies.industry),
                business_model=COALESCE(EXCLUDED.business_model, spectre.companies.business_model)
            RETURNING company_id;
        """, (
            str(new_id), name,
            (block.get("website") or block.get("domain") or "").strip() or None,
            (block.get("linkedin_url") or "").strip() or None,
            json.dumps({"shade": shade_meta}), json.dumps(block),
            (block.get("description") or "").strip() or None,
            (block.get("industry") or "").strip() or None,
            (block.get("business_model") or "").strip() or None,
        ))
        return cur.fetchone()["company_id"]

    def _insert_run(self, cur, mode, company_name, analytics, executive_summary, company_id, employee_count) -> uuid.UUID:
        run_id = uuid.uuid4()
        scope = self._pick_enum_value(cur, "run_scope", ["linkedin_employees", "employees", "shade"], "_insert_run")
        cur.execute("""
            INSERT INTO spectre.runs (
                run_id, scope, target_company_id, requested_employee_count,
                config_json, raw_json, started_at, ended_at
            ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,NOW(),NOW())
            RETURNING run_id;
        """, (
            str(run_id), scope, str(company_id), int(employee_count),
            json.dumps({"source": "shade", "mode": mode, "company_name": company_name, "requested_employee_count": int(employee_count)}),
            json.dumps({"analytics": analytics, "executive_summary": executive_summary}),
        ))
        return cur.fetchone()["run_id"]

    def _link_run_company(self, cur, run_id, company_id, mode):
        role = self._pick_enum_value(cur, "company_role_in_run", ["target", "primary"], "_link_run_company")
        try:
            cur.execute("""
                INSERT INTO spectre.run_companies (run_id, company_id, role_in_run, raw_json)
                VALUES (%s,%s,%s,%s::jsonb);
            """, (str(run_id), str(company_id), role, json.dumps({"source": "shade", "mode": mode})))
        except Exception:
            logger.warning("_link_run_company failed", exc_info=True)

    def _upsert_employee(self, cur, company_id, basic_info, summary=None) -> uuid.UUID:
        full_name = (basic_info.get("name") or "").strip() or "(Unknown)"
        linkedin_url = (basic_info.get("linkedin_url") or "").strip()
        if not linkedin_url:
            raise ValueError(f"Employee {full_name!r} missing linkedin_url.")
        current_title = (summary or {}).get("current_position") or basic_info.get("company")
        location = (summary or {}).get("location")
        profile_cache = (
            (summary or {}).get("detailed_professional_essay")
            or basic_info.get("gpt_detailed_essay")
            or (summary or {}).get("current_position")
            or basic_info.get("search_snippet")
        )
        new_id = uuid.uuid4()
        shade_meta = {
            "source": "shade", "is_seed": bool(basic_info.get("is_seed")),
            "search_snippet": basic_info.get("search_snippet", ""),
            "company": basic_info.get("company", ""), "summary": summary or {},
        }
        cur.execute("""
            INSERT INTO spectre.employees (
                employee_id, full_name, linkedin_url, current_company_id,
                current_title, location, metadata_json, profile_cache_text
            ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
            ON CONFLICT (linkedin_url) DO UPDATE SET
                full_name=EXCLUDED.full_name,
                current_company_id=EXCLUDED.current_company_id,
                current_title=COALESCE(EXCLUDED.current_title, spectre.employees.current_title),
                location=COALESCE(EXCLUDED.location, spectre.employees.location),
                metadata_json=COALESCE(spectre.employees.metadata_json,'{}'::jsonb)||COALESCE(EXCLUDED.metadata_json,'{}'::jsonb),
                profile_cache_text=COALESCE(EXCLUDED.profile_cache_text, spectre.employees.profile_cache_text)
            RETURNING employee_id;
        """, (
            str(new_id), full_name, linkedin_url, str(company_id),
            current_title, location, json.dumps({"shade": shade_meta}), profile_cache,
        ))
        return cur.fetchone()["employee_id"]

    def _link_run_employee(self, cur, run_id, employee_id, is_seed, company_id, basic_info, summary):
        role = self._pick_enum_value(cur, "employee_role_in_run", ["seed" if is_seed else "discovered"], "_link_run_employee")
        try:
            cur.execute("""
                INSERT INTO spectre.run_employees (run_id, employee_id, role_in_run, source_company_id, raw_json)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (run_id, employee_id) DO UPDATE SET
                    role_in_run=EXCLUDED.role_in_run, raw_json=EXCLUDED.raw_json;
            """, (
                str(run_id), str(employee_id), role, str(company_id),
                json.dumps({"source": "shade", "is_seed": is_seed, "basic_info": basic_info, "summary": summary}),
            ))
        except Exception:
            logger.warning("_link_run_employee failed", exc_info=True)

    # ---- Azure GPT helpers ----

    def _azure_call(self, messages: list, max_tokens: int = 800, temperature: float = 0.2) -> str:
        """Shared Azure OpenAI call. Returns content string or ''."""
        if not (self.azure_api_key and self.azure_endpoint and self.azure_deployment_id):
            return ""
        url = (
            f"{self.azure_endpoint}/openai/deployments/"
            f"{self.azure_deployment_id}/chat/completions"
            f"?api-version={self.azure_api_version}"
        )
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json", "api-key": self.azure_api_key},
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=40,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = (data or {}).get("usage", {})
            if usage.get("prompt_tokens") or usage.get("completion_tokens"):
                pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                tracker.track_gpt_tokens(pt, ct)
                if _rc():
                    _rc().track_gpt(pt, ct)
            return ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            logger.warning(f"Azure call failed: {e}")
            return ""

    def _strip_json_fence(self, s: str) -> str:
        s = (s or "").strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
            s = s.rstrip("`").strip()
        return s

    def _azure_enrich_employee(self, basic_info: Dict[str, Any], detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        context = {"basic_info": basic_info, "bright_profile": detail}
        prompt = (
            "You are an expert career intelligence analyst. "
            "Return a STRICT JSON object (no markdown) with keys:\n"
            "full_name, current_position, location, experience_years, skills_count, "
            "education_count, connections, detailed_professional_essay.\n"
            "Be factual and conservative.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)[:8000]}"
        )
        content = self._azure_call([
            {"role": "system", "content": "Return only valid JSON with the requested keys."},
            {"role": "user", "content": prompt},
        ])
        if not content:
            return None
        try:
            enriched = json.loads(self._strip_json_fence(content))
            if isinstance(enriched, dict):
                return enriched
        except Exception:
            pass
        return {
            "full_name": basic_info.get("name", "(Unknown)"), "current_position": "",
            "location": "", "experience_years": 0, "skills_count": 0,
            "education_count": 0, "connections": "Not available",
            "detailed_professional_essay": content,
        }

    def _extract_current_position_via_gpt(self, headline: str, experience: List[Dict[str, Any]], name: str) -> Dict[str, str]:
        if not (self.azure_api_key and self.azure_endpoint):
            if experience:
                return {"title": experience[0].get("title", ""), "company": experience[0].get("company", "")}
            return {"title": "", "company": ""}
        prompt = f"""Extract the CURRENT job title and company for {name}.
LinkedIn Headline: {headline}
Experience (most recent first):
{json.dumps(experience[:5], indent=2, ensure_ascii=False)[:3000]}
Return ONLY JSON: {{"title": "...", "company": "..."}}"""
        content = self._azure_call([
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ], max_tokens=150, temperature=0.1)
        try:
            r = json.loads(self._strip_json_fence(content))
            return {"title": r.get("title", ""), "company": r.get("company", "")}
        except Exception:
            if experience:
                return {"title": experience[0].get("title", ""), "company": experience[0].get("company", "")}
            return {"title": "", "company": ""}

    def _structure_bright_raw_via_gpt(self, raw_json: Dict[str, Any], basic_info: Dict[str, Any]) -> "ComprehensiveDataManager.StructuredBrightFields":
        target_name = (basic_info or {}).get("name", "") or (raw_json or {}).get("name", "")
        raw_dump = json.dumps(raw_json or {}, ensure_ascii=False)[:20000]
        prompt = f"""
Return ONLY a STRICT JSON object with EXACT keys:
{{"current_title":"","current_company":"","first_name":"","last_name":"","location":"","city":"","country_code":"","profile_url":"","linkedin_profile_id":"","current_company_company_id":""}}

Rules:
- current_title/current_company MUST come from the CURRENT job (end_date "Present"/empty or is_current=true).
- Do NOT copy headline if it includes multiple companies/buzzwords.
- If unknown, return "" (no hallucination).

Target person: {target_name}

RAW JSON:
{raw_dump}""".strip()

        content = self._azure_call([
            {"role": "system", "content": "Return only valid JSON. No markdown. No commentary."},
            {"role": "user", "content": prompt},
        ], max_tokens=400, temperature=0.0)
        try:
            obj = json.loads(self._strip_json_fence(content)) if content else {}
            return self.StructuredBrightFields(**{k: (obj.get(k) or "").strip() for k in self.StructuredBrightFields.__dataclass_fields__})
        except Exception:
            return self.StructuredBrightFields()

    # ---- Insert employee_details ----

    def _insert_employee_details(self, cur, run_id, employee_id, detailed_profile, summary, basic_info):
        cleaned = clean_bright_profile(detailed_profile or {})
        raw_profile = cleaned.get("raw_profile", {}) or {}
        vitals = cleaned.get("vitals", {}) or {}
        raw_json = detailed_profile or {}

        # Manual title override
        manual_override = (detailed_profile or {}).get("_manual_title_override")
        if manual_override:
            logger.info(f"🔧 Manual title override in employee_details: {manual_override}")
            vitals["title"] = manual_override
            vitals["position"] = manual_override

        # GPT structuring from full raw JSON
        structured = self._structure_bright_raw_via_gpt(raw_json, basic_info)
        if structured.current_title:
            vitals["title"] = structured.current_title
            vitals["position"] = structured.current_title
        if structured.current_company:
            vitals["company"] = structured.current_company
        if structured.first_name:
            raw_profile["first_name"] = structured.first_name
        if structured.last_name:
            raw_profile["last_name"] = structured.last_name
        if structured.profile_url:
            raw_profile["url"] = structured.profile_url
        if structured.linkedin_profile_id:
            raw_profile["linkedin_id"] = structured.linkedin_profile_id
        if structured.current_company_company_id:
            raw_profile["current_company_company_id"] = structured.current_company_company_id
        if structured.location and not vitals.get("location"):
            vitals["location"] = structured.location
        if structured.city and not vitals.get("city"):
            vitals["city"] = structured.city
        if structured.country_code and not vitals.get("country_code"):
            vitals["country_code"] = structured.country_code

        # Fallback GPT position extraction
        if not (vitals.get("title") and vitals.get("company")):
            pos = self._extract_current_position_via_gpt(
                vitals.get("position", "") or vitals.get("title", ""),
                vitals.get("experience", []),
                vitals.get("name", "") or basic_info.get("name", ""),
            )
            if not vitals.get("title") and pos.get("title"):
                vitals["title"] = pos["title"]
                vitals["position"] = pos["title"]
            if not vitals.get("company") and pos.get("company"):
                vitals["company"] = pos["company"]

        origin = self._pick_enum_value(cur, "data_origin", ["shade"], "_insert_employee_details")

        def to_json_text(val):
            if val is None:
                return None
            return json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)

        try:
            cur.execute("""
                INSERT INTO spectre.employee_details (
                    run_id, employee_id, data_origin, details_json, raw_json,
                    created_by_agent, model_name,
                    v_employee_id, v_name, v_company, v_title, v_position, v_about,
                    v_city, v_country_code, v_location, v_followers, v_connections,
                    v_experience, v_languages, v_education, v_activity, v_courses, v_posts,
                    rp_first_name, rp_last_name, rp_url, rp_avatar, rp_banner_image,
                    rp_linkedin_profile_id, rp_current_company_company_id
                ) VALUES (
                    %s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s
                );
            """, (
                str(run_id), str(employee_id), origin,
                json.dumps(cleaned), json.dumps(raw_json),
                "SHADE", self.azure_deployment_id or "gpt-4o",
                vitals.get("employee_id", ""), vitals.get("name", ""),
                vitals.get("company", ""), vitals.get("title", ""),
                vitals.get("position", ""), vitals.get("about", ""),
                vitals.get("city", ""), vitals.get("country_code", ""),
                vitals.get("location", ""), str(vitals.get("followers", 0)),
                str(vitals.get("connections", 0)),
                to_json_text(vitals.get("experience", [])),
                to_json_text(vitals.get("languages", [])),
                to_json_text(vitals.get("education", [])),
                to_json_text(vitals.get("activity", [])),
                to_json_text(vitals.get("courses", [])),
                to_json_text(vitals.get("posts", [])),
                raw_profile.get("first_name", ""), raw_profile.get("last_name", ""),
                raw_profile.get("url", ""), raw_profile.get("avatar", ""),
                raw_profile.get("banner_image", ""),
                raw_profile.get("linkedin_id", ""),
                raw_profile.get("current_company_company_id", ""),
            ))
        except Exception:
            logger.warning("_insert_employee_details failed", exc_info=True)

    # ---- Analytics & summary ----

    def _build_analytics(self, employees_block: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(employees_block)
        seeds = sum(1 for e in employees_block if e["basic_info"].get("is_seed"))
        with_details = sum(1 for e in employees_block if e.get("detailed_profile"))
        rate = (with_details / total * 100.0) if total else 0.0
        pos_counts: Dict[str, int] = {}
        for emp in employees_block:
            pos = (emp["summary"].get("current_position") or "").strip()
            if pos:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        top = [{"position": p, "count": c} for p, c in sorted(pos_counts.items(), key=lambda kv: kv[1], reverse=True)][:10]
        return {"employee_analytics": {"totals": {"employees_found": total, "seed_employees": seeds, "bright_profiles_matched": with_details, "scraping_success_rate": rate}, "top_positions": top}}

    def _build_company_block(self, cd: Optional[CompanyData]) -> Dict[str, Any]:
        if not cd:
            return {}
        return {
            "name": cd.name, "website": cd.website, "description": cd.description,
            "industry": cd.industry, "headquarters": cd.headquarters,
            "employee_estimate": cd.employee_estimate, "revenue_estimate": cd.revenue_estimate,
            "founded_year": cd.founded_year, "tech_stack": cd.tech_stack,
            "social_links": cd.social_links, "timestamp": cd.timestamp,
        }

    def _build_executive_summary(self, company_name, employees_block, analytics) -> str:
        totals = (analytics.get("employee_analytics") or {}).get("totals") or {}
        top = (analytics.get("employee_analytics") or {}).get("top_positions") or []
        return (
            f"Company Overview — {company_name} "
            f"Employees: {totals.get('employees_found', 0)} found; "
            f"detailed profiles success: {float(totals.get('scraping_success_rate', 0)):.2f}%. "
            f"Most common position: {top[0]['position'] if top else 'Not available'}."
        )

    # ---- MAIN ENTRY: save_comprehensive_data ----

    def save_comprehensive_data(
        self,
        mode: str,
        company_name: str,
        raw_employees: List[RawEmployee],
        detailed_profiles: Optional[List[Dict[str, Any]]] = None,
        company_data: Optional[CompanyData] = None,
        manual_title_override: Optional[str] = None,
    ) -> uuid.UUID:

        # Build Bright profile index
        detail_map: Dict[str, Dict[str, Any]] = {}
        profile_list: List[Dict[str, Any]] = []
        if detailed_profiles:
            for p in detailed_profiles:
                if not p:
                    continue
                profile_list.append(p)
                url = p.get("url") or p.get("profile_url") or p.get("linkedin_url") or p.get("link") or ""
                if url and "linkedin.com" in url:
                    detail_map[self._normalize_linkedin_url(url)] = p

        employees_block: List[Dict[str, Any]] = []

        for e in raw_employees:
            norm_url = self._normalize_linkedin_url(e.linkedin_url)
            detail = detail_map.get(norm_url)
            if not detail and profile_list:
                detail = self._fuzzy_match_profile(e, profile_list, norm_url)

            summary = self._summarize_profile(detail, e.name) if detail else {
                "full_name": e.name, "current_position": "Not available",
                "location": "Not available", "experience_years": 0,
                "skills_count": 0, "education_count": 0, "connections": "Not available",
            }

            # B2C title-override check: if extracted title == company name, use manual title
            if manual_title_override and mode == "b2c" and detail:
                cleaned_check = clean_bright_profile(detail or {})
                vitals_check = cleaned_check.get("vitals", {}) or {}
                ext_title = (vitals_check.get("title") or vitals_check.get("position") or summary.get("current_position") or "").strip()
                ext_company = (vitals_check.get("company") or e.company or "").strip()
                if ext_title and ext_company and ext_title.lower() == ext_company.lower():
                    logger.info(f"🔧 Title override: '{ext_title}' matches company. Using: '{manual_title_override}'")
                    summary["current_position"] = manual_title_override
                    detail["_manual_title_override"] = manual_title_override

            basic_info = {
                "name": e.name, "linkedin_url": e.linkedin_url,
                "company": e.company, "search_snippet": e.snippet, "is_seed": e.is_seed,
            }

            # GPT enrichment
            gpt_enriched = None
            if detail:
                try:
                    gpt_enriched = self._azure_enrich_employee(basic_info, detail)
                except Exception as ex:
                    logger.warning(f"GPT enrich failed for {e.name}: {ex}")

            if gpt_enriched:
                for k in ("full_name", "current_position", "location"):
                    if gpt_enriched.get(k):
                        summary[k] = gpt_enriched[k]
                for k in ("experience_years", "skills_count", "education_count"):
                    if k in gpt_enriched:
                        summary[k] = _safe_int(gpt_enriched.get(k), summary.get(k, 0))
                if gpt_enriched.get("connections"):
                    summary["connections"] = gpt_enriched["connections"]
                essay = gpt_enriched.get("detailed_professional_essay")
                if essay:
                    summary["detailed_professional_essay"] = essay
                    basic_info["gpt_detailed_essay"] = essay

            employees_block.append({"basic_info": basic_info, "detailed_profile": detail, "summary": summary})

        analytics = self._build_analytics(employees_block)
        exec_summary = self._build_executive_summary(company_name, employees_block, analytics)
        company_block = self._build_company_block(company_data) if company_data else {}

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                company_id = self._upsert_company(cur, {**company_block, "name": company_name})
                run_id = self._insert_run(cur, mode, company_name, analytics, exec_summary, company_id, len(employees_block))
                self._link_run_company(cur, run_id, company_id, mode)
                for emp in employees_block:
                    bi, det, sm = emp["basic_info"], emp.get("detailed_profile"), emp["summary"]
                    eid = self._upsert_employee(cur, company_id, bi, sm)
                    self._link_run_employee(cur, run_id, eid, bool(bi.get("is_seed")), company_id, bi, sm)
                    self._insert_employee_details(cur, run_id, eid, det, sm, bi)
            conn.commit()
            logger.info(f"✅ Saved SHADE output to DB (run_id={run_id})")
            return run_id
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ DB save failed: {e}", exc_info=True)
            raise
        finally:
            conn.close()


# =============================================================================
# Intel Gathering (B2C / B2B routing)
# =============================================================================


async def _b2c_intel_gathering(name: str, linkedin_url: str, manual_title: str, company_context: str = "") -> Tuple[List[RawEmployee], Optional[CompanyData]]:
    logger.info(f"🎯 B2C mode: {name} ({manual_title})")
    employee = RawEmployee(
        name=name.strip(), linkedin_url=linkedin_url.strip(),
        snippet=manual_title, company=company_context.strip() or "Individual Professional",
        is_seed=True,
    )
    company_data = CompanyData(
        name=company_context or "Individual Professional",
        description=f"B2C Target Profile: {name}, {manual_title}",
        timestamp=_now_iso(),
    )
    return [employee], company_data


async def _gather_intel(
    company_name: str, max_employees: int,
    seed_names: Optional[List[str]] = None, seed_urls: Optional[List[str]] = None,
    mode: str = "b2b",
    b2c_manual_title: str = "", b2c_target_name: str = "", b2c_target_url: str = "",
) -> Tuple[List[RawEmployee], Optional[CompanyData]]:

    if mode == "b2c":
        if not b2c_target_name or not b2c_target_url:
            raise ValueError("B2C mode requires target_name and target_url")
        return await _b2c_intel_gathering(b2c_target_name, b2c_target_url, b2c_manual_title or "Professional", company_name)

    # ---- B2B flow ----
    logger.info(f"🚀 B2B: {company_name} (cap={max_employees})")
    finder = GoogleCSEEmployeeFinder()
    researcher = CompanyReportGenerator()

    seed_names = [s.strip() for s in (seed_names or []) if s and s.strip()]
    seed_urls  = [u.strip() for u in (seed_urls or []) if u and u.strip()]

    def _norm(u: str) -> str:
        return u.split("?", 1)[0].split("#", 1)[0].rstrip("/")

    # URL seeds
    url_seeds: List[RawEmployee] = []
    seen_urls: set = set()
    for u in seed_urls:
        if "linkedin.com/in/" not in u:
            continue
        url = _norm(u)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        url_seeds.append(RawEmployee(
            name=_guess_name_from_linkedin_url(url), linkedin_url=url,
            snippet="Provided by user seed (URL).", company=company_name, is_seed=True,
        ))

    # Enrich URL seeds via Google
    for idx, e in enumerate(url_seeds, 1):
        try:
            g = await asyncio.to_thread(finder.find_employee_by_name_improved, company_name, e.name)
            if g:
                if g.name and g.name != e.name:
                    e.name = g.name
                if g.snippet:
                    e.snippet = g.snippet
        except Exception:
            pass

    if len(url_seeds) >= max_employees:
        cd = await researcher.generate_company_report(company_name)
        return url_seeds[:max_employees], cd

    # Name seeds
    remaining = max(0, max_employees - len(url_seeds))
    name_hits: List[RawEmployee] = []
    if remaining > 0 and seed_names:
        hits = await asyncio.gather(
            *[asyncio.to_thread(finder.find_employee_by_name_improved, company_name, n) for n in seed_names],
            return_exceptions=True,
        )
        for h in hits:
            if isinstance(h, RawEmployee) and h.linkedin_url and "linkedin.com/in/" in h.linkedin_url and h.linkedin_url not in seen_urls:
                h.is_seed = True
                name_hits.append(h)
                seen_urls.add(h.linkedin_url)
        name_hits = name_hits[:remaining]

    # Bulk discovery
    remaining = max(0, max_employees - len(url_seeds) - len(name_hits))
    discovered: List[RawEmployee] = []
    company_task = asyncio.create_task(researcher.generate_company_report(company_name))
    if remaining > 0:
        disc = await asyncio.to_thread(finder.find_employees, company_name, remaining)
        for e in disc or []:
            if e.linkedin_url and e.linkedin_url not in seen_urls:
                discovered.append(e)
                seen_urls.add(e.linkedin_url)
            if len(discovered) >= remaining:
                break

    all_emps = (url_seeds + name_hits + discovered)[:max_employees]
    company_data = await company_task
    logger.info(f"✅ Prepared {len(all_emps)} employees (target={max_employees})")
    return all_emps, company_data


# =============================================================================
# Main async pipeline
# =============================================================================


async def _run_async(context: Dict[str, Any]) -> Dict[str, Any]:
    global _run_costs
    _run_costs = RunCostTracker()

    mode = context.get("model_type", "b2b").lower()
    if mode not in ("b2c", "b2b"):
        mode = "b2b"

    company_name = context.get("company_name") or context.get("spectre_company") or context.get("target_company") or "Unknown Company"

    # FIX: read manual_title from context (was previously an undefined variable)
    manual_title = context.get("manual_target_title", "") or context.get("manual_company_title", "") or ""

    if mode == "b2c":
        employees, company_data = await _gather_intel(
            company_name=company_name, max_employees=1, mode="b2c",
            b2c_manual_title=manual_title,
            b2c_target_name=context.get("b2c_target_name", ""),
            b2c_target_url=context.get("b2c_target_url", ""),
        )
    else:
        max_employees = _safe_int(context.get("spectre_n") or context.get("limit") or context.get("max_employees") or 50, 50)
        employees, company_data = await _gather_intel(
            company_name=company_name, max_employees=max_employees,
            seed_names=context.get("seed_employee_names") or [],
            seed_urls=context.get("seed_employee_urls") or [],
            mode="b2b",
        )

    # Bright Data scraping
    detailed_profiles: List[Dict[str, Any]] = []
    if context.get("use_bright_data", True):
        logger.info(f"🔍 Fetching Bright Data profiles for {len(employees)} URLs...")
        detailed_profiles = await asyncio.to_thread(
            BrightDataScraper().scrape_profiles_one_shot,
            [e.linkedin_url for e in employees],
        )
        logger.info(f"✅ Bright Data returned {len(detailed_profiles)} profiles")

    # B2C: extract real company/title/name from Bright profile
    if mode == "b2c" and detailed_profiles and employees:
        profile = detailed_profiles[0]
        real_company = (profile.get("current_company_name") or (profile.get("current_company") or {}).get("name") or "").strip()
        real_title = (profile.get("position") or "").strip()
        real_name = (profile.get("name") or "").strip()
        logger.info(f"🔄 B2C Bright Data -> Company: {real_company}, Title: {real_title}, Name: {real_name}")
        if real_company:
            employees[0].company = real_company
            company_name = real_company
        if real_title:
            employees[0].snippet = real_title
        if real_name:
            employees[0].name = real_name
        if real_company:
            logger.info(f"🏢 B2C: Running company research for {real_company}")
            company_data = await CompanyReportGenerator().generate_company_report(real_company)
        else:
            company_data = CompanyData(name=company_name, description=f"{real_name}, {real_title}", timestamp=_now_iso())

    # Save to DB
    writer = ComprehensiveDataManager()
    run_costs_snapshot = _run_costs.summary() if _run_costs else {}
    run_id = writer.save_comprehensive_data(
        mode=mode,
        company_name=company_name,
        raw_employees=employees,
        detailed_profiles=detailed_profiles,
        company_data=company_data,
        manual_title_override=manual_title,  # ← FIX: was `manual_title` (undefined)
    )

    context.update({
        "intelligence_run_id": str(run_id),
        "spectre_employees": [
            {"name": e.name, "linkedin_url": e.linkedin_url, "company": e.company, "snippet": e.snippet, "is_seed": e.is_seed}
            for e in employees
        ],
        "company_data_available": bool(company_data),
        "shade_status": "ok",
        "mode": mode,
        "costs": _run_costs.summary() if _run_costs else {},
    })
    return context


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run_async(context))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"SHADE.run error: {e}", exc_info=True)
        return {**(context or {}), "shade_status": f"error: {e}"}


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="SHADE API", version="1.0.0")


class B2CRequest(BaseModel):
    linkedin_url: str
    manual_title: str = ""
    company_context: str = ""


class B2BRequest(BaseModel):
    company_name: str
    max_employees: int = 50
    seed_employee_names: List[str] = []
    seed_employee_urls: List[str] = []


class ShadeResponse(BaseModel):
    status: str
    mode: str
    run_id: Optional[str] = None
    employees: List[Dict[str, Any]] = []
    costs: Dict[str, Any] = {}
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run/b2c", response_model=ShadeResponse)
def run_b2c(req: B2CRequest):
    if not req.linkedin_url or "linkedin.com" not in req.linkedin_url:
        raise HTTPException(status_code=400, detail="A valid LinkedIn profile URL is required.")

    url = req.linkedin_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    guessed_name = _guess_name_from_linkedin_url(url) or "Unknown Professional"

    ctx = {
        "model_type": "b2c",
        "company_name": req.company_context or "Individual Professional",
        "b2c_target_name": guessed_name,
        "b2c_target_url": url,
        "manual_target_title": req.manual_title,
        "manual_company_title": req.manual_title,
        "use_bright_data": True,
    }

    result = run(ctx)
    status = result.get("shade_status", "unknown")
    return ShadeResponse(
        status=status,
        mode="b2c",
        run_id=result.get("intelligence_run_id"),
        employees=result.get("spectre_employees", []),
        costs=result.get("costs", {}),
        error=status if status != "ok" else None,
    )


@app.post("/run/b2b", response_model=ShadeResponse)
def run_b2b(req: B2BRequest):
    if not req.company_name.strip():
        raise HTTPException(status_code=400, detail="company_name is required.")

    ctx = {
        "model_type": "b2b",
        "company_name": req.company_name,
        "spectre_n": req.max_employees,
        "seed_employee_names": req.seed_employee_names,
        "seed_employee_urls": req.seed_employee_urls,
        "use_bright_data": True,
    }

    result = run(ctx)
    status = result.get("shade_status", "unknown")
    return ShadeResponse(
        status=status,
        mode="b2b",
        run_id=result.get("intelligence_run_id"),
        employees=result.get("spectre_employees", []),
        costs=result.get("costs", {}),
        error=status if status != "ok" else None,
    )


# =============================================================================
# CLI (kept for local testing)
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("🏹 SHADE – B2C LinkedIn Intelligence")
    print("=" * 60)

    raw_url = input("LinkedIn PROFILE URL: ").strip()
    if not raw_url or "linkedin.com" not in raw_url:
        print("❌ Valid LinkedIn URL required.")
        return

    url = raw_url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    guessed_name = _guess_name_from_linkedin_url(url) or "Unknown Professional"
    manual_title = input("Manual Job Title (optional): ").strip()

    ctx = {
        "model_type": "b2c",
        "company_name": "Individual Professional",
        "b2c_target_name": guessed_name,
        "b2c_target_url": url,
        "manual_target_title": manual_title,
        "manual_company_title": manual_title,
        "use_bright_data": True,
    }

    print(f"\n🚀 Running SHADE for: {guessed_name} ({url})")
    result = run(ctx)

    print(f"\n✅ Done! Status: {result.get('shade_status')}")
    print(f"   Run ID: {result.get('intelligence_run_id')}")
    emps = result.get("spectre_employees", [])
    if emps:
        e = emps[0]
        print(f"   Name: {e.get('name')}, Company: {e.get('company')}, Snippet: {e.get('snippet')}")

    costs = result.get("costs", {})
    if costs:
        print(f"\n💰 Run Costs:")
        print(f"   Total:       ${costs.get('total_cost_usd', 0):.4f}")
        gpt = costs.get("gpt", {})
        print(f"   GPT-4o:      ${gpt.get('total_cost_usd', 0):.4f}  ({gpt.get('total_tokens', 0)} tokens, {gpt.get('calls', 0)} calls)")
        bd = costs.get("bright_data", {})
        print(f"   Bright Data: ${bd.get('cost_usd', 0):.4f}  ({bd.get('profiles_scraped', 0)} profiles)")
        gc = costs.get("google_cse", {})
        print(f"   Google CSE:  ${gc.get('cost_usd', 0):.4f}  ({gc.get('queries', 0)} queries)")


if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        main()