#!/usr/bin/env python3
"""
MIRAGE HYBRID: DB-First Architecture with OG Discovery Algorithm
================================================================
Phase 0: Employee-ID driven input
Phase 1: GPT-based competitor detection
Phase 2: Single employee target profile from DB
Phase 3+4: Parallel Google discovery + GPT matching (4-WEIGHT, early-stop)
Phase 5: Bright Data enrichment + DB persistence
Phase 6: Final DB writes

Version: 7.0 - FastAPI + Cleaned Production Build
"""

from __future__ import annotations

import os
import json
import time
import logging
import uuid
import re
import sys
import hashlib
import asyncio
import aiohttp
import requests
import threading
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2.extras import RealDictCursor, Json
import tiktoken

load_dotenv()

# =============================================================================
# Logging
# =============================================================================

def _force_utf8_console():
    try:
        if sys.platform == "win32":
            os.system("chcp 65001 > nul")
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_force_utf8_console()


def setup_logging():
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(ch)
    return logging.getLogger("MIRAGE_HYBRID")


logger = setup_logging()
DEBUG_MIRAGE = os.getenv("MIRAGE_DEBUG", "1") == "1"

# =============================================================================
# MIRAGE Configuration — Change these constants to tune behavior
# =============================================================================
NUM_COMPETITORS = 7           # Number of competitor companies to detect
BUSINESS_MODEL = "b2c"        # "b2c" or "b2b"
TOP_K_MATCHES = 7             # Total top matches to return
MATCHES_PER_COMPANY = 1       # Max matches to take from each company


def dbg(title: str, obj: Any = None):
    if not DEBUG_MIRAGE:
        return
    print(f"\n{'=' * 90}\n  DEBUG: {title}\n{'=' * 90}")
    if obj is None:
        return
    try:
        if hasattr(obj, "__dict__"):
            obj = obj.__dict__
        print(json.dumps(obj, indent=2, default=str)[:8000])
    except Exception:
        print(str(obj)[:8000])


# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    db_url = os.getenv("SPECTRE_DB_URL")
    if not db_url:
        raise RuntimeError("SPECTRE_DB_URL is not set")
    conn = connect(db_url, cursor_factory=RealDictCursor)
    conn.autocommit = True
    return conn


# =============================================================================
# Cost / Token Tracking
# =============================================================================

class CostTracker:
    def __init__(self):
        self.gpt_input_tokens = 0
        self.gpt_output_tokens = 0
        self.gpt_calls = 0
        self.google_queries = 0
        self.google_results = 0
        self.bright_data_rows = 0

    def track_gpt_tokens(self, inp: int, out: int):
        self.gpt_input_tokens += inp
        self.gpt_output_tokens += out
        self.gpt_calls += 1

    def track_google_query(self, n: int = 0):
        self.google_queries += 1
        self.google_results += n

    def track_bright_data_rows(self, n: int):
        self.bright_data_rows += n

    def get_summary(self) -> Dict:
        ic = (self.gpt_input_tokens / 1_000_000) * 2.50
        oc = (self.gpt_output_tokens / 1_000_000) * 10.00
        gc = (self.google_queries / 1000) * 5.0
        return {
            "gpt": {
                "total_calls": self.gpt_calls,
                "input_tokens": self.gpt_input_tokens,
                "output_tokens": self.gpt_output_tokens,
                "total_tokens": self.gpt_input_tokens + self.gpt_output_tokens,
                "cost_usd": round(ic + oc, 4),
            },
            "google": {
                "total_queries": self.google_queries,
                "total_results": self.google_results,
                "cost_usd": round(gc, 4),
            },
            "bright_data": {"total_rows": self.bright_data_rows},
            "total_cost_usd": round(ic + oc + gc, 4),
        }


_global_cost = CostTracker()


class TokenUsageTracker:
    def __init__(self):
        self.gpt_input_tokens = 0
        self.gpt_output_tokens = 0
        self.gpt_calls = 0
        self.google_queries = 0
        self.google_results = 0
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 4

    def track_gpt_call(self, system_prompt: str, user_prompt: str, response: str):
        inp = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)
        out = self.count_tokens(response) if response else 0
        self.gpt_input_tokens += inp
        self.gpt_output_tokens += out
        self.gpt_calls += 1
        _global_cost.track_gpt_tokens(inp, out)

    def track_google_query(self, n: int = 0):
        self.google_queries += 1
        self.google_results += n
        _global_cost.track_google_query(n)

    def get_summary(self) -> Dict[str, Any]:
        ic = (self.gpt_input_tokens / 1_000_000) * 2.50
        oc = (self.gpt_output_tokens / 1_000_000) * 10.00
        gc = (self.google_queries / 1000) * 5.0
        return {
            "gpt": {
                "total_calls": self.gpt_calls,
                "input_tokens": self.gpt_input_tokens,
                "output_tokens": self.gpt_output_tokens,
                "total_tokens": self.gpt_input_tokens + self.gpt_output_tokens,
                "cost_usd": round(ic + oc, 4),
            },
            "google": {
                "total_queries": self.google_queries,
                "total_results": self.google_results,
                "cost_usd": round(gc, 4),
            },
            "total_cost_usd": round(ic + oc + gc, 4),
        }

    def print_summary(self):
        s = self.get_summary()
        print(f"\n{'=' * 60}\nTOKEN USAGE & COST SUMMARY\n{'=' * 60}")
        print(f"  GPT Calls: {s['gpt']['total_calls']}  |  "
              f"Tokens: {s['gpt']['total_tokens']:,}  |  "
              f"Cost: ${s['gpt']['cost_usd']:.2f}")
        print(f"  Google Queries: {s['google']['total_queries']}  |  "
              f"Results: {s['google']['total_results']}  |  "
              f"Cost: ${s['google']['cost_usd']:.2f}")
        print(f"  TOTAL COST: ${s['total_cost_usd']:.2f}\n{'=' * 60}\n")


_token_tracker = TokenUsageTracker()


# =============================================================================
# GPT Clients
# =============================================================================

class AzureGPTClient:
    """Async Azure OpenAI client with caching + token tracking."""

    def __init__(self, tracker: Optional[TokenUsageTracker] = None):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if not all([self.api_key, self.endpoint]):
            raise ValueError("Missing Azure OpenAI config (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")
        self.headers = {"Content-Type": "application/json", "api-key": self.api_key}
        self.cache: Dict[str, str] = {}
        self.tracker = tracker or _token_tracker

    async def chat_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 1500
    ) -> str:
        cache_key = hashlib.md5(f"{system_prompt}|{user_prompt}|{temperature}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Azure OpenAI error {resp.status}: {await resp.text()}")
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        self.tracker.track_gpt_call(system_prompt, user_prompt, content)
        self.cache[cache_key] = content
        return content


class GPTInterface:
    """Sync wrapper around AzureGPTClient (runs async on a background loop)."""

    def __init__(self, tracker: Optional[TokenUsageTracker] = None):
        self.azure = AzureGPTClient(tracker=tracker)
        self._bg_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def call_gpt(
        self, system_prompt: str, user_prompt: str,
        temperature: float = 0.2, max_tokens: int = 1500, **_kwargs,
    ) -> str:
        coro = self.azure.chat_completion(system_prompt, user_prompt, temperature, max_tokens)
        fut = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return fut.result()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CompetitorCompany:
    name: str
    industry: str
    similarity_score: float
    detection_method: str = "GPT Analysis"
    company_id: Optional[str] = None
    description: str = ""
    raw_detection_data: Dict = field(default_factory=dict)


@dataclass
class TargetEmployeeProfile:
    employee_id: str
    name: str
    title: str
    company: str
    department: str
    experience_years: float
    key_skills: List[str]
    company_id: str
    essay: str = ""
    linkedin_url: Optional[str] = None
    canonical_linkedin_id: Optional[str] = None
    seniority_level: str = ""
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    about_text: str = ""
    profile_summary: str = ""
    raw_profile_data: Dict = field(default_factory=dict)


@dataclass
class CompetitorEmployee:
    name: str
    title: str
    company: str
    linkedin_url: str
    search_snippet: str
    employee_id: Optional[str] = None
    company_id: Optional[str] = None
    canonical_linkedin_id: Optional[str] = None
    raw_data: Dict = field(default_factory=dict)


@dataclass
class EmployeeMatch:
    similarity_score: float
    target_employee_id: str
    matched_employee_id: str
    target_employee: str = ""
    competitor_employee: str = ""
    competitor_company: str = ""
    match_rationale: str = ""
    linkedin_url: str = ""
    competitor_role: str = ""
    matching_factors: Dict = field(default_factory=dict)
    confidence: str = "medium"
    notes: str = ""


# =============================================================================
# Utility Functions
# =============================================================================

def canonicalize_linkedin_url(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url.lstrip("/")
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("in")
        if idx + 1 < len(parts):
            return parts[idx + 1].strip("/")
    except ValueError:
        pass
    return parts[-1].strip("/") if parts else None


def normalize_linkedin_url(url: str) -> str:
    slug = canonicalize_linkedin_url(url)
    return f"https://www.linkedin.com/in/{slug}/" if slug else url


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_json_parse(content: str) -> Optional[Any]:
    if not content:
        return None
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*}", "}", content)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


# =============================================================================
# Title Extractor (GPT-based)
# =============================================================================

class TitleExtractor:
    def __init__(self, gpt_client: GPTInterface):
        self.gpt = gpt_client

    def extract_title_and_company(
        self, headline: str, experience: List[Dict[str, Any]],
        name: str, bright_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        if bright_profile:
            position = bright_profile.get("position", "")
            comp = bright_profile.get("current_company_name") or (
                bright_profile.get("current_company") or {}
            ).get("name", "")
            if position and comp:
                return {"title": position, "company": comp}

        if not experience:
            return {"title": headline or "", "company": ""}

        try:
            system_prompt = (
                "You are a LinkedIn data parser. Extract the CURRENT job title and company.\n"
                'Return ONLY JSON: {"title": "...", "company": "..."}'
            )
            user_prompt = (
                f"Extract the CURRENT job title and company for {name}.\n"
                f"LinkedIn Headline: {headline}\n"
                f"Experience (most recent first):\n"
                f"{json.dumps(experience[:5], indent=2, ensure_ascii=False)[:3000]}\n"
                "Return ONLY JSON."
            )
            resp = self.gpt.call_gpt(system_prompt, user_prompt, temperature=0.1, max_tokens=150)
            result = safe_json_parse(resp)
            if result:
                return {
                    "title": (result.get("title") or "").strip(),
                    "company": (result.get("company") or "").strip(),
                }
        except Exception as e:
            logger.warning(f"GPT title extraction failed: {e}")

        return {
            "title": experience[0].get("title", "") or "",
            "company": experience[0].get("company", "") or "",
        }


# =============================================================================
# Phase 0: Target Resolver (DB)
# =============================================================================

class TargetResolver:
    def __init__(self, conn):
        self.conn = conn

    def resolve_from_run_id(self, run_id: str) -> Tuple[str, Dict[str, Any]]:
        """Look up a run_id → get employee_id → resolve full target context.
        Checks run_employees first, then falls back to runs.raw_json.
        Returns (employee_id, target_context)."""
        logger.info(f"[Phase 0] Resolving from run_id: {run_id}")
        employee_id = None

        with self.conn.cursor() as cur:
            # 1) Check run_employees table first (this is where it lives)
            cur.execute(
                "SELECT employee_id, role_in_run FROM spectre.run_employees "
                "WHERE run_id = %s LIMIT 1",
                (run_id,),
            )
            row = cur.fetchone()
            if row:
                employee_id = row["employee_id"]
                logger.info(f"[Phase 0] Found employee in run_employees: {employee_id} (role: {row['role_in_run']})")

            # 2) Fallback: check runs.raw_json
            if not employee_id:
                cur.execute("SELECT raw_json FROM spectre.runs WHERE run_id = %s", (run_id,))
                run_row = cur.fetchone()
                if run_row:
                    raw = run_row.get("raw_json") or {}
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    employee_id = raw.get("target_employee_id")
                    if employee_id:
                        logger.info(f"[Phase 0] Found target employee in runs.raw_json: {employee_id}")

        if not employee_id:
            raise ValueError(f"Run {run_id} has no target employee_id (checked run_employees and runs)")

        logger.info(f"[Phase 0] Run {run_id} → employee_id: {employee_id}")
        target_context = self.resolve_target_from_employee_id(employee_id)
        return employee_id, target_context

    def find_existing_run(self, employee_id: str) -> Optional[str]:
        """Find the most recent run_id for an employee from the DB."""
        with self.conn.cursor() as cur:
            # Check run_employees table first
            cur.execute(
                "SELECT run_id FROM spectre.run_employees "
                "WHERE employee_id = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (employee_id,),
            )
            row = cur.fetchone()
            if row:
                logger.info(f"[Phase 0] Found existing run for {employee_id}: {row['run_id']}")
                return row["run_id"]

            # Fallback: check runs.raw_json
            cur.execute(
                "SELECT run_id FROM spectre.runs "
                "WHERE raw_json->>'target_employee_id' = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (employee_id,),
            )
            row = cur.fetchone()
            if row:
                logger.info(f"[Phase 0] Found existing run (from raw_json) for {employee_id}: {row['run_id']}")
                return row["run_id"]

        return None

    def resolve_target_from_employee_id(self, employee_id: str) -> Dict[str, Any]:
        logger.info(f"[Phase 0] Resolving target: {employee_id}")
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM spectre.employees WHERE employee_id = %s", (employee_id,))
            emp = cur.fetchone()
            if not emp:
                raise ValueError(f"Employee not found: {employee_id}")
            emp = dict(emp)

            company_id = emp.get("current_company_id")
            if not company_id:
                raise ValueError(f"Employee {employee_id} has no current_company_id")

            cur.execute("SELECT * FROM spectre.companies WHERE company_id = %s", (company_id,))
            comp = cur.fetchone()
            if not comp:
                raise ValueError(f"Company not found: {company_id}")
            comp = dict(comp)

            cur.execute(
                "SELECT * FROM spectre.employee_details WHERE employee_id = %s ORDER BY created_at DESC LIMIT 1",
                (employee_id,),
            )
            details_row = cur.fetchone()
            details = dict(details_row) if details_row else {}

        result = {
            "employee_id": employee_id,
            "employee_name": emp.get("full_name") or "",
            "current_title": emp.get("current_title") or "",
            "current_company_id": company_id,
            "company_name": comp.get("name") or "",
            "linkedin_url": emp.get("linkedin_url") or "",
            "canonical_linkedin_id": emp.get("canonical_linkedin_id") or "",
            "employee_data": emp,
            "company_data": comp,
            "employee_details": details,
        }
        logger.info(f"[Phase 0] Resolved: {result['employee_name']} at {result['company_name']}")
        return result


# =============================================================================
# Phase 1: Competitor Detection (GPT)
# =============================================================================

class CompetitorDetector:
    def __init__(self, gpt: GPTInterface, conn):
        self.gpt = gpt
        self.conn = conn

    def detect_competitors(
        self, company_name: str, industry: str = "", description: str = "",
        num_competitors: int = 10, business_model: str = "b2b", region: str = "",
    ) -> List[CompetitorCompany]:
        logger.info(f"[Phase 1] Detecting {num_competitors} competitors for: {company_name} (region: {region or 'auto'})")

        region_instruction = ""
        if region:
            region_instruction = (
                f"\n\nCRITICAL REGIONAL REQUIREMENT:\n"
                f"The target company operates in the {region} region.\n"
                f"You MUST return competitors that operate in the SAME region ({region}).\n"
                f"Prioritize local/regional competitors over global ones.\n"
                f"All competitors should have a significant presence in {region}."
            )

        system_prompt = f"""You are a competitive intelligence analyst.
Identify the top {num_competitors} direct competitors for a given company.

Return ONLY a JSON array:
[
  {{"name": "Company Name", "industry": "Industry", "similarity_score": 85, "description": "Brief"}}
]

Focus on direct competitors in the same market segment. Most similar first.{region_instruction}"""

        user_prompt = (
            f"Company: {company_name}\n"
            f"Industry: {industry or 'Not specified'}\n"
            f"Description: {description or 'Not specified'}\n"
            f"Business Model: {business_model.upper()}\n"
            f"Region: {region or 'Not specified'}\n\n"
            f"Return ONLY JSON array of {num_competitors} competitors."
        )

        try:
            response = self.gpt.call_gpt(system_prompt, user_prompt, temperature=0.3, max_tokens=2000)
            data = safe_json_parse(response)
            if not data:
                return []
            if isinstance(data, dict) and "competitors" in data:
                data = data["competitors"]
            if not isinstance(data, list):
                return []

            competitors = []
            for item in data[:num_competitors]:
                if not isinstance(item, dict):
                    continue
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                competitors.append(
                    CompetitorCompany(
                        name=name,
                        industry=(item.get("industry") or "").strip(),
                        similarity_score=float(item.get("similarity_score") or 0),
                        description=(item.get("description") or "").strip(),
                        raw_detection_data=item,
                    )
                )
            logger.info(f"[Phase 1] Detected {len(competitors)} competitors")
            return competitors

        except Exception as e:
            logger.error(f"Competitor detection failed: {e}")
            return []

    def get_or_create_company(self, name: str, industry: str = "", description: str = "") -> str:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT company_id FROM spectre.companies WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return row["company_id"]

            cid = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO spectre.companies (company_id, name, industry, description) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (company_id) DO NOTHING",
                (cid, name, industry, description),
            )
            logger.info(f"Created company: {name} ({cid})")
            return cid


# =============================================================================
# Phase 2: Target Profile Builder (DB → GPT)
# =============================================================================

class TargetProfileBuilder:
    def __init__(self, gpt: GPTInterface):
        self.gpt = gpt

    def build_profile_from_db(self, ctx: Dict[str, Any]) -> TargetEmployeeProfile:
        logger.info(f"[Phase 2] Building target profile for: {ctx['employee_id']}")

        emp = ctx.get("employee_data", {})
        details = ctx.get("employee_details", {})
        about_text = emp.get("v_about") or emp.get("profile_cache_text") or details.get("about") or ""

        system_prompt = (
            "You are an expert at analyzing professional profiles.\n"
            "Extract skills, department, seniority, experience years.\n"
            'Return ONLY JSON: {"skills":[], "department":"", "seniority_level":"", "years_experience": N}'
        )
        user_prompt = (
            f"Name: {ctx['employee_name']}\nTitle: {ctx['current_title']}\n"
            f"Company: {ctx['company_name']}\nAbout: {about_text[:1000]}\n\n"
            "Extract skills, department, seniority, experience. Return ONLY JSON."
        )

        try:
            response = self.gpt.call_gpt(system_prompt, user_prompt, temperature=0.2, max_tokens=800)
            extracted = safe_json_parse(response) or {}
        except Exception as e:
            logger.error(f"Profile building failed: {e}")
            extracted = {}

        skills = extracted.get("skills") or []
        if not isinstance(skills, list):
            skills = []
        try:
            exp_years = float(extracted.get("years_experience") or 0)
        except (TypeError, ValueError):
            exp_years = 0.0

        return TargetEmployeeProfile(
            employee_id=ctx["employee_id"],
            name=ctx["employee_name"],
            title=ctx["current_title"],
            company=ctx["company_name"],
            company_id=ctx["current_company_id"],
            linkedin_url=ctx.get("linkedin_url"),
            canonical_linkedin_id=ctx.get("canonical_linkedin_id"),
            department=extracted.get("department") or "",
            experience_years=exp_years,
            key_skills=skills,
            seniority_level=extracted.get("seniority_level") or "",
            about_text=about_text,
            raw_profile_data=emp,
        )


# =============================================================================
# Phase 3+4: Parallel Search + Match (OG Broad-Query + 4-Weight Scoring)
# =============================================================================

class CompetitorEmployeeFinder:
    """
    Parallel search + match pipeline.
    B2C: 3 GPT-generated role-anchored query templates per target.
    B2B: 15-20 GPT-generated broad queries per company.
    """

    def __init__(self, gpt_client: AzureGPTClient):
        self.gpt = gpt_client
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.pages_per_query = 2
        self.results_per_page = 10
        self.query_sleep = 3.0
        self._current_run_id: Optional[str] = None

        self.use_mock_data = not (self.google_api_key and self.google_cse_id)
        if self.use_mock_data:
            logger.warning("Google CSE not configured - mock mode")

    # ==================================================================
    # Main entry: Parallel Search + Match across all companies
    # ==================================================================

    async def search_and_match_parallel(
        self,
        target_profile: TargetEmployeeProfile,
        competitors: List[CompetitorCompany],
        matcher: "ProfileMatcher",
        *,
        business_model: str = "b2b",
        top_k_per_company: int = 3,
        score_threshold: float = 40.0,
        max_concurrent_companies: int = 4,
    ) -> Tuple[Dict[str, List[CompetitorEmployee]], Dict[str, List[EmployeeMatch]]]:
        if self.use_mock_data:
            return {c.name: [] for c in competitors}, {}

        mode = (business_model or "b2b").lower().strip()
        logger.info(f"[PHASE 3+4] PARALLEL SEARCH+MATCH ({mode.upper()}, {len(competitors)} companies)")

        queries_by_company = await self._generate_all_queries(target_profile, competitors, mode)

        sem = asyncio.Semaphore(max_concurrent_companies)

        async def _run_one(comp: CompetitorCompany):
            async with sem:
                queries = queries_by_company.get(comp.name, [])
                if not queries:
                    return comp.name, [], []
                emps, matches = await self._search_and_match_one_company(
                    company_name=comp.name, queries=queries,
                    target_profile=target_profile, matcher=matcher,
                    top_k=top_k_per_company, score_threshold=score_threshold,
                )
                return comp.name, emps, matches

        results = await asyncio.gather(*[_run_one(c) for c in competitors], return_exceptions=True)

        employees_map: Dict[str, List[CompetitorEmployee]] = {}
        matches_map: Dict[str, List[EmployeeMatch]] = {}

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Company pipeline failed: {r}")
                continue
            comp_name, emps, matches = r
            employees_map[comp_name] = emps
            if matches:
                matches_map[comp_name] = matches

        total_emps = sum(len(v) for v in employees_map.values())
        total_matches = sum(len(v) for v in matches_map.values())
        logger.info(f"[PHASE 3+4] DONE: {total_emps} employees, {total_matches} matches")
        return employees_map, matches_map

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    async def _generate_all_queries(
        self, target_profile: TargetEmployeeProfile,
        competitors: List[CompetitorCompany], mode: str,
    ) -> Dict[str, List[str]]:
        queries_by_company: Dict[str, List[str]] = {}

        if mode == "b2c":
            templates = await self._generate_b2c_templates(target_profile)
            logger.info(f"[B2C] Generated {len(templates)} query templates")
            for comp in competitors:
                queries = []
                for tpl in templates:
                    q = tpl.replace("<COMPANY>", f'"{comp.name}"').replace('""', '"')
                    if "site:linkedin.com/in" not in q:
                        q = f"site:linkedin.com/in {q}"
                    if "-former" not in q:
                        q += " -former -ex -previous -past"
                    q = re.sub(r"\s+", " ", q).strip()
                    if len(q) >= 25:
                        queries.append(q)
                queries_by_company[comp.name] = list(dict.fromkeys(queries))
        else:
            sem = asyncio.Semaphore(5)
            async def _gen(comp):
                async with sem:
                    qs = await self._generate_broad_queries(comp, [target_profile])
                    return comp.name, qs
            gen_results = await asyncio.gather(*[_gen(c) for c in competitors], return_exceptions=True)
            for r in gen_results:
                if isinstance(r, tuple):
                    queries_by_company[r[0]] = r[1]

        return queries_by_company

    async def _generate_b2c_templates(self, target: TargetEmployeeProfile) -> List[str]:
        system = (
            "You are a LinkedIn sourcing expert. Given a target employee profile, "
            "create exactly 3 Google search queries to find SIMILAR employees at competitor companies.\n\n"
            "Rules:\n"
            "1. Every query MUST start with site:linkedin.com/in\n"
            "2. Use <COMPANY> as placeholder for the competitor company name\n"
            "3. Every query MUST end with -former -ex -previous -past\n"
            "4. Keep queries BROAD enough to find results\n"
            "5. Use the target's department, title keywords, and seniority to guide queries\n\n"
            "Return ONLY a JSON array of exactly 3 query strings."
        )
        user = (
            f"Target Profile:\n"
            f"  Name: {target.name}\n  Title: {(target.title or '').strip()}\n"
            f"  Department: {(target.department or '').strip()}\n"
            f"  Seniority: {(target.seniority_level or '').strip()}\n"
            f"  Skills: {', '.join(target.key_skills[:8]) if target.key_skills else ''}\n"
            f"  About: {(target.about_text or '')[:800]}\n\n"
            "Generate exactly 3 Google search query templates with <COMPANY> placeholder."
        )

        try:
            resp = await self.gpt.chat_completion(system, user, temperature=0.2, max_tokens=400)
            qs = safe_json_parse(resp)
            if isinstance(qs, list) and qs:
                cleaned = []
                for q in qs[:3]:
                    q = (q or "").strip()
                    if not q:
                        continue
                    if "<COMPANY>" not in q.upper():
                        q = f'site:linkedin.com/in <COMPANY> {q}'
                    cleaned.append(q)
                if cleaned:
                    return cleaned[:3]
        except Exception as e:
            logger.warning(f"GPT B2C template gen failed: {e}")

        return self._fallback_b2c_templates(target)

    def _fallback_b2c_templates(self, target: TargetEmployeeProfile) -> List[str]:
        title = (target.title or "").strip()
        dept = (target.department or "").strip()
        neg = "-former -ex -previous -past"
        first_word = title.split()[0] if title.split() else ""
        templates = []
        if title:
            templates.append(f'site:linkedin.com/in <COMPANY> "{title}" {neg}')
        if dept:
            templates.append(f'site:linkedin.com/in <COMPANY> "{dept}" {neg}')
        if first_word and len(first_word) > 2:
            templates.append(f'site:linkedin.com/in <COMPANY> "{first_word}" {neg}')
        if len(templates) < 3:
            templates.append(f'site:linkedin.com/in <COMPANY> {neg}')
        return templates[:3]

    async def _generate_broad_queries(
        self, comp: CompetitorCompany, targets: List[TargetEmployeeProfile],
    ) -> List[str]:
        depts = list(set(t.department for t in targets if t.department))
        roles = list(set(t.title.split()[0] for t in targets if t.title))[:10]
        system = (
            "Generate 15-20 Google queries to find current employees at a company.\n"
            "Rules: site:linkedin.com/in, company in quotes, end with -former -ex -previous -past.\n"
            "Return ONLY a JSON array of strings."
        )
        user = (
            f"Company: {comp.name}\nIndustry: {comp.industry}\n"
            f"Departments: {', '.join(depts)}\nRoles: {', '.join(roles)}\n"
            f"Generate 15-20 queries for {comp.name}."
        )
        try:
            resp = await self.gpt.chat_completion(system, user, temperature=0.3, max_tokens=800)
            qs = json.loads(resp) if resp else []
            if isinstance(qs, list) and qs:
                return [self._ensure_company_quoted(q, comp.name) for q in qs[:25]]
        except Exception as e:
            logger.warning(f"GPT query gen failed for {comp.name}: {e}")
        return self._fallback_queries(comp.name)

    @staticmethod
    def _ensure_company_quoted(query: str, company_name: str) -> str:
        cn = company_name.strip()
        quoted = f'"{cn}"'
        if quoted in query:
            return query
        if cn in query:
            return query.replace(cn, quoted, 1)
        if "site:linkedin.com/in" in query:
            return query.replace("site:linkedin.com/in", f"site:linkedin.com/in {quoted}", 1)
        return f'site:linkedin.com/in {quoted} {query}'

    def _fallback_queries(self, company_name: str) -> List[str]:
        cq = f'"{company_name.strip()}"'
        neg = "-former -ex -previous -past"
        depts = ["Engineering", "Sales", "Marketing", "Product", "Finance",
                 "Data", "Design", "Operations", "Legal", "HR", "Education"]
        levels = ["Director", "VP", "Head", "Lead", "Senior", "Manager"]
        qs = [f'site:linkedin.com/in {cq} "{d}" {neg}' for d in depts]
        qs += [f'site:linkedin.com/in {cq} "{l}" {neg}' for l in levels]
        qs.append(f"site:linkedin.com/in {cq} {neg}")
        return qs

    # ------------------------------------------------------------------
    # Per-company search + match with early stop
    # ------------------------------------------------------------------

    async def _search_and_match_one_company(
        self, company_name: str, queries: List[str],
        target_profile: TargetEmployeeProfile, matcher: "ProfileMatcher",
        top_k: int = 3, score_threshold: float = 40.0,
    ) -> Tuple[List[CompetitorEmployee], List[EmployeeMatch]]:
        logger.info(f"[{company_name}] Starting pipeline ({len(queries)} queries)")

        all_emps: List[CompetitorEmployee] = []
        seen_urls: set = set()
        confirmed_matches: List[EmployeeMatch] = []
        unsorted_candidates: List[CompetitorEmployee] = []
        SCORE_EVERY_N = 2

        async with aiohttp.ClientSession() as session:
            for i, query in enumerate(queries, 1):
                logger.info(f"[{company_name}] Query {i}/{len(queries)}: {query[:100]}")
                items = await self._run_query(session, query)
                _token_tracker.track_google_query(len(items))

                for item in items:
                    emp = self._parse_item(item, company_name)
                    if emp:
                        url_key = (emp.linkedin_url or "").strip().rstrip("/").lower()
                        if url_key and url_key not in seen_urls:
                            seen_urls.add(url_key)
                            all_emps.append(emp)
                            unsorted_candidates.append(emp)

                should_score = (
                    len(unsorted_candidates) >= 8
                    or i % SCORE_EVERY_N == 0
                    or i == len(queries)
                )

                if should_score and unsorted_candidates:
                    new_matches = await self._score_candidates_async(
                        target_profile, unsorted_candidates, company_name,
                        matcher, score_threshold,
                    )
                    confirmed_matches.extend(new_matches)
                    unsorted_candidates = []

                    if len(confirmed_matches) >= top_k:
                        confirmed_matches.sort(key=lambda m: m.similarity_score, reverse=True)
                        confirmed_matches = confirmed_matches[:top_k]
                        logger.info(
                            f"[EARLY STOP] {company_name}: {top_k} matches after "
                            f"{i}/{len(queries)} queries (saved {len(queries) - i})"
                        )
                        break

                await asyncio.sleep(self.query_sleep)

        confirmed_matches.sort(key=lambda m: m.similarity_score, reverse=True)
        confirmed_matches = confirmed_matches[:top_k]
        logger.info(f"[{company_name}] DONE: {len(all_emps)} employees, {len(confirmed_matches)} matches")
        return all_emps, confirmed_matches

    async def _score_candidates_async(
        self, target_profile: TargetEmployeeProfile,
        candidates: List[CompetitorEmployee], company_name: str,
        matcher: "ProfileMatcher", score_threshold: float,
    ) -> List[EmployeeMatch]:
        def _do_score():
            validated = matcher._validate_company_membership(candidates, company_name)
            if not validated:
                return []
            filtered = matcher._prefilter_with_weights(target_profile, validated, company_name)
            if not filtered:
                return []
            batch = filtered[:matcher.BATCH_SIZE]
            matches = matcher._score_batch(target_profile, batch, company_name, score_threshold)
            for m in matches:
                bonus = matcher._compute_4weight_bonus(target_profile, m, company_name)
                m.similarity_score = min(100.0, m.similarity_score + bonus)
                m.notes = f"4-weight bonus: +{bonus:.1f}"
            return matches

        return await asyncio.get_event_loop().run_in_executor(None, _do_score)

    # ------------------------------------------------------------------
    # Google CSE helpers
    # ------------------------------------------------------------------

    async def _run_query(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        url = "https://www.googleapis.com/customsearch/v1"
        all_items: List[Dict] = []

        for page in range(self.pages_per_query):
            start_at = page * self.results_per_page + 1
            params = {
                "key": self.google_api_key, "cx": self.google_cse_id,
                "q": query, "num": self.results_per_page, "start": start_at,
            }
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 429:
                        logger.warning("[GOOGLE CSE] 429 rate limited, sleeping 5s")
                        await asyncio.sleep(5)
                        continue
                    if resp.status != 200:
                        logger.error(f"[GOOGLE CSE] status={resp.status}")
                        break
                    data = await resp.json()
                    items = data.get("items", []) or []
                    all_items.extend(items)
                    if len(items) < self.results_per_page:
                        break
            except Exception as e:
                logger.exception(f"[GOOGLE CSE] exception: {e}")
                break
            await asyncio.sleep(self.query_sleep)

        return all_items

    def _parse_item(self, item: Dict, company_name: str) -> Optional[CompetitorEmployee]:
        link = item.get("link", "") or ""
        if "linkedin.com/in" not in link and "linkedin.com/pub" not in link:
            return None
        if not self._validate_current_employment(item, company_name):
            return None

        raw_title = item.get("title", "") or ""
        snippet = (item.get("snippet", "") or "").strip()
        name = raw_title.split(" - ")[0].split(" |")[0].strip()
        if not name or len(name) < 2:
            return None

        job_title = ""
        if " - " in raw_title:
            parts = raw_title.split(" - ", 1)
            if len(parts) > 1:
                job_title = parts[1].split(" | ")[0].strip()
        if not job_title:
            job_title = self._extract_title_from_snippet(snippet, company_name)

        return CompetitorEmployee(
            name=name, title=job_title, company=company_name,
            linkedin_url=normalize_linkedin_url(link.rstrip("/")),
            canonical_linkedin_id=canonicalize_linkedin_url(link),
            search_snippet=snippet[:200],
            raw_data={"google_item": item},
        )

    @staticmethod
    def _validate_current_employment(item: Dict, company_name: str) -> bool:
        title = (item.get("title", "") or "").lower()
        snippet = (item.get("snippet", "") or "").lower()
        text = f"{title} {snippet}"
        comp = (company_name or "").lower().strip()

        for bad in ["former", "ex-", "previously at", "formerly at",
                     "was at", "used to work", "left", "departed"]:
            if bad in text:
                return False

        if comp and comp in text:
            return True
        if comp and " " in comp:
            if all(w in text for w in comp.split()):
                return True
        return False

    @staticmethod
    def _extract_title_from_snippet(snippet: str, company_name: str) -> str:
        if not snippet:
            return ""
        s = snippet.strip()
        comp = (company_name or "").strip()
        if comp:
            m = re.search(rf"(.+?)\s+at\s+{re.escape(comp)}", s, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        for sep in ["·", " • ", "|"]:
            if sep in s:
                return s.split(sep, 1)[0].strip()
        return " ".join(s.split()[:6]).strip()

    # ------------------------------------------------------------------
    # DB write for discovered employees
    # ------------------------------------------------------------------

    def write_discovered_employees_to_db(
        self, employees: List[CompetitorEmployee], company_id: str,
        conn, title_extractor: Optional[TitleExtractor] = None,
    ) -> List[CompetitorEmployee]:
        logger.info(f"[Phase 3] Writing {len(employees)} discovered employees to DB")
        run_id = self._current_run_id

        with conn.cursor() as cur:
            for emp in employees:
                if emp.canonical_linkedin_id:
                    cur.execute(
                        "SELECT employee_id FROM spectre.employees WHERE canonical_linkedin_id = %s LIMIT 1",
                        (emp.canonical_linkedin_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        emp.employee_id = row["employee_id"]
                        emp.company_id = company_id
                        continue

                if title_extractor:
                    try:
                        extracted = title_extractor.extract_title_and_company(
                            headline=emp.title or "", experience=[], name=emp.name, bright_profile=None,
                        )
                        if extracted.get("title"):
                            emp.title = extracted["title"]
                    except Exception as e:
                        logger.warning(f"Title extraction failed for {emp.name}: {e}")

                eid = str(uuid.uuid4())
                cur.execute(
                    """INSERT INTO spectre.employees (
                        employee_id, full_name, current_title, current_company_id,
                        linkedin_url, canonical_linkedin_id, raw_json,
                        profile_cache_text, last_processed_run_id, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                    ON CONFLICT (employee_id) DO NOTHING""",
                    (eid, emp.name, emp.title, company_id, emp.linkedin_url,
                     emp.canonical_linkedin_id, Json(emp.raw_data), "Matched via MIRAGE", run_id),
                )
                emp.employee_id = eid
                emp.company_id = company_id

        return employees


# =============================================================================
# Phase 4: Profile Matcher — 4-WEIGHT PER-COMPANY SYSTEM
# =============================================================================

class ProfileMatcher:
    """
    4-weight scoring: Company(30) + Role(30) + Department(20) + Experience(20).
    Matching done per-company via the parallel pipeline.
    """

    MIN_SCORE = 40.0
    BATCH_SIZE = 15
    WEIGHT_COMPANY = 30
    WEIGHT_ROLE = 30
    WEIGHT_DEPARTMENT = 20
    WEIGHT_EXPERIENCE = 20

    DEPT_KEYWORDS = {
        "engineering": ["engineer", "developer", "tech", "software", "devops", "sre"],
        "sales": ["sales", "account", "business development", "revenue", "bd"],
        "marketing": ["marketing", "growth", "digital", "campaign", "brand"],
        "product": ["product", "pm", "product manager", "product owner"],
        "data": ["data", "analyst", "analytics", "scientist", "ml", "ai"],
        "finance": ["finance", "accounting", "fp&a", "cfo", "controller"],
        "operations": ["operations", "ops", "logistics", "supply chain"],
        "design": ["design", "ux", "ui", "designer", "creative"],
        "education": ["education", "instructor", "curriculum", "learning",
                      "training", "tutor", "teaching", "academic", "cpa",
                      "review", "exam", "course"],
        "legal": ["legal", "compliance", "regulatory", "counsel"],
        "hr": ["hr", "people", "talent", "recruiting", "human resources"],
    }

    def __init__(self, gpt: GPTInterface):
        self.gpt = gpt

    def _validate_company_membership(
        self, employees: List[CompetitorEmployee], company_name: str,
    ) -> List[CompetitorEmployee]:
        comp_lower = company_name.lower().strip()
        comp_words = set(comp_lower.split())
        validated = []
        for emp in employees:
            emp_company = (emp.company or "").lower().strip()
            if emp_company == comp_lower:
                validated.append(emp); continue
            if comp_lower in emp_company or emp_company in comp_lower:
                validated.append(emp); continue
            if comp_words and all(w in emp_company for w in comp_words):
                validated.append(emp); continue
            if comp_lower in f"{emp.title} {emp.search_snippet}".lower():
                validated.append(emp); continue
        return validated

    def _prefilter_with_weights(
        self, target: TargetEmployeeProfile,
        candidates: List[CompetitorEmployee], company_name: str,
    ) -> List[CompetitorEmployee]:
        target_title_words = set(target.title.lower().split())
        target_dept = target.department.lower()

        scored = []
        for cand in candidates:
            total_weight = 0
            cand_text = (cand.title + " " + cand.search_snippet).lower()

            # Company (30)
            cand_company = (cand.company or "").lower().strip()
            comp_lower = company_name.lower().strip()
            if comp_lower in cand_company or cand_company in comp_lower:
                total_weight += self.WEIGHT_COMPANY
            else:
                total_weight += self.WEIGHT_COMPANY // 2

            # Role (30)
            cand_words = set(cand.title.lower().split())
            overlap = len(target_title_words & cand_words)
            total_weight += min(overlap * 10, self.WEIGHT_ROLE)

            # Department (20)
            kws = self.DEPT_KEYWORDS.get(target_dept, [])
            dept_matches = sum(1 for kw in kws if kw in cand_text)
            total_weight += min(dept_matches * 7, self.WEIGHT_DEPARTMENT)

            # Experience (20)
            t_sen = self._extract_seniority(target.title)
            c_sen = self._extract_seniority(cand.title)
            if t_sen == c_sen:
                total_weight += self.WEIGHT_EXPERIENCE
            elif abs(self._seniority_to_level(t_sen) - self._seniority_to_level(c_sen)) <= 1:
                total_weight += self.WEIGHT_EXPERIENCE // 2

            scored.append((total_weight, cand))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for w, c in scored if w > 0][:80]

    def _compute_4weight_bonus(
        self, target: TargetEmployeeProfile, match: EmployeeMatch, company_name: str,
    ) -> float:
        bonus = 0.0

        # Company confirmation
        match_company = (match.competitor_company or "").lower().strip()
        expected = company_name.lower().strip()
        if expected in match_company or match_company in expected:
            bonus += 5.0

        # Role overlap
        target_words = set(target.title.lower().split())
        match_words = set((match.competitor_role or "").lower().split())
        bonus += min(len(target_words & match_words) * 2.0, 5.0)

        # Department alignment
        target_dept = target.department.lower()
        match_text = (match.competitor_role or "").lower()
        dept_signals = {
            "engineering": ["engineer", "developer", "software"],
            "sales": ["sales", "account", "revenue"],
            "marketing": ["marketing", "growth"],
            "education": ["education", "instructor", "cpa", "review", "exam"],
            "product": ["product"],
            "data": ["data", "analyst"],
        }
        if any(kw in match_text for kw in dept_signals.get(target_dept, [])):
            bonus += 3.0

        # Seniority alignment
        if self._extract_seniority(target.title) == self._extract_seniority(match.competitor_role or ""):
            bonus += 2.0

        return min(bonus, 15.0)

    def _score_batch(
        self, target: TargetEmployeeProfile, candidates: List[CompetitorEmployee],
        company_name: str, threshold: float,
    ) -> List[EmployeeMatch]:
        system_prompt = f"""You are an HR analyst comparing employee profiles for competitive intelligence.

CRITICAL RULES:
1. Only score candidates who CURRENTLY work at {company_name}
2. Reject anyone with "former", "ex-", "previous" employment signals
3. Score 0-100 based on similarity to the target

Scoring criteria (weighted):
- Company match (30%): Must currently work at {company_name}
- Role similarity (30%): How similar is their job title/function?
- Department alignment (20%): Same functional area?
- Experience/seniority (20%): Similar career level?

Return ONLY JSON:
{{"matches": [
  {{"candidate_index": 0, "similarity_score": 85, "confidence": "high",
   "matching_factors": {{"company_match": true, "role_similarity": "explanation",
   "department_match": "explanation", "seniority_match": "explanation"}},
   "competitor_role": "Clean Job Title"}}
]}}

Only include scores >= {int(threshold)}. Reject candidates NOT at {company_name}."""

        target_summary = {
            "name": target.name, "title": target.title, "department": target.department,
            "seniority": target.seniority_level, "skills": target.key_skills[:10],
            "experience_years": target.experience_years, "company": target.company,
        }
        cand_summary = [
            {"index": idx, "name": c.name, "title": c.title,
             "company": c.company, "snippet": c.search_snippet[:200]}
            for idx, c in enumerate(candidates)
        ]

        user_prompt = (
            f"TARGET EMPLOYEE (find similar people at {company_name}):\n"
            f"{json.dumps(target_summary, indent=2)}\n\n"
            f"CANDIDATES (should all be at {company_name}):\n"
            f"{json.dumps(cand_summary, indent=2)}\n\n"
            f"Score each candidate. ONLY include people currently at {company_name}.\n"
            "Return ONLY JSON."
        )

        try:
            response = self.gpt.call_gpt(system_prompt, user_prompt, temperature=0.2, max_tokens=2000)
            data = safe_json_parse(response)
            if not data or "matches" not in data:
                return []

            matches = []
            for s in data["matches"]:
                idx = s.get("candidate_index", -1)
                score = float(s.get("similarity_score") or 0)
                if not (0 <= idx < len(candidates) and score >= threshold):
                    continue
                cand = candidates[idx]

                # Final company gate
                cand_comp = (cand.company or "").lower().strip()
                expected_comp = company_name.lower().strip()
                if expected_comp not in cand_comp and cand_comp not in expected_comp:
                    logger.warning(f"[Phase 4] REJECTED: {cand.name} company={cand.company} != {company_name}")
                    continue

                matches.append(EmployeeMatch(
                    similarity_score=score,
                    target_employee_id=target.employee_id,
                    matched_employee_id=cand.employee_id or "",
                    target_employee=target.name,
                    competitor_employee=cand.name,
                    competitor_company=company_name,
                    match_rationale=json.dumps(s.get("matching_factors", {})),
                    linkedin_url=cand.linkedin_url,
                    competitor_role=s.get("competitor_role", "") or cand.title,
                    matching_factors=s.get("matching_factors", {}),
                    confidence=s.get("confidence", "medium"),
                ))
            return matches
        except Exception as e:
            logger.error(f"GPT scoring failed: {e}")
            return []

    @staticmethod
    def _extract_seniority(title: str) -> str:
        tl = title.lower()
        if any(x in tl for x in ["junior", "entry", "associate", "jr"]):
            return "junior"
        if any(x in tl for x in ["director", "vp", "vice president", "head", "chief"]):
            return "executive"
        if any(x in tl for x in ["senior", "lead", "principal", "staff", "sr"]):
            return "senior"
        if any(x in tl for x in ["manager", "supervisor"]):
            return "manager"
        return "mid"

    @staticmethod
    def _seniority_to_level(seniority: str) -> int:
        return {"junior": 1, "mid": 2, "senior": 3, "manager": 4, "executive": 5}.get(seniority, 2)


# =============================================================================
# Phase 5: Bright Data Enrichment
# =============================================================================

class BrightDataScraper:
    def __init__(self):
        self.api_key = os.getenv("BRIGHT_DATA_API_KEY")
        self.dataset_id = os.getenv("BRIGHT_DATA_DATASET_ID")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def scrape_profiles_one_shot(self, urls: List[str]) -> List[Dict]:
        if not (self.api_key and self.dataset_id):
            logger.warning("Bright Data not configured")
            return []

        urls = _dedupe_preserve_order([u for u in urls if u])[:100]
        if not urls:
            return []

        logger.info(f"Processing {len(urls)} URLs with Bright Data")
        results = []
        for url in urls:
            result = self._fetch_profile(url)
            if result.get("profiles"):
                results.extend(result["profiles"])

        logger.info(f"Bright Data collected {len(results)} profiles")
        _global_cost.track_bright_data_rows(len(results))
        return results

    def _fetch_profile(self, url: str) -> Dict:
        slug = canonicalize_linkedin_url(url)
        if not slug:
            return {"url": url, "profiles": [], "error": "slug_extraction_failed"}

        logger.info(f"Bright Data lookup: {slug}")
        start = time.time()

        try:
            resp = requests.post(
                "https://api.brightdata.com/datasets/filter",
                headers=self.headers,
                json={"dataset_id": self.dataset_id, "records_limit": 1000,
                      "filter": {"name": "url", "operator": "includes", "value": slug}},
                timeout=30,
            )
            if resp.status_code != 200:
                return {"url": url, "profiles": [], "error": f"trigger_{resp.status_code}"}
            snapshot_id = resp.json().get("snapshot_id")
        except Exception as e:
            return {"url": url, "profiles": [], "error": str(e)}

        status_url = f"https://api.brightdata.com/datasets/snapshots/{snapshot_id}"
        while time.time() - start < 300:
            try:
                sr = requests.get(status_url, headers=self.headers, timeout=15)
                if sr.status_code == 200:
                    status = sr.json().get("status")
                    if status == "ready":
                        break
                    if status == "failed":
                        return {"url": url, "profiles": [], "error": "snapshot_failed"}
            except Exception:
                pass
            time.sleep(5)
        else:
            return {"url": url, "profiles": [], "error": "timeout"}

        dl_url = f"https://api.brightdata.com/datasets/snapshots/{snapshot_id}/download?format=json"
        try:
            while True:
                dr = requests.get(dl_url, headers=self.headers, timeout=30)
                if dr.status_code == 200:
                    break
                if dr.status_code == 202:
                    time.sleep(5)
                    continue
                return {"url": url, "profiles": [], "error": f"download_{dr.status_code}"}
            return {"url": url, "profiles": dr.json(), "error": None}
        except Exception as e:
            return {"url": url, "profiles": [], "error": str(e)}


class BrightDataEnricher:
    def __init__(self, scraper: BrightDataScraper, conn):
        self.scraper = scraper
        self.conn = conn

    def enrich_matched_employees(
        self, matches: List[EmployeeMatch],
        all_employees: Dict[str, CompetitorEmployee], run_id: str,
    ) -> Tuple[List[EmployeeMatch], Dict[str, Dict]]:
        matches = sorted(matches, key=lambda m: m.similarity_score, reverse=True)[:TOP_K_MATCHES]
        logger.info(f"[Phase 5] Enriching TOP {len(matches)} matched employees")

        urls = []
        eid_to_url = {}
        for m in matches:
            emp = all_employees.get(m.matched_employee_id)
            if emp and emp.linkedin_url:
                urls.append(emp.linkedin_url)
                eid_to_url[m.matched_employee_id] = emp.linkedin_url

        if not urls:
            return matches, {}

        urls = _dedupe_preserve_order(urls)[:TOP_K_MATCHES]
        profiles = self.scraper.scrape_profiles_one_shot(urls)

        url_to_profile = {}
        for p in profiles:
            pu = p.get("url", "")
            if pu:
                url_to_profile[normalize_linkedin_url(pu)] = p

        bright_profiles = {}
        for eid, url in eid_to_url.items():
            profile = url_to_profile.get(normalize_linkedin_url(url))
            if profile:
                bright_profiles[eid] = profile

        logger.info(f"[Phase 5] Enriched {len(bright_profiles)} employees")
        return matches, bright_profiles


# =============================================================================
# Phase 6: DB Writer
# =============================================================================

class MirageDBWriter:
    def __init__(self, conn):
        self.conn = conn

    def create_run(self, target_employee_id: str, target_company_id: str) -> str:
        run_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO spectre.runs (
                    run_id, scope, status, target_company_id,
                    requested_employee_count, allow_cache, freshness_days,
                    force_refresh, config_json, raw_json, started_at, created_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW()) RETURNING run_id""",
                (run_id, "competitive_intelligence", "in_progress",
                 target_company_id, None, True, 30, False, Json({}),
                 Json({"target_employee_id": target_employee_id,
                       "target_company_id": target_company_id,
                       "created_at": datetime.now().isoformat()})),
            )
            run_id = cur.fetchone()["run_id"]
        logger.info(f"[DB] Created run: {run_id}")
        return run_id

    def update_run_status(self, run_id: str, status: str, summary: Dict = None):
        with self.conn.cursor() as cur:
            if summary:
                cur.execute(
                    "UPDATE spectre.runs SET status=%s, raw_json=%s, ended_at=NOW() WHERE run_id=%s",
                    (status, Json(summary), run_id),
                )
            else:
                cur.execute("UPDATE spectre.runs SET status=%s WHERE run_id=%s", (status, run_id))

    def add_company_to_run(self, run_id: str, company_id: str, role: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO spectre.run_companies (
                    run_id, company_id, role_in_run, role, raw_json, created_at
                ) VALUES (%s,%s,%s,%s,%s,NOW()) ON CONFLICT (run_id, company_id) DO NOTHING""",
                (run_id, company_id, role, role, Json({})),
            )

    def add_employee_to_run(self, run_id: str, employee_id: str, role: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO spectre.run_employees (
                    run_id, employee_id, role_in_run, source_company_id, raw_json, created_at
                ) VALUES (%s,%s,%s,%s,%s,NOW()) ON CONFLICT (run_id, employee_id) DO NOTHING""",
                (run_id, employee_id, role, None, Json({})),
            )

    def write_matches(self, run_id: str, matches: List[EmployeeMatch]):
        logger.info(f"[DB] Writing {len(matches)} matches")
        with self.conn.cursor() as cur:
            for m in matches:
                cur.execute(
                    """SELECT 1 FROM spectre.employee_matches
                       WHERE run_id=%s AND employee_id=%s AND matched_employee_id=%s LIMIT 1""",
                    (run_id, m.target_employee_id, m.matched_employee_id),
                )
                if cur.fetchone():
                    continue

                cur.execute(
                    "SELECT current_company_id FROM spectre.employees WHERE employee_id=%s",
                    (m.matched_employee_id,),
                )
                row = cur.fetchone()
                matched_company_id = row["current_company_id"] if row else None
                if not matched_company_id:
                    logger.warning(f"No company_id for {m.matched_employee_id}, skipping")
                    continue

                cur.execute(
                    """INSERT INTO spectre.employee_matches (
                        run_id, employee_id, matched_employee_id, matched_name,
                        matched_title, matched_company_id, matched_company_name,
                        match_score, match_type, match_source, rationale_json,
                        raw_json, created_by_agent, model_name, created_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())""",
                    (run_id, m.target_employee_id, m.matched_employee_id,
                     m.competitor_employee, m.competitor_role, matched_company_id,
                     m.competitor_company, m.similarity_score, "competitor",
                     "mirage_hybrid", Json(m.matching_factors),
                     Json({"linkedin_url": m.linkedin_url, "confidence": m.confidence}),
                     "mirage_hybrid_v7.0", "gpt-4o"),
                )
                self.add_employee_to_run(run_id, m.matched_employee_id, "matched")

    def write_match_details(
        self, run_id: str, matches: List[EmployeeMatch],
        all_employees: Dict[str, CompetitorEmployee],
        bright_profiles: Dict[str, Dict] = None,
        title_extractor: Optional[TitleExtractor] = None,
    ):
        logger.info(f"[DB] Writing employee_details for {len(matches)} matches")
        bright_profiles = bright_profiles or {}

        with self.conn.cursor() as cur:
            for m in matches:
                eid = m.matched_employee_id
                emp = all_employees.get(eid)
                if not emp:
                    continue

                cur.execute(
                    "SELECT 1 FROM spectre.employee_details WHERE run_id=%s AND employee_id=%s LIMIT 1",
                    (run_id, eid),
                )
                if cur.fetchone():
                    continue

                bp = bright_profiles.get(eid, {})

                v_title = emp.title
                v_company = emp.company
                if title_extractor and bp:
                    try:
                        ext = title_extractor.extract_title_and_company(
                            headline=bp.get("position", "") or emp.title,
                            experience=bp.get("experience", []) or [],
                            name=emp.name, bright_profile=bp,
                        )
                        v_title = ext.get("title") or emp.title
                        v_company = ext.get("company") or emp.company
                    except Exception:
                        pass

                if bp:
                    vals = {
                        "v_name": bp.get("name", emp.name),
                        "v_about": bp.get("about", ""),
                        "v_location": bp.get("location", ""),
                        "v_city": bp.get("city", ""),
                        "v_country_code": bp.get("country_code", ""),
                        "v_followers": str(bp.get("followers", 0)),
                        "v_connections": str(bp.get("connections", 0)),
                        "v_experience": json.dumps(bp.get("experience", [])),
                        "v_languages": json.dumps(bp.get("languages", [])),
                        "v_education": json.dumps(bp.get("education", [])),
                        "v_activity": json.dumps(bp.get("activity", [])),
                        "v_courses": json.dumps(bp.get("courses", [])),
                        "v_posts": json.dumps(bp.get("posts", [])),
                        "rp_first_name": bp.get("first_name", ""),
                        "rp_last_name": bp.get("last_name", ""),
                        "rp_url": bp.get("url", emp.linkedin_url),
                        "rp_avatar": bp.get("avatar", ""),
                        "rp_banner": bp.get("banner_image", ""),
                        "rp_lid": bp.get("linkedin_id", ""),
                        "rp_ccid": bp.get("current_company_company_id", ""),
                    }
                    enriched = True
                else:
                    vals = {
                        "v_name": emp.name, "v_about": emp.search_snippet,
                        "v_location": "", "v_city": "", "v_country_code": "",
                        "v_followers": "0", "v_connections": "0",
                        "v_experience": "[]", "v_languages": "[]", "v_education": "[]",
                        "v_activity": "[]", "v_courses": "[]", "v_posts": "[]",
                        "rp_first_name": "", "rp_last_name": "",
                        "rp_url": emp.linkedin_url, "rp_avatar": "", "rp_banner": "",
                        "rp_lid": "", "rp_ccid": "",
                    }
                    enriched = False

                details_json = {
                    "name": vals["v_name"], "title": v_title, "company": v_company,
                    "linkedin_url": emp.linkedin_url,
                    "match_score": m.similarity_score,
                    "match_confidence": m.confidence,
                    "matching_factors": m.matching_factors,
                    "bright_data_enriched": enriched,
                }

                cur.execute(
                    """INSERT INTO spectre.employee_details (
                        run_id, employee_id, data_origin, details_json, raw_json,
                        created_by_agent, model_name,
                        v_name, v_title, v_company, v_position, v_about,
                        v_location, v_city, v_country_code,
                        v_followers, v_connections,
                        v_experience, v_languages, v_education, v_activity, v_courses, v_posts,
                        rp_first_name, rp_last_name, rp_url, rp_avatar, rp_banner_image,
                        rp_linkedin_profile_id, rp_current_company_company_id,
                        created_at
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,
                        NOW()
                    ) ON CONFLICT (run_id, employee_id) DO NOTHING""",
                    (run_id, eid, "fresh", Json(details_json),
                     Json(bp if bp else emp.raw_data),
                     "mirage_hybrid_v7.0", "gpt-4o",
                     vals["v_name"], v_title, v_company, v_title, vals["v_about"],
                     vals["v_location"], vals["v_city"], vals["v_country_code"],
                     vals["v_followers"], vals["v_connections"],
                     vals["v_experience"], vals["v_languages"], vals["v_education"],
                     vals["v_activity"], vals["v_courses"], vals["v_posts"],
                     vals["rp_first_name"], vals["rp_last_name"], vals["rp_url"],
                     vals["rp_avatar"], vals["rp_banner"],
                     vals["rp_lid"], vals["rp_ccid"]),
                )


# =============================================================================
# Main System
# =============================================================================

class MirageHybridSystem:
    """
    MIRAGE Hybrid System v7.0 — Cleaned Production Build

    Phase 0: Employee ID → resolve target from DB
    Phase 1: GPT competitor detection
    Phase 2: Build target profile from DB
    Phase 3+4: Parallel search + 4-weight match (early-stop)
    Phase 5: Bright Data enrichment (top 3)
    Phase 6: Write to DB
    """

    def __init__(self):
        logger.info("Initializing MIRAGE Hybrid System v7.0")
        self.conn = get_db_connection()
        self.gpt = GPTInterface()
        self.azure_gpt = AzureGPTClient(tracker=_token_tracker)
        self.title_extractor = TitleExtractor(self.gpt)

        self.target_resolver = TargetResolver(self.conn)
        self.competitor_detector = CompetitorDetector(self.gpt, self.conn)
        self.profile_builder = TargetProfileBuilder(self.gpt)
        self.employee_finder = CompetitorEmployeeFinder(gpt_client=self.azure_gpt)
        self.matcher = ProfileMatcher(self.gpt)
        self.enricher = BrightDataEnricher(BrightDataScraper(), self.conn)
        self.db_writer = MirageDBWriter(self.conn)
        logger.info("MIRAGE Hybrid System v7.0 initialized")

    async def run_full_analysis(
        self, employee_id: Optional[str] = None, run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not employee_id and not run_id:
            raise ValueError("Must provide employee_id, run_id, or both")

        start_time = time.time()
        logger.info("=" * 80)
        logger.info("MIRAGE HYBRID ANALYSIS START (v7.0)")
        logger.info(f"  Competitors: {NUM_COMPETITORS}  |  Mode: {BUSINESS_MODEL}  |  "
                     f"Top matches: {TOP_K_MATCHES}  |  Per company: {MATCHES_PER_COMPANY}")
        logger.info("=" * 80)

        # ── Phase 0: Resolve target + run_id ──
        # Priority: if run_id given, resolve employee from it.
        #           if employee_id given, find existing run from DB.
        if run_id and not employee_id:
            # run_id only → look up employee from the run
            employee_id, target_context = self.target_resolver.resolve_from_run_id(run_id)
        elif employee_id and not run_id:
            # employee_id only → resolve target, find existing run
            target_context = self.target_resolver.resolve_target_from_employee_id(employee_id)
            run_id = self.target_resolver.find_existing_run(employee_id)
            if not run_id:
                raise ValueError(
                    f"No existing run found for employee {employee_id}. "
                    "Please provide a run_id or create a run first."
                )
        else:
            # Both provided → use run_id, resolve employee
            target_context = self.target_resolver.resolve_target_from_employee_id(employee_id)

        logger.info(f"[Phase 0] Using run_id: {run_id}")
        logger.info(f"[Phase 0] Target: {target_context['employee_name']} at {target_context['company_name']}")

        target_company_id = target_context["current_company_id"]
        target_company_name = target_context["company_name"]

        # Update existing run status to in_progress
        self.db_writer.update_run_status(run_id, "in_progress")
        self.db_writer.add_company_to_run(run_id, target_company_id, "target")
        self.db_writer.add_employee_to_run(run_id, employee_id, "target")

        # Phase 1: Detect competitors (region-aware)
        emp_location = target_context["employee_data"].get("location") or ""
        company_region = target_context["company_data"].get("metadata_json") or {}
        if isinstance(company_region, str):
            company_region = json.loads(company_region)
        region = emp_location or company_region.get("region", "") or company_region.get("country", "")
        logger.info(f"[Phase 1] Detected region: {region or 'none'}")

        competitors = self.competitor_detector.detect_competitors(
            company_name=target_company_name,
            industry=target_context["company_data"].get("industry", ""),
            description=target_context["company_data"].get("description", ""),
            num_competitors=NUM_COMPETITORS,
            business_model=BUSINESS_MODEL,
            region=region,
        )
        for comp in competitors:
            comp.company_id = self.competitor_detector.get_or_create_company(
                comp.name, comp.industry, comp.description,
            )
            self.db_writer.add_company_to_run(run_id, comp.company_id, "competitor")

        logger.info(f"Detected {len(competitors)} competitors: {[c.name for c in competitors]}")

        # Phase 2: Build target profile
        target_profile = self.profile_builder.build_profile_from_db(target_context)
        dbg("Target profile", {
            "name": target_profile.name, "title": target_profile.title,
            "department": target_profile.department, "seniority": target_profile.seniority_level,
            "skills": target_profile.key_skills[:10],
        })

        # Phase 3+4: Parallel search + match (MATCHES_PER_COMPANY per company)
        self.employee_finder._current_run_id = run_id
        competitor_employees_map, matches_by_company = await self.employee_finder.search_and_match_parallel(
            target_profile=target_profile, competitors=competitors,
            matcher=self.matcher, business_model=BUSINESS_MODEL,
            top_k_per_company=MATCHES_PER_COMPANY, score_threshold=40.0,
            max_concurrent_companies=4,
        )

        # Write discovered employees to DB
        all_competitor_employees: List[CompetitorEmployee] = []
        for comp in competitors:
            emps = competitor_employees_map.get(comp.name, [])
            if emps:
                emps = self.employee_finder.write_discovered_employees_to_db(
                    employees=emps, company_id=comp.company_id,
                    conn=self.conn, title_extractor=self.title_extractor,
                )
                competitor_employees_map[comp.name] = emps
            all_competitor_employees.extend(emps)

        # Update match employee_ids from DB-written employees
        emp_by_url: Dict[str, CompetitorEmployee] = {}
        for emp in all_competitor_employees:
            if emp.employee_id and emp.linkedin_url:
                emp_by_url[emp.linkedin_url.strip().rstrip("/").lower()] = emp

        for company_matches in matches_by_company.values():
            for m in company_matches:
                if not m.matched_employee_id:
                    url_key = (m.linkedin_url or "").strip().rstrip("/").lower()
                    emp = emp_by_url.get(url_key)
                    if emp and emp.employee_id:
                        m.matched_employee_id = emp.employee_id

        # Flatten and select top TOP_K_MATCHES across all companies
        all_matches: List[EmployeeMatch] = []
        for company_matches in matches_by_company.values():
            all_matches.extend(company_matches)
        all_matches.sort(key=lambda m: m.similarity_score, reverse=True)
        all_matches = all_matches[:TOP_K_MATCHES]

        logger.info(f"Found {len(all_matches)} top matches across {len(matches_by_company)} companies")

        # Phase 5: Enrich with Bright Data
        employee_lookup = {emp.employee_id: emp for emp in all_competitor_employees if emp.employee_id}
        enriched_matches, bright_profiles = self.enricher.enrich_matched_employees(
            all_matches, employee_lookup, run_id,
        )

        # Phase 6: Write to DB
        self.db_writer.write_matches(run_id, enriched_matches)
        self.db_writer.write_match_details(
            run_id, enriched_matches, employee_lookup, bright_profiles,
            title_extractor=self.title_extractor,
        )

        execution_time = time.time() - start_time
        summary = {
            "target_employee_id": employee_id,
            "target_company_id": target_company_id,
            "target_company_name": target_company_name,
            "num_competitors": len(competitors),
            "competitors_with_matches": len(matches_by_company),
            "total_discovered_employees": len(all_competitor_employees),
            "total_matches": len(enriched_matches),
            "matches_by_company": {k: len(v) for k, v in matches_by_company.items()},
            "execution_time_seconds": round(execution_time, 2),
            "business_model": BUSINESS_MODEL,
            "matching_algorithm": "4-weight_per_company_v7.0",
            "completed_at": datetime.now().isoformat(),
        }
        self.db_writer.update_run_status(run_id, "completed", summary)

        logger.info(f"MIRAGE COMPLETE: {len(enriched_matches)} matches in {execution_time:.1f}s")

        return {
            "run_id": run_id,
            "target_employee_id": employee_id,
            "target_company_name": target_company_name,
            "competitors": [asdict(c) for c in competitors],
            "matches_by_company": {k: len(v) for k, v in matches_by_company.items()},
            "total_matches": len(enriched_matches),
            "execution_time_seconds": round(execution_time, 2),
            "summary": summary,
            "cost_summary": _token_tracker.get_summary(),
        }


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="MIRAGE Hybrid API",
    description="Competitive intelligence employee matching system",
    version="7.0",
)



class AnalysisRequest(BaseModel):
    employee_id: Optional[str] = Field(default=None, description="UUID of the target employee")
    run_id: Optional[str] = Field(default=None, description="UUID of an existing run")


class AnalysisResponse(BaseModel):
    run_id: str
    target_employee_id: str
    target_company_name: str
    competitors: List[Dict[str, Any]]
    matches_by_company: Dict[str, int]
    total_matches: int
    execution_time_seconds: float
    summary: Dict[str, Any]
    cost_summary: Dict[str, Any]


# Lazy singleton — created on first request
_mirage_instance: Optional[MirageHybridSystem] = None

def get_mirage() -> MirageHybridSystem:
    global _mirage_instance
    if _mirage_instance is None:
        _mirage_instance = MirageHybridSystem()
    return _mirage_instance


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "7.0", "service": "mirage-hybrid"}


@app.post("/peers", response_model=AnalysisResponse)
async def run_analysis(req: AnalysisRequest):
    if not req.employee_id and not req.run_id:
        raise HTTPException(status_code=400, detail="Must provide employee_id, run_id, or both")
    try:
        mirage = get_mirage()
        result = await mirage.run_full_analysis(
            employee_id=req.employee_id, run_id=req.run_id,
        )
        return AnalysisResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CLI Entry Point
# =============================================================================

async def cli_main():
    import argparse

    parser = argparse.ArgumentParser(description="MIRAGE Hybrid v7.0")
    parser.add_argument("--employee-id", default=None, help="UUID of target employee")
    parser.add_argument("--run-id", default=None, help="UUID of an existing run")
    args = parser.parse_args()

    if not args.employee_id and not args.run_id:
        parser.error("Must provide --employee-id, --run-id, or both")

    print(f"{'=' * 80}\nMIRAGE HYBRID v7.0\n{'=' * 80}")
    if args.employee_id:
        print(f"Employee ID: {args.employee_id}")
    if args.run_id:
        print(f"Run ID: {args.run_id}")
    print(f"Competitors: {NUM_COMPETITORS}  |  Mode: {BUSINESS_MODEL}  |  "
          f"Top matches: {TOP_K_MATCHES}  |  Per company: {MATCHES_PER_COMPANY}")
    print(f"Weights: Company(30) + Role(30) + Department(20) + Experience(20)\n{'=' * 80}")

    try:
        mirage = MirageHybridSystem()
        results = await mirage.run_full_analysis(
            employee_id=args.employee_id, run_id=args.run_id,
        )

        print(f"\n{'=' * 80}\nRESULTS\n{'=' * 80}")
        print(f"Run ID: {results['run_id']}")
        print(f"Target: {results['target_company_name']}")
        print(f"Matches: {results['total_matches']}")
        for comp, count in results.get("matches_by_company", {}).items():
            print(f"  - {comp}: {count}")
        print(f"Time: {results['execution_time_seconds']}s\n{'=' * 80}")
        _token_tracker.print_summary()

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        port = int(os.getenv("MIRAGE_PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        asyncio.run(cli_main())