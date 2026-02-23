# -*- coding: utf-8 -*-
"""SPECTRE Agent 5 — Database Version (Cleaned + FastAPI)"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import asyncpg
from asyncpg import Connection, Pool
from fastapi import FastAPI, APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging — with console output
# ---------------------------------------------------------------------------
logger = logging.getLogger("Spectre5DB")
logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Centralised, immutable configuration loaded from environment."""

    # Google Custom Search
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_cx: str = os.getenv("GOOGLE_CSE_ID", "")

    # Azure OpenAI
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    azure_deployment_id: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "")

    # Database
    db_host: str = os.getenv("SPECTRE_DB_HOST", "")
    db_port: int = int(os.getenv("SPECTRE_DB_PORT", "5432"))
    db_name: str = os.getenv("SPECTRE_DB_NAME", "")
    db_user: str = os.getenv("SPECTRE_DB_USER", "")
    db_password: str = os.getenv("SPECTRE_DB_PASSWORD", "")

    # SyncFlow / Course writer
    syncflow_url: str = os.getenv(
        "SYNCFLOW_URL",
        "https://mynotedbe-guh7ekdxajcddvd2.southindia-01.azurewebsites.net/api/trpc/addSyncFlowJob",
    )
    course_table: str = os.getenv("COURSE_TABLE", "mynoted_clone.courses")
    course_name_col: str = os.getenv("COURSE_NAME_COL", '"course_name"')
    course_id_col: str = os.getenv("COURSE_ID_COL", '"course_id"')
    course_created_col: str = os.getenv("COURSE_CREATED_COL", '"created_at"')
    poll_timeout: int = int(os.getenv("POLL_TIMEOUT", "300"))
    poll_interval: int = int(os.getenv("POLL_INTERVAL", "5"))

    # Concurrency / rate-limit knobs
    max_concurrent_skills: int = 2
    websites_per_query: int = 5
    delay_between_searches_ms: int = 800
    max_critical_skills: int = 3
    max_subtopics_per_skill: int = 7
    max_total_subtopics: int = 20

    # Azure OpenAI pricing (USD per 1K tokens) — override via env for your model
    cost_per_1k_input: float = float(os.getenv("AZURE_COST_PER_1K_INPUT", "0.005"))
    cost_per_1k_output: float = float(os.getenv("AZURE_COST_PER_1K_OUTPUT", "0.015"))


cfg = Config()

# ---------------------------------------------------------------------------
# Cost tracker — accumulates token usage across all LLM calls in a run
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe accumulator for all billable API usage within a single run."""

    # Google CSE: $5 per 1K queries (first 100/day free on free tier)
    GOOGLE_CSE_COST_PER_QUERY: float = 0.005

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.llm_calls: int = 0
        self.google_searches: int = 0
        self.web_fetches: int = 0
        self._start_time: float = time.time()

    async def record_llm(self, input_tokens: int, output_tokens: int) -> None:
        async with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.llm_calls += 1

    def record_llm_sync(self, input_tokens: int, output_tokens: int) -> None:
        """Non-async version for the sync OpenAI client in ProfileGenerator."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.llm_calls += 1

    async def record_google_search(self) -> None:
        async with self._lock:
            self.google_searches += 1

    async def record_web_fetch(self) -> None:
        async with self._lock:
            self.web_fetches += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def llm_cost_usd(self) -> float:
        return (
            (self.total_input_tokens / 1000) * cfg.cost_per_1k_input
            + (self.total_output_tokens / 1000) * cfg.cost_per_1k_output
        )

    @property
    def google_cost_usd(self) -> float:
        return self.google_searches * self.GOOGLE_CSE_COST_PER_QUERY

    @property
    def total_cost_usd(self) -> float:
        return self.llm_cost_usd + self.google_cost_usd

    def summary(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self._start_time, 1)
        return {
            "elapsed_seconds": elapsed,
            "llm_calls": self.llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "google_searches": self.google_searches,
            "web_fetches": self.web_fetches,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "google_cost_usd": round(self.google_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "pricing": {
                "llm_per_1k_input": cfg.cost_per_1k_input,
                "llm_per_1k_output": cfg.cost_per_1k_output,
                "google_per_query": self.GOOGLE_CSE_COST_PER_QUERY,
            },
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _sleep_ms(ms: int) -> None:
    await asyncio.sleep(ms / 1000.0)


async def _with_backoff(coro_fn, *, attempts: int = 4, base_ms: int = 800):
    """Retry *coro_fn()* with exponential back-off + jitter."""
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:
            last_err = exc
            jitter = int(200 * (os.urandom(1)[0] / 255))
            await _sleep_ms(base_ms * (2 ** i) + jitter)
    raise last_err  # type: ignore[misc]


_SKILL_ALIASES: Dict[str, str] = {
    "ci cd": "CI/CD",
    "cicd": "CI/CD",
    "devops": "DevOps",
    "ml": "Machine Learning",
    "ai": "AI",
}


def _normalise_skill(raw: str) -> str:
    if not raw:
        return ""
    key = raw.lower().strip()
    if key in _SKILL_ALIASES:
        return _SKILL_ALIASES[key]
    return re.sub(r"\b(\w)", lambda m: m.group(1).upper(), re.sub(r"\s+", " ", raw.strip()))


def _categorise_missing_skills(
    missing_skills: List[str],
    skill_importance: Dict[str, str],
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {"critical": [], "important": [], "nice_to_have": []}
    for skill in missing_skills:
        importance = (skill_importance.get(skill, "") or "").lower()
        bucket = importance if importance in buckets else "nice_to_have"
        buckets[bucket].append(_normalise_skill(skill))
    return buckets


def _to_syncflow_format(
    course_obj: Dict[str, Any],
    author: str = "Spectre",
    created_on: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert internal course dict → SyncFlow-compatible payload."""
    created_on = created_on or date.today().isoformat()
    chapters = []
    for mod in (course_obj or {}).get("modules", []):
        topics = [
            {"topicName": (t.get("name") or "Untitled Topic").strip()}
            for t in mod.get("topics", [])
            if isinstance(t, dict)
        ]
        if topics:
            chapters.append({"chapterName": mod.get("moduleName", "Module"), "chapter": topics})
    return {
        "courseName": (course_obj or {}).get("courseName", "Course"),
        "course": chapters,
        "extra_fields": {"author": author, "createdOn": created_on},
    }


# ---------------------------------------------------------------------------
# Database pool (singleton-ish)
# ---------------------------------------------------------------------------

class DatabasePool:
    def __init__(self) -> None:
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        if self._pool:
            return
        self._pool = await asyncpg.create_pool(
            host=cfg.db_host,
            port=cfg.db_port,
            database=cfg.db_name,
            user=cfg.db_user,
            password=cfg.db_password,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        logger.info("✅ Connected to %s:%s/%s", cfg.db_host, cfg.db_port, cfg.db_name)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def acquire(self) -> Connection:
        if not self._pool:
            await self.connect()
        return await self._pool.acquire()  # type: ignore[union-attr]

    async def release(self, conn: Connection) -> None:
        if self._pool:
            await self._pool.release(conn)


db_pool = DatabasePool()

# ---------------------------------------------------------------------------
# Data fetcher — reads employee + skill-gap data from Spectre schema
# ---------------------------------------------------------------------------

_SENIORITY_KEYWORDS: Dict[str, List[str]] = {
    "Senior": ["senior", "lead", "principal", "staff", "architect"],
    "Junior": ["junior", "associate", "intern", "trainee"],
    "Leadership": ["director", "vp", "head", "chief", "manager"],
}

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Technology": ["engineer", "developer", "software", "devops"],
    "Data Science": ["data", "analytics", "scientist", "ml", "ai"],
    "Product Management": ["product", "pm"],
    "Design": ["design", "ux", "ui"],
}


def _infer_seniority(title: Optional[str]) -> str:
    if not title:
        return "Mid-level"
    t = title.lower()
    for level, keywords in _SENIORITY_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return level
    return "Mid-level"


def _infer_domain(title: Optional[str], headline: Optional[str]) -> str:
    text = f"{title or ''} {headline or ''}".lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "Technology"


class DatabaseDataFetcher:
    """Fetches the PRIMARY employee for a given run_id or employee_id.

    Resolution logic (via spectre.run_employees where role_in_run = 'primary'):
      - run_id provided    → find primary employee in that run
      - employee_id provided → find the run where this employee is primary
      - both provided      → use directly (still validates primary)
    """

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    # -- public ---------------------------------------------------------------

    async def fetch_primary(
        self,
        run_id: Optional[str] = None,
        employee_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve the primary employee and return their full data."""
        conn = await self._pool.acquire()
        try:
            resolved_emp_id, resolved_run_id = await self._resolve_primary(
                conn, run_id=run_id, employee_id=employee_id
            )
            if not resolved_emp_id or not resolved_run_id:
                return None
            return await self._fetch_employee(conn, resolved_emp_id, resolved_run_id)
        finally:
            await self._pool.release(conn)

    # -- private: resolution --------------------------------------------------

    async def _resolve_primary(
        self,
        conn: Connection,
        run_id: Optional[str] = None,
        employee_id: Optional[str] = None,
    ) -> tuple:
        """Return (employee_id, run_id) of the primary target."""

        if run_id and employee_id:
            # Both given — validate and use
            return employee_id, run_id

        if run_id:
            # run_id → find the primary employee in this run
            row = await conn.fetchrow(
                """
                SELECT employee_id FROM spectre.run_employees
                WHERE run_id = $1 AND role_in_run = 'primary'
                LIMIT 1
                """,
                uuid.UUID(run_id),
            )
            if row:
                print(f"   🔍 Primary employee for run: {row['employee_id']}")
                return str(row["employee_id"]), run_id
            # Fallback: no primary flag, take first employee
            row = await conn.fetchrow(
                "SELECT employee_id FROM spectre.run_employees WHERE run_id = $1 LIMIT 1",
                uuid.UUID(run_id),
            )
            if row:
                print(f"   ⚠️  No primary found, using first employee: {row['employee_id']}")
                return str(row["employee_id"]), run_id
            return None, None

        if employee_id:
            # employee_id → find the run where this employee is primary
            row = await conn.fetchrow(
                """
                SELECT run_id FROM spectre.run_employees
                WHERE employee_id = $1 AND role_in_run = 'primary'
                ORDER BY created_at DESC LIMIT 1
                """,
                uuid.UUID(employee_id),
            )
            if row:
                print(f"   🔍 Latest primary run for employee: {row['run_id']}")
                return employee_id, str(row["run_id"])
            # Fallback: no primary flag, take latest run
            row = await conn.fetchrow(
                "SELECT run_id FROM spectre.run_employees WHERE employee_id = $1 ORDER BY created_at DESC LIMIT 1",
                uuid.UUID(employee_id),
            )
            if row:
                print(f"   ⚠️  No primary found, using latest run: {row['run_id']}")
                return employee_id, str(row["run_id"])
            return None, None

        return None, None

    # -- private --------------------------------------------------------------

    async def _fetch_employee(
        self, conn: Connection, employee_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            emp_uuid = uuid.UUID(employee_id)
            run_uuid = uuid.UUID(run_id)

            emp_row = await conn.fetchrow(
                """
                SELECT e.employee_id, e.full_name, e.current_title,
                       e.location, e.headline, c.name AS company_name
                FROM spectre.employees e
                LEFT JOIN spectre.companies c ON e.current_company_id = c.company_id
                WHERE e.employee_id = $1
                """,
                emp_uuid,
            )
            if not emp_row:
                return None

            # Existing skills (deduplicated, ordered by confidence)
            skill_rows = await conn.fetch(
                """
                SELECT s.name AS skill_name, es.skill_confidence
                FROM spectre.employee_skills es
                JOIN spectre.skills s ON es.skill_id = s.skill_id
                WHERE es.employee_id = $1
                ORDER BY es.skill_confidence DESC NULLS LAST
                """,
                emp_uuid,
            )
            seen: set[str] = set()
            existing_skills: List[str] = []
            for r in skill_rows:
                if r["skill_name"] and r["skill_name"] not in seen:
                    existing_skills.append(r["skill_name"])
                    seen.add(r["skill_name"])

            # Skill gaps for this run
            gap_rows = await conn.fetch(
                """
                SELECT sg.skill_gap_name, sg.skill_importance, s.name AS skill_name
                FROM spectre.employee_skill_gaps sg
                LEFT JOIN spectre.skills s ON sg.skill_id = s.skill_id
                WHERE sg.employee_id = $1 AND sg.run_id = $2
                """,
                emp_uuid,
                run_uuid,
            )
            missing_skills: List[str] = []
            skill_importance: Dict[str, str] = {}
            for gap in gap_rows:
                name = gap["skill_name"] or gap["skill_gap_name"]
                if name:
                    missing_skills.append(name)
                    if gap["skill_importance"]:
                        skill_importance[name] = gap["skill_importance"]

            title = emp_row["current_title"]
            headline = emp_row["headline"]

            return {
                "manipal_employee": emp_row["full_name"] or "Unknown",
                "employee_id": employee_id,
                "run_id": run_id,
                "role": title or "Professional",
                "company": emp_row["company_name"] or "Organization",
                "location": emp_row["location"] or "",
                "headline": headline or "",
                "existing_skills": existing_skills,
                "missing_skills": missing_skills,
                "skill_importance": skill_importance,
                "seniority": _infer_seniority(title),
                "domain": _infer_domain(title, headline),
            }
        except Exception as exc:
            logger.error("Failed to fetch employee %s: %s", employee_id, exc)
            return None


# ---------------------------------------------------------------------------
# Course writer — POST to SyncFlow, poll for course_id, persist locally
# ---------------------------------------------------------------------------

class DatabaseCourseWriter:
    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    async def save_course(
        self, employee_id: str, course_data: Dict[str, Any]
    ) -> Optional[str]:
        try:
            emp_uuid = uuid.UUID(employee_id)
            course_name = course_data.get("courseName", "Untitled Course")
            raw_json = json.dumps(course_data, ensure_ascii=False)

            # 1. POST to SyncFlow
            print(f"\n📤 Sending to SyncFlow API...")
            print(f"   URL: {cfg.syncflow_url}")
            print(f"   Course: {course_name}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    cfg.syncflow_url,
                    json=course_data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    status = resp.status
                    text = await resp.text()
                    print(f"   ✅ HTTP {status}: {text[:200]}")
                    if status not in (200, 201):
                        print(f"   ❌ SyncFlow failed: {status}")
                        return None

            # 2. Poll for the course_id in the mynoted table
            print(f"🔍 Polling database for course_id...")
            print(f"   Table: {cfg.course_table}")
            print(f"   Looking for: {course_name}")
            course_id = await self._poll_for_course_id(course_name)
            if not course_id:
                print(f"   ❌ Timeout waiting for course_id")
                return None
            print(f"   ✅ Got course_id: {course_id}")

            # 3. Persist in spectre.employee_courses
            print(f"💾 Saving to spectre.employee_courses...")
            conn = await self._pool.acquire()
            try:
                result = await conn.fetchrow(
                    """
                    INSERT INTO spectre.employee_courses
                        (employee_id, course_id, course_name, raw_json, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (employee_id, course_id) DO UPDATE SET
                    course_name = EXCLUDED.course_name,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = NOW()
                    RETURNING course_id
                    """,
                    emp_uuid,
                    uuid.UUID(course_id) if isinstance(course_id, str) else course_id,
                    course_name,
                    raw_json,
                    datetime.utcnow(),
                )
                if result:
                    print(f"   ✅ Saved to employee_courses: {course_name}")
                    return str(course_id)
                return None
            finally:
                await self._pool.release(conn)

        except Exception as exc:
            print(f"❌ Save error: {exc}")
            logger.error("save_course failed: %s", exc, exc_info=True)
            return None

    async def _poll_for_course_id(self, course_name: str) -> Optional[str]:
        sql = (
            f"SELECT {cfg.course_id_col} AS id, {cfg.course_created_col} AS created_at "
            f"FROM {cfg.course_table} "
            f"WHERE {cfg.course_name_col} = $1 "
            f"ORDER BY {cfg.course_created_col} DESC LIMIT 1"
        )
        deadline = time.time() + cfg.poll_timeout
        attempts = 0

        print(f"   Timeout: {cfg.poll_timeout}s, Interval: {cfg.poll_interval}s")

        while time.time() < deadline:
            attempts += 1
            conn = await self._pool.acquire()
            try:
                row = await conn.fetchrow(sql, course_name)
                if row:
                    print(f"   ✅ Found after {attempts} checks (~{attempts * cfg.poll_interval}s)")
                    return str(row["id"])
                remaining = int(deadline - time.time())
                print(f"   ⏳ Check #{attempts}: Not found yet... ({remaining}s remaining)")
                await asyncio.sleep(cfg.poll_interval)
            except Exception as exc:
                print(f"   ⚠️  DB query error: {exc}")
                await asyncio.sleep(cfg.poll_interval)
            finally:
                await self._pool.release(conn)

        print(f"   ❌ Timeout after {attempts} checks (~{attempts * cfg.poll_interval}s)")
        return None


# ---------------------------------------------------------------------------
# Azure OpenAI helper
# ---------------------------------------------------------------------------

async def _azure_chat(
    session: ClientSession,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1200,
    cost_tracker: Optional[CostTracker] = None,
) -> str:
    url = (
        f"{cfg.azure_endpoint}/openai/deployments/{cfg.azure_deployment_id}"
        f"/chat/completions?api-version={cfg.azure_api_version}"
    )
    headers = {"Content-Type": "application/json", "api-key": cfg.azure_api_key}
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    async def _post():
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Azure OpenAI error {resp.status}")
            data = await resp.json()
            # Track token usage
            if cost_tracker:
                usage = data.get("usage", {})
                await cost_tracker.record_llm(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return await _with_backoff(_post)


# ---------------------------------------------------------------------------
# Profile generator
# ---------------------------------------------------------------------------

class ProfileGenerator:
    def __init__(self, cost_tracker: Optional[CostTracker] = None) -> None:
        self._cache: Dict[str, str] = {}
        self._client: Any = None
        self._cost_tracker = cost_tracker
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=cfg.azure_api_key,
                api_version=cfg.azure_api_version,
                azure_endpoint=cfg.azure_endpoint,
            )
        except Exception as exc:
            logger.warning("Could not init sync AzureOpenAI client: %s", exc)

    async def generate(self, employee_data: Dict[str, Any]) -> str:
        name = employee_data.get("manipal_employee", "")
        if name in self._cache:
            return self._cache[name]

        if not self._client:
            return self._fallback(employee_data)

        try:
            clean = {
                "name": name,
                "current_position": employee_data.get("role"),
                "current_company": employee_data.get("company"),
                "existing_skills": employee_data.get("existing_skills", []),
                "target_skills": employee_data.get("missing_skills", []),
            }
            resp = self._client.chat.completions.create(
                model=cfg.azure_deployment_id,
                messages=[
                    {"role": "system", "content": "Create ultra-concise career profile. 2-3 paragraphs."},
                    {"role": "user", "content": f"Create profile:\n{json.dumps(clean, indent=2)}"},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            # Track token usage from sync client
            if self._cost_tracker and resp.usage:
                self._cost_tracker.record_llm_sync(
                    resp.usage.prompt_tokens or 0,
                    resp.usage.completion_tokens or 0,
                )
            profile = resp.choices[0].message.content.strip()
            self._cache[name] = profile
            return profile
        except Exception:
            return self._fallback(employee_data)

    @staticmethod
    def _fallback(data: Dict[str, Any]) -> str:
        name = data.get("manipal_employee", "Professional")
        role = data.get("role", "")
        skills = data.get("existing_skills", [])
        if skills:
            return f"{name} is a {role} skilled in {', '.join(skills[:5])}."
        return f"{name} is a {role} ready to develop new capabilities."


# ---------------------------------------------------------------------------
# SSR Foundation Agent — web search → scrape → summarise → Web-of-Truth
# ---------------------------------------------------------------------------

class SSRFoundationAgent:
    def __init__(self, session: ClientSession, cost_tracker: Optional[CostTracker] = None) -> None:
        self._session = session
        self._cost_tracker = cost_tracker

    async def build_web_of_truth(self, topic: str, profile: Dict[str, Any]) -> str:
        queries = await self._generate_queries(topic, profile)
        urls: List[str] = []
        for q in queries:
            try:
                urls.extend(await self._google_search(q))
            except Exception:
                pass
            await _sleep_ms(cfg.delay_between_searches_ms)

        if not urls:
            return "NO_URLS"

        summaries: List[str] = []
        for url in urls[:15]:
            try:
                content = await self._fetch_and_clean(url)
                if content:
                    summary = await self._summarise(content, topic, profile)
                    if summary:
                        summaries.append(summary)
            except Exception:
                pass
            await _sleep_ms(300)

        if not summaries:
            return "NO_SUMMARIES"

        combined = "\n\n".join(summaries)[:12_000]
        return await _azure_chat(
            self._session,
            [
                {"role": "system", "content": f"Create Web of Truth for {topic}. 300-400 words."},
                {"role": "user", "content": f"Skill: {topic}\n\n{combined}"},
            ],
            temperature=0.2,
            max_tokens=1000,
            cost_tracker=self._cost_tracker,
        )

    # -- private helpers ------------------------------------------------------

    async def _google_search(self, query: str) -> List[str]:
        encoded = aiohttp.helpers.quote(query, safe="")
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={cfg.google_api_key}&cx={cfg.google_cx}&q={encoded}"
        )

        async def _get():
            async with self._session.get(url, timeout=15) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"CSE error {resp.status}")
                data = await resp.json()
                return [
                    item["link"]
                    for item in data.get("items", [])[:cfg.websites_per_query]
                    if item.get("link")
                ]

        result = await _with_backoff(_get)
        if self._cost_tracker:
            await self._cost_tracker.record_google_search()
        return result

    async def _fetch_and_clean(self, url: str) -> Optional[str]:
        async def _get():
            async with self._session.get(
                url, headers={"User-Agent": "SpectreSpider/1.0"}, timeout=15
            ) as resp:
                if resp.status >= 400:
                    raise RuntimeError("Fetch error")
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type:
                    return None
                html = await resp.text(errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
                    tag.decompose()
                node = soup.select_one("main") or soup.select_one("article") or soup.select_one("body")
                text = node.get_text(" ") if node else soup.get_text(" ")
                text = re.sub(r"\s+", " ", text).strip()
                return text[:15_000] if len(text) >= 300 else None

        result = await _with_backoff(_get)
        if self._cost_tracker and result is not None:
            await self._cost_tracker.record_web_fetch()
        return result

    async def _summarise(self, content: str, topic: str, profile: Dict[str, Any]) -> Optional[str]:
        role = profile.get("role", "")
        out = await _azure_chat(
            self._session,
            [
                {"role": "system", "content": f"Extract 150-200 word summary for {role} learning {topic}."},
                {"role": "user", "content": f"Topic: {topic}\nContent:\n{content[:10_000]}"},
            ],
            temperature=0.2,
            max_tokens=600,
            cost_tracker=self._cost_tracker,
        )
        if re.search(r"\bIRRELEVANT\b", out, re.IGNORECASE):
            return None
        return out.strip() or None

    async def _generate_queries(self, topic: str, profile: Dict[str, Any]) -> List[str]:
        role = profile.get("role", "")
        out = await _azure_chat(
            self._session,
            [
                {"role": "system", "content": f"Generate 4-5 queries for {role} learning {topic}."},
                {"role": "user", "content": f"Skill: {topic}\nRole: {role}"},
            ],
            temperature=0.2,
            max_tokens=300,
            cost_tracker=self._cost_tracker,
        )
        queries = [q.strip() for q in out.split("\n") if q.strip()]
        return queries[:5] if queries else [f"{topic} tutorial"]


# ---------------------------------------------------------------------------
# Spider King — turns Web-of-Truth into a structured course
# ---------------------------------------------------------------------------

class SpiderKing:
    def __init__(self, session: ClientSession, cost_tracker: Optional[CostTracker] = None) -> None:
        self._session = session
        self._cost_tracker = cost_tracker

    async def create_course(
        self, web_of_truth: str, skill: str, profile: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        role = profile.get("role", "")
        system_prompt = (
            f'Create ULTRA-SHORT course for {role}. MAX {cfg.max_subtopics_per_skill} topics.\n'
            f'JSON: {{"courseName": "...", "modules": [{{"moduleName": "...", '
            f'"topics": [{{"name": "...", "description": "...", "aiActivity": "...", '
            f'"duration": "30-60 min"}}]}}]}}'
        )
        out = await _azure_chat(
            self._session,
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Skill: {skill}\nRole: {role}\n\n"
                        f"Web of Truth:\n{web_of_truth}\n\n"
                        f"MAX {cfg.max_subtopics_per_skill} topics."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1500,
            cost_tracker=self._cost_tracker,
        )
        return self._parse_course_json(out)

    def merge_skill_courses(
        self, courses: List[Dict[str, Any]], profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        all_topics: List[Dict[str, Any]] = []
        for course in courses:
            if not course or not course.get("modules"):
                continue
            for module in course["modules"]:
                for topic in module.get("topics", []):
                    if len(all_topics) >= cfg.max_total_subtopics:
                        break
                    all_topics.append({
                        "skillArea": module.get("moduleName", "Module"),
                        "name": topic.get("name"),
                        "description": topic.get("description"),
                        "aiActivity": topic.get("aiActivity"),
                        "duration": topic.get("duration", "45 min"),
                    })

        chunk = max(1, len(all_topics) // (3 if len(all_topics) > 15 else 2))
        modules = []
        for i in range(0, len(all_topics), chunk):
            batch = all_topics[i : i + chunk]
            if batch:
                modules.append({
                    "moduleName": f"Module {len(modules) + 1}: {batch[0]['skillArea']}",
                    "topics": [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "aiActivity": t["aiActivity"],
                            "duration": t["duration"],
                        }
                        for t in batch
                    ],
                })

        name = profile.get("name", "Learner")
        role = profile.get("role", "Professional")
        return {"courseName": f"{role} Fast-Track for {name}", "modules": modules[:3]}

    # -- private --------------------------------------------------------------

    @staticmethod
    def _parse_course_json(raw: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if obj.get("courseName") and isinstance(obj.get("modules"), list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None


# ---------------------------------------------------------------------------
# Progress tracker (console output)
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._lock = asyncio.Lock()

    async def skill_done(self, employee: str, skill: str, success: bool) -> None:
        async with self._lock:
            print(f"   {'✅' if success else '❌'} {skill}")

    async def employee_done(self, employee: str, skills_processed: int) -> None:
        async with self._lock:
            self._completed += 1
            print(f"🎓 {employee}: {skills_processed} modules ({self._completed}/{self._total})")


# ---------------------------------------------------------------------------
# Core processor — orchestrates the pipeline per employee
# ---------------------------------------------------------------------------

class CourseProcessor:
    def __init__(
        self,
        session: ClientSession,
        progress: ProgressTracker,
        profiles: ProfileGenerator,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self._session = session
        self._ssr = SSRFoundationAgent(session, cost_tracker)
        self._spider = SpiderKing(session, cost_tracker)
        self._progress = progress
        self._profiles = profiles
        self._cost_tracker = cost_tracker

    async def process_employee(
        self, data: Dict[str, Any], index: int, total: int
    ) -> Dict[str, Any]:
        name = data.get("manipal_employee", "Unknown")
        employee_id = data.get("employee_id")
        run_id = data.get("run_id")
        role = data.get("role", "Unknown")
        company = data.get("company", "Organization")

        base = {
            "employeeName": name,
            "employeeId": employee_id,
            "runId": run_id,
            "role": role,
            "company": company,
        }

        try:
            print(f"\n{'='*60}")
            print(f"👤 Processing {index+1}/{total}: {name}")
            print(f"{'='*60}")
            profile_text = await self._profiles.generate(data)

            missing_raw = data.get("missing_skills", [])
            if not missing_raw:
                print(f"   ℹ️  No skill gaps found")
                return {**base, "processingStatus": "complete", "reason": "No skill gaps"}

            critical = await self._select_top_skills(data, profile_text)
            if not critical:
                cats = _categorise_missing_skills(missing_raw, data.get("skill_importance", {}))
                critical = (
                    cats["critical"][:cfg.max_critical_skills]
                    or cats["important"][:cfg.max_critical_skills]
                    or cats["nice_to_have"][:cfg.max_critical_skills]
                )
            if not critical:
                print(f"   ℹ️  No prioritised skills")
                return {**base, "processingStatus": "complete", "reason": "No prioritised skills"}

            print(f"🎯 Focus: {len(critical)} skills: {', '.join(critical)}")

            profile = {
                "name": name,
                "role": role,
                "seniority": data.get("seniority", "Mid-level"),
                "company": company,
                "domain": data.get("domain", "Technology"),
                "generated_profile": profile_text,
                "existing_skills": data.get("existing_skills", []),
                "time_budget_hours_per_week": 4,
                "deadline_days": 30,
                "ai_tools": ["ChatGPT"],
            }

            skill_courses = await self._process_skills(critical, profile)
            merged = (
                self._spider.merge_skill_courses(skill_courses, profile)
                if skill_courses
                else {"courseName": f"Learning Path for {name}", "modules": []}
            )
            course = _to_syncflow_format(merged, "Spectre", date.today().isoformat())

            await self._progress.employee_done(name, len(skill_courses))
            return {**base, "course": course, "processingStatus": "complete"}

        except Exception as exc:
            print(f"❌ Error processing {name}: {exc}")
            logger.error("Error processing %s: %s", name, exc, exc_info=True)
            return {**base, "error": str(exc), "processingStatus": "failed"}

    # -- private helpers ------------------------------------------------------

    async def _select_top_skills(
        self, data: Dict[str, Any], profile_text: str
    ) -> List[str]:
        missing = data.get("missing_skills", [])
        if not missing:
            return []

        payload = {
            "employee_name": data.get("manipal_employee"),
            "role": data.get("role"),
            "existing_skills": data.get("existing_skills", []),
            "missing_skills": missing,
            "skill_importance": data.get("skill_importance", {}),
        }
        try:
            raw = await _azure_chat(
                self._session,
                [
                    {
                        "role": "system",
                        "content": 'Pick TOP 3 skill gaps. JSON: {"skills": [{"name": "...", "reason": "..."}]}',
                    },
                    {"role": "user", "content": json.dumps(payload, indent=2) + "\n\nPick 3 skills."},
                ],
                temperature=0.2,
                max_tokens=600,
                cost_tracker=self._cost_tracker,
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return []
            obj = json.loads(match.group(0))
            selected: List[str] = []
            for item in obj.get("skills", []):
                name = item.get("name")
                if name and name in missing:
                    selected.append(_normalise_skill(name))
                if len(selected) >= cfg.max_critical_skills:
                    break
            return selected
        except Exception:
            return []

    async def _process_skills(
        self, skills: List[str], profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(cfg.max_concurrent_skills)

        async def _one(skill: str) -> Optional[Dict[str, Any]]:
            async with sem:
                try:
                    wot = await self._ssr.build_web_of_truth(skill, profile)
                    if wot in ("NO_URLS", "NO_SUMMARIES"):
                        await self._progress.skill_done(profile["name"], skill, False)
                        return None
                    course = await self._spider.create_course(wot, skill, profile)
                    await self._progress.skill_done(profile["name"], skill, course is not None)
                    return course
                except Exception:
                    await self._progress.skill_done(profile["name"], skill, False)
                    return None

        results = await asyncio.gather(*(_one(s) for s in skills), return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def generate_courses(
    *,
    run_id: Optional[str] = None,
    employee_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a course for the PRIMARY employee.

    Args:
        run_id: UUID of the run → finds the primary employee in that run.
        employee_id: UUID of the employee → finds the run where they are primary.
        Either one is sufficient. Both can be provided.
    """
    if not run_id and not employee_id:
        raise ValueError("Provide at least one of run_id or employee_id")

    print("🚀 SPECTRE Course Builder (Database)")
    print("=" * 60)
    await db_pool.connect()
    cost_tracker = CostTracker()

    try:
        # Stage 1: resolve & fetch the primary employee
        print(f"\n📊 STAGE 1: Fetching from Database")
        if run_id:
            print(f"   Run ID: {run_id}")
        if employee_id:
            print(f"   Employee ID: {employee_id}")

        fetcher = DatabaseDataFetcher(db_pool)
        emp = await fetcher.fetch_primary(run_id=run_id, employee_id=employee_id)

        if not emp:
            print("❌ No primary employee found")
            return {"success": False, "error": "No primary employee found", "courses_generated": 0, "cost": cost_tracker.summary()}

        print(f"✅ Target: {emp.get('manipal_employee')} ({emp.get('role')})")
        print(f"   Employee: {emp.get('employee_id')}")
        print(f"   Run: {emp.get('run_id')}")

        # Stage 2: generate course
        print(f"\n🎯 STAGE 2: Course Generation")
        print("=" * 60)
        progress = ProgressTracker(1)
        profiles = ProfileGenerator(cost_tracker)

        async with aiohttp.ClientSession() as session:
            processor = CourseProcessor(session, progress, profiles, cost_tracker)
            t0 = time.time()
            result = await processor.process_employee(emp, 0, 1)
            elapsed = time.time() - t0

        print(f"\n⚡ Generation completed in {elapsed:.1f}s")

        # Stage 3: persist course
        print(f"\n💾 STAGE 3: Saving to Database")
        saved: List[Dict[str, Any]] = []
        if isinstance(result, dict) and result.get("processingStatus") == "complete":
            eid = result.get("employeeId")
            course_data = result.get("course")
            if eid and course_data:
                writer = DatabaseCourseWriter(db_pool)
                cid = await writer.save_course(eid, course_data)
                if cid:
                    saved.append({
                        "employeeId": eid,
                        "employeeName": result.get("employeeName"),
                        "courseId": cid,
                        "courseName": course_data.get("courseName"),
                    })

        # Summary
        cost = cost_tracker.summary()
        print(f"\n{'=' * 60}")
        print(f"📊 SUMMARY")
        print(f"   ✅ Saved: {len(saved)} course(s)")
        print(f"   💰 Cost: {cost['llm_calls']} LLM calls, "
              f"{cost['total_tokens']} tokens, "
              f"{cost['google_searches']} searches")
        print(f"   💵 Total: ${cost['total_cost_usd']:.4f} "
              f"(LLM: ${cost['llm_cost_usd']:.4f} + Google: ${cost['google_cost_usd']:.4f})")
        print(f"   ⏱  Elapsed: {cost['elapsed_seconds']}s")
        print(f"{'=' * 60}")

        return {
            "success": True,
            "courses_generated": len(saved),
            "saved_courses": saved,
            "cost": cost,
        }

    except Exception as exc:
        logger.error("❌ generate_courses failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "error": str(exc),
            "courses_generated": 0,
            "cost": cost_tracker.summary(),
        }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Manage DB pool across the server's lifetime."""
    await db_pool.connect()
    yield
    await db_pool.close()


# app = FastAPI(
#     title="SPECTRE Agent 5",
#     description="Hyper-personalised course generation from employee skill gaps",
#     version="5.2.0",
#     lifespan=_lifespan,
# )
router = APIRouter(prefix="/spectre", tags=["Spectre"])

class GenerateRequest(BaseModel):
    run_id: Optional[str] = None
    employee_id: Optional[str] = None


class CourseResponse(BaseModel):
    success: bool
    courses_generated: int = 0
    saved_courses: List[Dict[str, Any]] = []
    error: Optional[str] = None
    cost: Optional[Dict[str, Any]] = None


@router.post("/generate", response_model=CourseResponse)
async def api_generate_courses(
    body: Optional[GenerateRequest] = None,
    run_id: Optional[str] = Query(None, description="UUID of the run → finds primary employee"),
    employee_id: Optional[str] = Query(None, description="UUID of the employee → finds their primary run"),
):
    """Generate a course for the PRIMARY employee.

    Provide run_id, employee_id, or both. The system resolves the primary
    target from spectre.run_employees (role='primary').
    """
    rid = (body.run_id if body and body.run_id else None) or run_id
    eid = (body.employee_id if body and body.employee_id else None) or employee_id

    if not rid and not eid:
        raise HTTPException(status_code=400, detail="Provide run_id, employee_id, or both.")
    try:
        result = await generate_courses(run_id=rid, employee_id=eid)
        return CourseResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Unhandled error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point (kept for testing)
# ---------------------------------------------------------------------------

async def _cli_main() -> None:
    args = sys.argv[1:]
    run_id: Optional[str] = None
    employee_id: Optional[str] = None

    it = iter(args)
    for arg in it:
        if arg == "--run-id":
            run_id = next(it, None)
        elif arg == "--employee-id":
            employee_id = next(it, None)

    if not employee_id and not run_id:
        print(
            "Usage:\n"
            "  python spectre_agent5_db.py --serve [--host 0.0.0.0] [--port 8000]\n"
            "  python spectre_agent5_db.py --run-id <uuid>\n"
            "  python spectre_agent5_db.py --employee-id <uuid>\n"
            "  python spectre_agent5_db.py --run-id <uuid> --employee-id <uuid>\n"
            "\n"
            "  Either flag resolves to the PRIMARY employee via run_employees."
        )
        sys.exit(1)

    t0 = time.time()
    try:
        result = await generate_courses(run_id=run_id, employee_id=employee_id)
        if result.get("success"):
            print(f"\n✅ Done — {result['courses_generated']} course(s) generated")
            cost = result.get("cost", {})
            if cost:
                print(f"💰 Cost: {cost.get('llm_calls', 0)} LLM calls, "
                      f"{cost.get('total_tokens', 0)} tokens, "
                      f"{cost.get('google_searches', 0)} searches — "
                      f"${cost.get('total_cost_usd', 0):.4f}")
        else:
            print(f"\n❌ Failed: {result.get('error')}")
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        await db_pool.close()
    print(f"⏱  {math.ceil(time.time() - t0)}s")


def _parse_serve_args() -> tuple:
    """Parse --serve, --host, --port from sys.argv (sync, before event loop)."""
    host = "0.0.0.0"
    port = 8000
    it = iter(sys.argv[1:])
    for arg in it:
        if arg == "--host":
            host = next(it, "0.0.0.0")
        elif arg == "--port":
            port = int(next(it, "8000"))
    return host, port


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        host, port = _parse_serve_args()
        print(f"🚀 Starting SPECTRE API on {host}:{port}")
        print(f"   Docs: http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(_cli_main())