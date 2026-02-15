"""
Mutation 1 — Manual Peer Add
==============================
A run-scoped mutation that appends a new comparison peer into an existing
run_id workspace, then selectively recomputes only the downstream artifacts
impacted by the peer-set change.

Pipeline:
  1. Normalize URL + idempotency check
  2. SHADE primitives  → employee upsert + run link + employee_details
  3. CIPHER            → single-employee skill extraction for the new peer
  4. FRACTAL           → recompute target's gaps (full run re-analysis)
  5. Agent 5 (conditional) → course regeneration ONLY if critical gaps changed
  6. ATLAS             → scoped report refresh with incremented run_version

Endpoint:
  POST /mutation/add-peer

Usage:
  python mutation_1_add_peer.py --serve --port 8010
  curl -X POST http://localhost:8010/mutation/add-peer \
       -H "Content-Type: application/json" \
       -d '{"run_id":"...","peer_linkedin_url":"https://linkedin.com/in/..."}'

Requires co-located agent modules (adjust import names at top if needed):
  - shade_agent.py        (SHADE — Document 2)
  - agent3_db.py          (CIPHER — Document 3)
  - agent4_db_skill_gap.py (FRACTAL — Document 4)
  - spectre_agent5_db.py  (Spider — Document 5)
  - atlas_v2.py           (ATLAS — Document 6)
"""

from __future__ import annotations

import os
import re
import json
import uuid
import time
import logging
import asyncio
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# AGENT IMPORTS — matched to actual filenames in your project
# ═══════════════════════════════════════════════════════════════
import traceback

print("\n" + "=" * 60)
print("🔌 LOADING AGENT MODULES...")
print("=" * 60)

# SHADE (Agent 1) — shade_1.py
SHADE_AVAILABLE = False
try:
    from shade_1 import (
        BrightDataScraper,
        BrightDataTriggerScraper,
        ComprehensiveDataManager,
        clean_bright_profile,
        get_db_connection as shade_get_db_connection,
        _guess_name_from_linkedin_url,
        RunCostTracker as ShadeRunCostTracker,
    )
    SHADE_AVAILABLE = True
    print("  ✅ SHADE (shade_1) loaded")
except Exception as e:
    print(f"  ❌ SHADE (shade_1) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# CIPHER (Agent 3) — cipher_3.py
CIPHER_AVAILABLE = False
try:
    from cipher_3 import SkillExtractor
    CIPHER_AVAILABLE = True
    print("  ✅ CIPHER (cipher_3) loaded")
except Exception as e:
    print(f"  ❌ CIPHER (cipher_3) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# FRACTAL (Agent 4) — fractal_4.py
FRACTAL_AVAILABLE = False
try:
    from fractal_4 import run_agent4, DB_CONFIG as FRACTAL_DB_CONFIG
    FRACTAL_AVAILABLE = True
    print("  ✅ FRACTAL (fractal_4) loaded")
except Exception as e:
    print(f"  ❌ FRACTAL (fractal_4) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# SPIDER (Agent 5) — spectre_spider_5.py
SPIDER_AVAILABLE = False
try:
    from spectre_spider_5 import generate_courses as spider_generate_courses
    SPIDER_AVAILABLE = True
    print("  ✅ SPIDER (spectre_spider_5) loaded")
except Exception as e:
    print(f"  ❌ SPIDER (spectre_spider_5) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# ATLAS (Agent 6) — atlas_6.py
ATLAS_AVAILABLE = False
try:
    from atlas_6 import run_atlas, SpectreDB as AtlasSpectreDB
    ATLAS_AVAILABLE = True
    print("  ✅ ATLAS (atlas_6) loaded")
except Exception as e:
    print(f"  ❌ ATLAS (atlas_6) FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── HARD FAIL CHECK — refuse to run without critical agents ──
_REQUIRED_AGENTS = {
    "SHADE": SHADE_AVAILABLE,
    "CIPHER": CIPHER_AVAILABLE,
    "FRACTAL": FRACTAL_AVAILABLE,
    "ATLAS": ATLAS_AVAILABLE,
}
_missing = [name for name, loaded in _REQUIRED_AGENTS.items() if not loaded]

print()
if _missing:
    print(f"🚨 MISSING REQUIRED AGENTS: {', '.join(_missing)}")
    print("   The mutation pipeline will REFUSE to run without these.")
    print("   Fix the import errors above, then restart.")
else:
    print("✅ All required agents loaded successfully!")
print("=" * 60 + "\n")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_DSN = (
    "postgresql://monsteradmin:M0nsteradmin"
    "@monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require"
)

DB_CONFIG = {
    "host": os.getenv("SPECTRE_DB_HOST", "monsterdb.postgres.database.azure.com"),
    "port": int(os.getenv("SPECTRE_DB_PORT", "5432")),
    "dbname": os.getenv("SPECTRE_DB_NAME", "postgres"),
    "user": os.getenv("SPECTRE_DB_USER", "monsteradmin"),
    "password": os.getenv("SPECTRE_DB_PASSWORD", "M0nsteradmin"),
    "sslmode": os.getenv("SPECTRE_DB_SSLMODE", "require"),
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - MUTATION1 - %(levelname)s - %(message)s",
)
log = logging.getLogger("mutation1")

# ── Module load confirmation ──────────────────────────────────
log.info("=" * 60)
log.info("📦 mutation_1_add_peer.py LOADED")
log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# COST TRACKER — aggregates across all agent calls in one mutation
# ═══════════════════════════════════════════════════════════════

GPT4O_INPUT_PER_1M = 2.50
GPT4O_OUTPUT_PER_1M = 10.00
BRIGHT_DATA_COST_PER_ROW = 0.01
GOOGLE_CSE_COST_PER_QUERY = 0.005


@dataclass
class MutationCostTracker:
    """Accumulates costs for the entire mutation pipeline."""

    gpt_input_tokens: int = 0
    gpt_output_tokens: int = 0
    gpt_calls: int = 0
    bright_data_rows: int = 0
    google_queries: int = 0
    started_at: float = field(default_factory=time.time)

    # Per-step breakdowns
    step_costs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_step(self, step_name: str, cost_data: Dict[str, Any]) -> None:
        """Merge cost info from an agent step."""
        self.step_costs[step_name] = cost_data

        # Try to extract GPT tokens from various cost formats
        gpt = cost_data.get("gpt", cost_data)
        self.gpt_input_tokens += int(gpt.get("input_tokens", 0) or gpt.get("prompt_tokens", 0) or gpt.get("inputTokens", 0) or 0)
        self.gpt_output_tokens += int(gpt.get("output_tokens", 0) or gpt.get("completion_tokens", 0) or gpt.get("outputTokens", 0) or 0)
        self.gpt_calls += int(gpt.get("calls", 0) or gpt.get("llm_calls", 0) or gpt.get("totalCalls", 0) or 0)
        self.bright_data_rows += int(cost_data.get("bright_data", {}).get("profiles_scraped", 0))
        self.google_queries += int(cost_data.get("google_cse", {}).get("queries", 0) or cost_data.get("google_searches", 0) or 0)

    def summary(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self.started_at, 2)
        input_cost = (self.gpt_input_tokens / 1_000_000) * GPT4O_INPUT_PER_1M
        output_cost = (self.gpt_output_tokens / 1_000_000) * GPT4O_OUTPUT_PER_1M
        gpt_total = input_cost + output_cost
        bright_cost = self.bright_data_rows * BRIGHT_DATA_COST_PER_ROW
        google_cost = self.google_queries * GOOGLE_CSE_COST_PER_QUERY
        total = gpt_total + bright_cost + google_cost

        return {
            "total_cost_usd": round(total, 6),
            "elapsed_seconds": elapsed,
            "gpt": {
                "calls": self.gpt_calls,
                "input_tokens": self.gpt_input_tokens,
                "output_tokens": self.gpt_output_tokens,
                "total_tokens": self.gpt_input_tokens + self.gpt_output_tokens,
                "cost_usd": round(gpt_total, 6),
            },
            "bright_data": {
                "profiles_scraped": self.bright_data_rows,
                "cost_usd": round(bright_cost, 6),
            },
            "google_cse": {
                "queries": self.google_queries,
                "cost_usd": round(google_cost, 6),
            },
            "step_breakdown": self.step_costs,
        }


# ═══════════════════════════════════════════════════════════════
# URL NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_linkedin_url(url: str) -> str:
    """
    Canonical form: 'linkedin.com/in/<slug>' (lowercase, no trailing numbers,
    no query params). Matches SHADE's ComprehensiveDataManager._normalize_linkedin_url.
    """
    if not url:
        return ""
    url = url.strip()
    if "linkedin.com" not in url:
        return url
    try:
        parts = [x for x in urlparse(url).path.split("/") if x]
        slug = parts[1] if len(parts) >= 2 and parts[0] == "in" else (parts[0] if parts else "")
        slug = unquote(slug).split("?")[0].split("#")[0].strip("/")
        slug = re.sub(r"-\d+$", "", slug)  # remove trailing numeric suffix
        return f"linkedin.com/in/{slug.lower()}" if slug else "linkedin.com"
    except Exception:
        return url


def guess_name_from_url(url: str) -> str:
    """Best-effort name from a LinkedIn URL slug. Strips trailing numeric/hex IDs."""
    try:
        parts = [x for x in urlparse(url).path.split("/") if x]
        slug = parts[1] if len(parts) >= 2 and parts[0] == "in" else (parts[0] if parts else "")
        slug = unquote(slug).split("?")[0].strip("-_/")
        # Remove trailing numeric/hex IDs (e.g., "-86b136111", "-12345")
        slug = re.sub(r"-[0-9a-f]{6,}$", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"-\d+$", "", slug)
        name = slug.replace("-", " ").replace("_", " ").strip()
        return " ".join(w.capitalize() for w in name.split()) if name else "(peer from URL)"
    except Exception:
        return "(peer from URL)"


# ═══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS (mutation-specific)
# ═══════════════════════════════════════════════════════════════

def _get_conn():
    """Get a psycopg2 connection using the shared DB config."""
    dsn = os.getenv("SPECTRE_DB_URL", DEFAULT_DSN)
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn


class MutationDB:
    """Handles all mutation-specific DB reads and writes."""

    def __init__(self, conn=None):
        self._conn = conn or _get_conn()

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = _get_conn()
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── Run validation ──────────────────────────────────────
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, scope, status, target_company_id, config_json "
                "FROM spectre.runs WHERE run_id = %s",
                (run_id,),
            )
            return cur.fetchone()

    def get_target_employee_for_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the TARGET employee in a run.
        Tries role_in_run in priority order: primary → seed → target → 
        then falls back to first employee in the run (sorted by created_at ASC,
        so the original/first-added employee is picked).
        """
        # Priority order of roles that indicate "this is the target person"
        target_roles = ["primary", "seed", "target"]

        with self.conn.cursor() as cur:
            # First: check what roles exist in this run
            cur.execute("""
                SELECT DISTINCT re.role_in_run::text AS role
                FROM spectre.run_employees re
                WHERE re.run_id = %s
            """, (run_id,))
            existing_roles = [r["role"] for r in cur.fetchall()]
            log.info(f"  Roles found in run: {existing_roles}")

            # Try each target role in priority order
            for role in target_roles:
                if role in existing_roles:
                    cur.execute("""
                        SELECT re.employee_id, re.role_in_run::text AS role_in_run,
                               e.full_name, e.current_title,
                               e.linkedin_url, re.source_company_id
                        FROM spectre.run_employees re
                        JOIN spectre.employees e ON e.employee_id = re.employee_id
                        WHERE re.run_id = %s AND re.role_in_run::text = %s
                        ORDER BY re.created_at ASC
                        LIMIT 1
                    """, (run_id, role))
                    row = cur.fetchone()
                    if row:
                        log.info(f"  Found target via role='{role}': {row['full_name']}")
                        return dict(row)

            # Fallback: just take the first employee added to the run
            log.warning(f"  No standard target role found. Using first employee in run.")
            cur.execute("""
                SELECT re.employee_id, re.role_in_run::text AS role_in_run,
                       e.full_name, e.current_title,
                       e.linkedin_url, re.source_company_id
                FROM spectre.run_employees re
                JOIN spectre.employees e ON e.employee_id = re.employee_id
                WHERE re.run_id = %s
                ORDER BY re.created_at ASC
                LIMIT 1
            """, (run_id,))
            row = cur.fetchone()
            if row:
                log.info(f"  Fallback target: {row['full_name']} (role={row['role_in_run']})")
                return dict(row)

            return None

    # ── Idempotency check ───────────────────────────────────
    def is_peer_already_in_run(self, run_id: str, canonical_url: str) -> Optional[str]:
        """
        Check if a LinkedIn URL is already linked to this run.
        Returns employee_id if found, None otherwise.
        """
        with self.conn.cursor() as cur:
            # Match on canonical form — strip to slug for flexibility
            cur.execute("""
                SELECT e.employee_id
                FROM spectre.run_employees re
                JOIN spectre.employees e ON e.employee_id = re.employee_id
                WHERE re.run_id = %s
                  AND LOWER(e.linkedin_url) LIKE %s
            """, (run_id, f"%{canonical_url}%"))
            row = cur.fetchone()
            return str(row["employee_id"]) if row else None

    # ── Enum-safe role resolution ───────────────────────────
    def resolve_role_in_run(self, preferred: str = "matched") -> str:
        """Pick a valid employee_role_in_run enum value for a PEER (never primary/target)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT e.enumlabel FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE n.nspname = 'spectre' AND t.typname = 'employee_role_in_run'
                ORDER BY e.enumsortorder
            """)
            allowed = [r["enumlabel"] for r in cur.fetchall()]

        log.info(f"  Allowed role_in_run values: {allowed}")

        if not allowed:
            log.error("  No enum values found!")
            return "matched"

        # Roles that must NEVER be assigned to a manually-added peer
        forbidden = {"primary", "target"}

        # Try preferred, then safe fallbacks matching YOUR actual enum
        for candidate in [preferred, "matched", "competitor_candidate", "batch_member"]:
            if candidate in allowed and candidate not in forbidden:
                log.info(f"  Selected role: '{candidate}'")
                return candidate

        # Last resort: any non-forbidden role
        for val in allowed:
            if val not in forbidden:
                log.info(f"  Last-resort role: '{val}'")
                return val

        log.error(f"  All enum values are forbidden: {allowed}. Forcing 'matched'.")
        return "matched"

    # ── Employee upsert (SHADE-style) ───────────────────────
    def upsert_peer_employee(
        self,
        linkedin_url: str,
        full_name: str,
        current_title: Optional[str],
        company_id: Optional[str],
        profile_cache_text: Optional[str] = None,
        location: Optional[str] = None,
    ) -> str:
        """Insert or update employee in spectre.employees. Returns employee_id."""
        new_id = str(uuid.uuid4())
        meta = json.dumps({
            "mutation1": {
                "source": "mutation_1_add_peer",
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        })
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employees (
                    employee_id, full_name, linkedin_url,
                    current_company_id, current_title, location,
                    metadata_json, profile_cache_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (linkedin_url) DO UPDATE SET
                    full_name = COALESCE(EXCLUDED.full_name, spectre.employees.full_name),
                    current_company_id = COALESCE(EXCLUDED.current_company_id, spectre.employees.current_company_id),
                    current_title = COALESCE(EXCLUDED.current_title, spectre.employees.current_title),
                    location = COALESCE(EXCLUDED.location, spectre.employees.location),
                    metadata_json = COALESCE(spectre.employees.metadata_json, '{}'::jsonb)
                                    || COALESCE(EXCLUDED.metadata_json, '{}'::jsonb),
                    profile_cache_text = COALESCE(EXCLUDED.profile_cache_text, spectre.employees.profile_cache_text),
                    updated_at = NOW()
                RETURNING employee_id
            """, (
                new_id, full_name, linkedin_url,
                company_id, current_title, location,
                meta, profile_cache_text,
            ))
            return str(cur.fetchone()["employee_id"])

    # ── Link employee to run ────────────────────────────────
    def link_employee_to_run(
        self,
        run_id: str,
        employee_id: str,
        role_in_run: str,
        source_company_id: Optional[str] = None,
    ) -> bool:
        """Insert into spectre.run_employees with ON CONFLICT DO NOTHING."""
        meta = json.dumps({
            "source": "mutation_1_add_peer",
            "manual_add": True,
            "added_at": datetime.now(timezone.utc).isoformat(),
        })
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.run_employees
                    (run_id, employee_id, role_in_run, source_company_id, raw_json)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id, employee_id) DO NOTHING
            """, (run_id, employee_id, role_in_run, source_company_id, meta))
            return cur.rowcount > 0

    # ── Insert into employee_matches (ATLAS reads this!) ──────
    def insert_employee_match(
        self,
        run_id: str,
        target_employee_id: str,
        peer_employee_id: str,
        peer_name: str,
        peer_title: str,
        peer_company: str,
        peer_company_id: str = "",
        match_score: float = 90.0,
    ) -> bool:
        """
        Insert into spectre.employee_matches so ATLAS can find the peer.
        ATLAS Step 2 queries employee_matches — NOT run_employees.
        Without this row, the new peer is invisible to ATLAS.
        """
        rationale = json.dumps({
            "source": "mutation_1_add_peer",
            "method": "manual_peer_add",
            "added_at": datetime.now(timezone.utc).isoformat(),
        })

        # Resolve a valid match_type from the enum
        match_type = "manual"  # fallback
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT enumlabel FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE n.nspname = 'spectre' AND t.typname = 'employee_match_type'
                ORDER BY e.enumsortorder
            """)
            allowed = [r["enumlabel"] for r in cur.fetchall()]
            log.info(f"  Allowed match_type values: {allowed}")

            # Pick best available
            for preferred in ["manual", "linkedin_search", "google_search", "discovered"]:
                if preferred in allowed:
                    match_type = preferred
                    break
            else:
                if allowed:
                    match_type = allowed[0]
            log.info(f"  Using match_type: '{match_type}'")

        # Ensure we have a valid company_id (column is NOT NULL)
        if not peer_company_id and peer_company:
            cid = self.upsert_company(peer_company)
            if cid:
                peer_company_id = cid
        if not peer_company_id:
            log.warning(f"  ⚠️  No company_id — creating placeholder")
            peer_company_id = self.upsert_company(peer_company or "Unknown") or str(uuid.uuid4())

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_matches
                    (run_id, employee_id, matched_employee_id,
                     matched_name, matched_title, matched_company_id,
                     matched_company_name,
                     match_score, match_type, raw_json, rationale_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT DO NOTHING
            """, (
                run_id, target_employee_id, peer_employee_id,
                peer_name, peer_title or "", peer_company_id,
                peer_company or "",
                match_score, match_type, rationale, rationale,
            ))
            inserted = cur.rowcount > 0
        self.conn.commit()
        return inserted

    # ── Employee details (SHADE-style via ComprehensiveDataManager) ──
    def insert_employee_details_from_bright(
        self,
        run_id: str,
        employee_id: str,
        bright_profile: Dict[str, Any],
    ) -> None:
        """
        Clean a Bright Data profile and insert into spectre.employee_details.
        Reuses SHADE's clean_bright_profile + GPT structuring if SHADE is available.
        """
        if SHADE_AVAILABLE:
            # Use SHADE's ComprehensiveDataManager for GPT-enriched details
            cdm = ComprehensiveDataManager()
            basic_info = {
                "name": bright_profile.get("name", ""),
                "linkedin_url": bright_profile.get("url", ""),
                "company": bright_profile.get("current_company_name", ""),
                "search_snippet": bright_profile.get("position", ""),
                "is_seed": False,
            }
            summary = {
                "full_name": bright_profile.get("name", ""),
                "current_position": bright_profile.get("position", ""),
                "location": bright_profile.get("location", ""),
            }
            try:
                cdm._insert_employee_details(
                    self.conn.cursor(), run_id, employee_id,
                    bright_profile, summary, basic_info,
                )
                return
            except Exception as e:
                log.warning(f"SHADE _insert_employee_details failed, using fallback: {e}")
                self.conn.rollback()

        # Fallback: minimal insert without GPT enrichment
        cleaned = {}
        try:
            if SHADE_AVAILABLE:
                cleaned = clean_bright_profile(bright_profile)
            else:
                cleaned = {"raw_profile": bright_profile, "vitals": {}}
        except Exception:
            cleaned = {"raw_profile": bright_profile, "vitals": {}}

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_details (
                    run_id, employee_id, data_origin, details_json, raw_json,
                    created_by_agent, model_name,
                    v_name, v_company, v_title, v_position, v_location
                ) VALUES (
                    %s, %s, 'shade', %s::jsonb, %s::jsonb,
                    'mutation_1', 'gpt-4o',
                    %s, %s, %s, %s, %s
                )
            """, (
                run_id, employee_id,
                json.dumps(cleaned), json.dumps(bright_profile),
                bright_profile.get("name", ""),
                bright_profile.get("current_company_name", ""),
                bright_profile.get("position", ""),
                bright_profile.get("position", ""),
                bright_profile.get("location", ""),
            ))

    # ── Upsert company (minimal) ────────────────────────────
    def upsert_company(self, company_name: str) -> Optional[str]:
        """Ensure company exists; return company_id."""
        if not company_name or not company_name.strip():
            return None
        new_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.companies (company_id, name)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                RETURNING company_id
            """, (new_id, company_name.strip()))
            row = cur.fetchone()
            return str(row["company_id"]) if row else None

    # ── Critical gaps snapshot ──────────────────────────────
    def get_critical_gaps(self, run_id: str, employee_id: str) -> List[str]:
        """
        Return sorted list of critical/important skill gap names.
        Used for before/after comparison to decide if Agent 5 should run.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT skill_gap_name
                FROM spectre.employee_skill_gaps
                WHERE run_id = %s AND employee_id = %s
                  AND skill_importance IN ('Critical', 'Important', 'critical', 'important')
                ORDER BY skill_gap_name
            """, (run_id, employee_id))
            return sorted([r["skill_gap_name"] for r in cur.fetchall()])

    # ── Report versioning ───────────────────────────────────
    def get_current_report_version(self, run_id: str, employee_id: str) -> int:
        """Get the latest report_version (as int) for the atlas report."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT report_version
                FROM spectre.employee_reports
                WHERE run_id = %s AND employee_id = %s AND report_type = 'atlas'
                LIMIT 1
            """, (run_id, employee_id))
            row = cur.fetchone()
            if row and row["report_version"]:
                try:
                    return int(row["report_version"])
                except (ValueError, TypeError):
                    return 1
            return 0

    def stamp_report_version(
        self,
        run_id: str,
        employee_id: str,
        new_version: int,
        peer_name: str,
        peer_linkedin_url: str,
    ) -> None:
        """
        Update the atlas report's version metadata after ATLAS has written it.
        ATLAS writes the report first (via save_employee_report with ON CONFLICT
        UPDATE), then we stamp the mutation metadata on top.
        """
        mutation_summary = f"mutation_1:add_peer:{peer_name}:{peer_linkedin_url}"
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE spectre.employee_reports
                SET report_version      = %s,
                    created_by_mutation  = 'mutation_1_add_peer',
                    mutation_summary     = %s,
                    version_created_at   = %s
                WHERE run_id = %s AND employee_id = %s AND report_type = 'atlas'
            """, (
                str(new_version),
                mutation_summary,
                datetime.now(timezone.utc),
                run_id,
                employee_id,
            ))
        self.conn.commit()

    def archive_current_report(self, run_id: str, employee_id: str) -> Optional[int]:
        """
        BEFORE ATLAS overwrites: copy the current 'atlas' row to 'atlas_v{N}'.
        This preserves the previous version so the user can see history.
        Returns the archived version number, or None if no report existed.
        """
        with self.conn.cursor() as cur:
            # Read the current report
            cur.execute("""
                SELECT report_version, report_json, created_at, model_name,
                       created_by_agent, created_by_mutation, mutation_summary,
                       version_created_at
                FROM spectre.employee_reports
                WHERE run_id = %s AND employee_id = %s AND report_type = 'atlas'
            """, (run_id, employee_id))
            row = cur.fetchone()
            if not row:
                log.info("  No existing atlas report to archive")
                return None

            current_v = 0
            if row["report_version"]:
                try:
                    current_v = int(row["report_version"])
                except (ValueError, TypeError):
                    current_v = 0

            archive_type = f"atlas_v{current_v}"
            log.info(f"  📦 Archiving current report as report_type='{archive_type}'")

            # Insert the archive copy — report_json comes back as dict from
            # RealDictCursor+jsonb, so we must wrap it for re-insertion
            report_json_val = row["report_json"]
            if isinstance(report_json_val, dict) or isinstance(report_json_val, list):
                report_json_val = json.dumps(report_json_val)

            cur.execute("""
                INSERT INTO spectre.employee_reports (
                    run_id, employee_id, report_type, report_json,
                    created_at, model_name, created_by_agent,
                    report_version, created_by_mutation, mutation_summary,
                    version_created_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, employee_id, report_type) DO NOTHING
            """, (
                run_id, employee_id, archive_type, report_json_val,
                row["created_at"], row["model_name"], row["created_by_agent"],
                str(current_v), row["created_by_mutation"], row["mutation_summary"],
                row["version_created_at"] or row["created_at"],
            ))
            self.conn.commit()
            log.info(f"  ✅ Archived as '{archive_type}' (rowcount={cur.rowcount})")
            return current_v

    def verify_peer_in_employee_details(self, run_id: str, employee_id: str) -> bool:
        """Check if a peer has an entry in employee_details (required by ATLAS)."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS cnt
                FROM spectre.employee_details
                WHERE run_id = %s AND employee_id = %s
            """, (run_id, employee_id))
            row = cur.fetchone()
            return row["cnt"] > 0 if row else False

    def ensure_minimal_employee_details(self, run_id: str, employee_id: str,
                                         full_name: str, title: Optional[str],
                                         linkedin_url: str) -> None:
        """
        Insert a minimal employee_details row if none exists.
        ATLAS requires this row to include the peer in its report.
        """
        if self.verify_peer_in_employee_details(run_id, employee_id):
            log.info(f"  ✅ employee_details already exists for {employee_id}")
            return

        log.info(f"  ⚠️  No employee_details found — inserting minimal row for ATLAS")
        minimal_details = json.dumps({
            "source": "mutation_1_minimal",
            "full_name": full_name,
            "current_title": title or "",
            "linkedin_url": linkedin_url,
        })
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_details (
                    run_id, employee_id, data_origin, details_json,
                    created_by_agent, model_name,
                    v_name, v_title, v_position
                ) VALUES (%s, %s, 'mutation', %s::jsonb, 'mutation_1', 'none', %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                run_id, employee_id, minimal_details,
                full_name, title or "", title or "",
            ))
        self.conn.commit()
        log.info(f"  ✅ Minimal employee_details inserted")

    def get_peer_count(self, run_id: str) -> int:
        """Count non-target employees in the run."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS cnt
                FROM spectre.run_employees
                WHERE run_id = %s AND role_in_run::text NOT IN ('primary', 'seed', 'target')
            """, (run_id,))
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0

    # ── Course versioning ────────────────────────────────────
    def get_next_course_version(self, employee_id: str) -> int:
        """Get the next course_version for this employee."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(course_version), -1) + 1 AS next_v
                FROM spectre.employee_courses
                WHERE employee_id = %s
            """, (employee_id,))
            row = cur.fetchone()
            return row["next_v"] if row else 0

    def save_course_versioned(
        self,
        employee_id: str,
        course_id: str,
        course_name: str,
        course_version: int,
        run_id: str,
        created_by: str = "mutation_1_add_peer",
        raw_json: Optional[Dict] = None,
    ) -> bool:
        """
        Insert a versioned course row. Spider saves with version=0,
        mutation saves with the correct incremented version.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_courses
                    (employee_id, course_id, course_version, course_name,
                     run_id, created_by, raw_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (employee_id, course_id, course_version) DO NOTHING
            """, (
                employee_id, course_id, course_version, course_name,
                run_id, created_by,
                json.dumps(raw_json or {}, ensure_ascii=False),
            ))
            inserted = cur.rowcount > 0
        self.conn.commit()
        return inserted

    def get_latest_course_for_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recently created course for this employee."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT course_id, course_name, raw_json, created_at
                FROM spectre.employee_courses
                WHERE employee_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (employee_id,))
            return cur.fetchone()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()


# ═══════════════════════════════════════════════════════════════
# SCRAPING HELPER — reuses SHADE's BrightDataScraper
# ═══════════════════════════════════════════════════════════════

def scrape_peer_profile(linkedin_url: str) -> Optional[Dict[str, Any]]:
    """
    Scrape a single LinkedIn profile via Bright Data.
    Returns the raw profile dict or None.
    """
    if not SHADE_AVAILABLE:
        log.error("SHADE module not available — cannot scrape profiles.")
        return None

    scraper = BrightDataScraper()
    profiles = scraper.scrape_profiles_one_shot([linkedin_url], timeout_sec=180)
    if profiles:
        return profiles[0]

    # Fallback: try the trigger scraper (fresh scrape, slower)
    log.info("Dataset filter returned nothing — trying trigger scraper...")
    trigger = BrightDataTriggerScraper()
    if trigger.is_configured():
        profiles = trigger.scrape_profiles_trigger([linkedin_url], timeout_sec=240)
        if profiles:
            return profiles[0]

    log.warning(f"No profile data returned for {linkedin_url}")
    return None


# ═══════════════════════════════════════════════════════════════
# CORE MUTATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_mutation_1(
    run_id: str,
    peer_linkedin_url: str,
    peer_title: Optional[str] = None,
    peer_name: Optional[str] = None,
    skip_agent5: bool = False,
) -> Dict[str, Any]:
    """
    Execute Mutation 1: add a peer to an existing run.

    Returns a result dict with success status, employee info,
    steps completed, and cost breakdown.
    """
    log.info("=" * 70)
    log.info("MUTATION 1 — Manual Peer Add")
    log.info(f"  run_id: {run_id}")
    log.info(f"  peer_url: {peer_linkedin_url}")
    log.info(f"  peer_title: {peer_title or '(auto-detect)'}")
    log.info(f"  peer_name: {peer_name or '(auto-detect)'}")
    log.info("=" * 70)

    # ── HARD FAIL: refuse to run if critical agents are missing ──
    missing = []
    if not SHADE_AVAILABLE:  missing.append("SHADE (shade_1)")
    if not CIPHER_AVAILABLE: missing.append("CIPHER (cipher_3)")
    if not FRACTAL_AVAILABLE: missing.append("FRACTAL (fractal_4)")
    if not ATLAS_AVAILABLE:  missing.append("ATLAS (atlas_6)")
    if missing:
        error_msg = (
            f"❌ CANNOT RUN — missing required agents: {', '.join(missing)}. "
            f"Check the import errors printed at startup and fix them first."
        )
        log.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "run_id": run_id,
            "steps_completed": [],
            "costs": {},
        }

    costs = MutationCostTracker()
    steps_completed: List[Dict[str, Any]] = []
    db = MutationDB()

    try:
        # ─── STEP 0: VALIDATE RUN + FIND TARGET ──────────────
        log.info("\n🔍 Step 0: Validating run and finding target employee...")
        t0 = time.time()

        run_info = db.get_run(run_id)
        if not run_info:
            raise ValueError(f"Run not found in spectre.runs: {run_id}")

        log.info(f"  ✅ Run found: scope={run_info.get('scope')}, status={run_info.get('status')}")

        target = db.get_target_employee_for_run(run_id)
        if not target:
            # Extra debug: list all employees in the run
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT re.employee_id, re.role_in_run::text AS role,
                           e.full_name
                    FROM spectre.run_employees re
                    JOIN spectre.employees e ON e.employee_id = re.employee_id
                    WHERE re.run_id = %s
                """, (run_id,))
                all_emps = cur.fetchall()
                log.error(f"  All employees in run: {[dict(r) for r in all_emps]}")
            raise ValueError(
                f"No target employee found in run {run_id}. "
                f"Found {len(all_emps)} employee(s) but none matched target roles. "
                f"Check spectre.run_employees for this run_id."
            )

        target_employee_id = str(target["employee_id"])
        target_name = target["full_name"]
        target_company_id = str(target["source_company_id"]) if target.get("source_company_id") else None

        log.info(f"  ✅ Target: {target_name} ({target_employee_id})")
        log.info(f"  ✅ Existing peers: {db.get_peer_count(run_id)}")
        steps_completed.append({
            "step": "validate_run",
            "status": "ok",
            "target_employee": target_name,
            "elapsed_s": round(time.time() - t0, 2),
        })

        # ─── STEP 1: NORMALIZE URL + IDEMPOTENCY CHECK ───────
        log.info("\n🔗 Step 1: Normalizing URL and checking idempotency...")
        t1 = time.time()

        canonical_url = normalize_linkedin_url(peer_linkedin_url)
        if not canonical_url or "linkedin.com" not in canonical_url:
            raise ValueError(f"Invalid LinkedIn URL: {peer_linkedin_url}")

        existing_id = db.is_peer_already_in_run(run_id, canonical_url)
        if existing_id:
            log.info(f"  ⚠️  Peer already in run (employee_id={existing_id}). Idempotent — skipping.")
            db.close()
            return {
                "success": True,
                "idempotent": True,
                "message": f"Peer already exists in run (employee_id={existing_id})",
                "run_id": run_id,
                "peer_employee_id": existing_id,
                "steps_completed": steps_completed,
                "costs": costs.summary(),
            }

        # Auto-detect name from URL if not provided
        if not peer_name:
            peer_name = guess_name_from_url(peer_linkedin_url)

        log.info(f"  ✅ Canonical URL: {canonical_url}")
        log.info(f"  ✅ Peer name: {peer_name}")
        steps_completed.append({
            "step": "normalize_and_dedup",
            "status": "ok",
            "canonical_url": canonical_url,
            "peer_name": peer_name,
            "elapsed_s": round(time.time() - t1, 2),
        })

        # ─── STEP 2: SHADE — SCRAPE PROFILE ──────────────────
        log.info("\n📡 Step 2: Scraping peer profile via Bright Data (SHADE)...")
        t2 = time.time()
        bright_profile = None

        if SHADE_AVAILABLE:
            bright_profile = scrape_peer_profile(peer_linkedin_url)
            if bright_profile:
                # Extract real name/title/company from scraped data
                real_name = (bright_profile.get("name") or "").strip()
                real_title = (bright_profile.get("position") or "").strip()
                real_company = (
                    bright_profile.get("current_company_name")
                    or (bright_profile.get("current_company") or {}).get("name", "")
                ).strip()
                real_location = (bright_profile.get("location") or "").strip()

                if real_name:
                    peer_name = real_name
                if real_title and not peer_title:
                    peer_title = real_title

                log.info(f"  ✅ Scraped: {peer_name} | {peer_title} | {real_company}")
                costs.bright_data_rows += 1
            else:
                log.warning("  ⚠️  No profile data returned — proceeding with URL-derived info")
        else:
            log.warning("  ⚠️  SHADE module not available — skipping scrape")

        steps_completed.append({
            "step": "shade_scrape",
            "status": "ok" if bright_profile else "skipped",
            "profile_found": bool(bright_profile),
            "elapsed_s": round(time.time() - t2, 2),
        })

        # ─── STEP 3: SHADE — UPSERT EMPLOYEE + LINK TO RUN ──
        log.info("\n💾 Step 3: Upserting employee and linking to run (SHADE DB)...")
        t3 = time.time()

        # Upsert company if we have one
        peer_company_id = None
        if bright_profile:
            company_name = (
                bright_profile.get("current_company_name")
                or (bright_profile.get("current_company") or {}).get("name", "")
            )
            if company_name:
                peer_company_id = db.upsert_company(company_name)

        # Build profile_cache_text for downstream agents
        profile_cache = None
        if bright_profile:
            about = bright_profile.get("about", "")
            position = bright_profile.get("position", "")
            profile_cache = f"{peer_name}. {position}. {about}"[:2000] if about else position

        # Upsert employee
        peer_employee_id = db.upsert_peer_employee(
            linkedin_url=peer_linkedin_url,
            full_name=peer_name,
            current_title=peer_title,
            company_id=peer_company_id or target_company_id,
            profile_cache_text=profile_cache,
            location=bright_profile.get("location") if bright_profile else None,
        )
        log.info(f"  ✅ Employee upserted: {peer_employee_id}")

        # Link to run
        role = db.resolve_role_in_run("matched")
        linked = db.link_employee_to_run(
            run_id=run_id,
            employee_id=peer_employee_id,
            role_in_run=role,
            source_company_id=peer_company_id or target_company_id,
        )
        log.info(f"  ✅ Linked to run with role='{role}' (new_link={linked})")

        # ── Insert into employee_matches (ATLAS reads this, NOT run_employees!) ──
        peer_company_name = ""
        if bright_profile:
            peer_company_name = (
                bright_profile.get("current_company_name")
                or (bright_profile.get("current_company") or {}).get("name", "")
                or ""
            )
        match_inserted = db.insert_employee_match(
            run_id=run_id,
            target_employee_id=target_employee_id,
            peer_employee_id=peer_employee_id,
            peer_name=peer_name,
            peer_title=peer_title,
            peer_company=peer_company_name,
            peer_company_id=peer_company_id or target_company_id or "",
            match_score=90.0,
        )
        log.info(f"  ✅ employee_matches row inserted={match_inserted} (ATLAS will now see this peer)")

        # Insert employee_details if we have a Bright profile
        if bright_profile:
            try:
                db.insert_employee_details_from_bright(run_id, peer_employee_id, bright_profile)
                log.info("  ✅ Employee details inserted")
            except Exception as e:
                log.warning(f"  ⚠️  Employee details insert failed (non-fatal): {e}")
                db.rollback()

        db.commit()
        steps_completed.append({
            "step": "shade_upsert",
            "status": "ok",
            "peer_employee_id": peer_employee_id,
            "role_in_run": role,
            "company_id": peer_company_id,
            "elapsed_s": round(time.time() - t3, 2),
        })

        # ─── STEP 3b: ENSURE EMPLOYEE_DETAILS EXISTS ─────────
        # ATLAS requires employee_details to include peer in report.
        # If SHADE didn't populate it (no bright_profile), insert minimal row.
        log.info("\n🔍 Step 3b: Verifying employee_details for ATLAS compatibility...")
        db.ensure_minimal_employee_details(
            run_id=run_id,
            employee_id=peer_employee_id,
            full_name=peer_name,
            title=peer_title,
            linkedin_url=peer_linkedin_url,
        )

        # ─── STEP 4: CIPHER — SKILL EXTRACTION FOR NEW PEER ─
        log.info("\n🧠 Step 4: Extracting skills for new peer (CIPHER)...")
        t4 = time.time()
        cipher_result = {"success": False, "skills_extracted": 0}

        if CIPHER_AVAILABLE:
            try:
                extractor = SkillExtractor()
                cipher_result = extractor.process_employee(peer_employee_id, run_id)
                extractor.close()

                if cipher_result.get("cost"):
                    costs.record_step("cipher", cipher_result["cost"])

                log.info(f"  ✅ Extracted {cipher_result.get('skills_extracted', 0)} skills")
            except Exception as e:
                log.error(f"  ❌ CIPHER failed: {e}")
                cipher_result = {"success": False, "error": str(e), "skills_extracted": 0}
        else:
            log.warning("  ⚠️  CIPHER module not available — skipping skill extraction")

        steps_completed.append({
            "step": "cipher_skill_extraction",
            "status": "ok" if cipher_result.get("success") else "failed",
            "skills_extracted": cipher_result.get("skills_extracted", 0),
            "elapsed_s": round(time.time() - t4, 2),
        })

        # ─── STEP 5: SNAPSHOT CRITICAL GAPS (BEFORE) ─────────
        log.info("\n📸 Step 5: Snapshotting critical gaps before FRACTAL recompute...")
        gaps_before = db.get_critical_gaps(run_id, target_employee_id)
        log.info(f"  ✅ Critical gaps BEFORE: {len(gaps_before)} → {gaps_before[:5]}")

        # ─── STEP 6: FRACTAL — RECOMPUTE GAPS ────────────────
        log.info("\n📊 Step 6: Recomputing skill gaps for target (FRACTAL)...")
        t6 = time.time()
        fractal_result = {"gaps_found": 0}

        if FRACTAL_AVAILABLE:
            try:
                gap_results, run_cost = run_agent4(
                    run_id=run_id,
                    use_llm=True,
                    employee_id=target_employee_id,
                )
                fractal_result = {
                    "gaps_found": len(gap_results) if gap_results else 0,
                    "cost": run_cost.to_dict(),
                }
                costs.record_step("fractal", run_cost.to_dict())
                log.info(f"  ✅ FRACTAL found {fractal_result['gaps_found']} gaps")
            except Exception as e:
                log.error(f"  ❌ FRACTAL failed: {e}")
                fractal_result = {"gaps_found": 0, "error": str(e)}
        else:
            log.warning("  ⚠️  FRACTAL module not available — skipping gap recomputation")

        steps_completed.append({
            "step": "fractal_gap_recompute",
            "status": "ok" if fractal_result.get("gaps_found", 0) > 0 else "no_gaps",
            "gaps_found": fractal_result.get("gaps_found", 0),
            "elapsed_s": round(time.time() - t6, 2),
        })

        # ─── STEP 7: SNAPSHOT CRITICAL GAPS (AFTER) + DIFF ───
        log.info("\n📸 Step 7: Comparing critical gaps after FRACTAL...")
        gaps_after = db.get_critical_gaps(run_id, target_employee_id)
        log.info(f"  ✅ Critical gaps AFTER: {len(gaps_after)} → {gaps_after[:5]}")

        gaps_changed = (gaps_before != gaps_after)
        new_gaps = set(gaps_after) - set(gaps_before)
        removed_gaps = set(gaps_before) - set(gaps_after)

        log.info(f"  {'🔄' if gaps_changed else '✅'} Gaps changed: {gaps_changed}")
        if new_gaps:
            log.info(f"     New:     {sorted(new_gaps)}")
        if removed_gaps:
            log.info(f"     Removed: {sorted(removed_gaps)}")

        steps_completed.append({
            "step": "gap_diff",
            "gaps_changed": gaps_changed,
            "gaps_before_count": len(gaps_before),
            "gaps_after_count": len(gaps_after),
            "new_gaps": sorted(new_gaps),
            "removed_gaps": sorted(removed_gaps),
        })

        # ─── STEP 8: AGENT 5 — CONDITIONAL COURSE REGEN ─────
        agent5_ran = False
        course_version_saved = None
        if gaps_changed and not skip_agent5 and SPIDER_AVAILABLE:
            log.info("\n🕷️ Step 8: Critical gaps changed → running Agent 5 (Spider) for course regen...")
            t8 = time.time()

            # Get next course version BEFORE spider runs
            next_course_v = db.get_next_course_version(target_employee_id)
            log.info(f"  Next course version: v{next_course_v}")

            try:
                # Agent 5 is async — run it in an event loop
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    agent5_result = loop.run_until_complete(
                        spider_generate_courses(run_id=run_id, employee_id=target_employee_id)
                    )
                finally:
                    loop.close()

                agent5_ran = True
                if agent5_result.get("cost"):
                    costs.record_step("agent5_spider", agent5_result["cost"])

                courses_generated = agent5_result.get("courses_generated", 0)
                log.info(f"  ✅ Agent 5 generated {courses_generated} course(s)")

                # Spider may have saved with version=0, or may have failed to save.
                # Either way, find the latest course and stamp it with our version.
                latest_course = db.get_latest_course_for_employee(target_employee_id)
                if latest_course:
                    cid = str(latest_course["course_id"])
                    cname = latest_course.get("course_name", "")
                    log.info(f"  📚 Latest course: {cname} ({cid})")

                    # If spider saved with version=0, insert our versioned copy
                    if next_course_v > 0:
                        raw = latest_course.get("raw_json") or {}
                        saved = db.save_course_versioned(
                            employee_id=target_employee_id,
                            course_id=cid,
                            course_name=cname,
                            course_version=next_course_v,
                            run_id=run_id,
                            created_by="mutation_1_add_peer",
                            raw_json=raw if isinstance(raw, dict) else {},
                        )
                        if saved:
                            course_version_saved = next_course_v
                            log.info(f"  ✅ Course saved as version={next_course_v}")
                        else:
                            log.info(f"  ℹ️  Course version {next_course_v} already exists (idempotent)")
                    else:
                        course_version_saved = 0
                        log.info(f"  ✅ First course (version=0) saved by Spider")
                elif courses_generated == 0:
                    log.warning("  ⚠️  Spider failed to save course — no course found in DB")

            except Exception as e:
                log.error(f"  ❌ Agent 5 failed (non-fatal): {e}")

            steps_completed.append({
                "step": "agent5_course_regen",
                "status": "ok" if agent5_ran else "failed",
                "reason": "critical_gaps_changed",
                "course_version": course_version_saved,
                "elapsed_s": round(time.time() - t8, 2),
            })
        else:
            skip_reason = (
                "gaps_unchanged" if not gaps_changed
                else "skip_flag" if skip_agent5
                else "module_unavailable"
            )
            log.info(f"\n⏭️ Step 8: Skipping Agent 5 ({skip_reason})")
            steps_completed.append({
                "step": "agent5_course_regen",
                "status": "skipped",
                "reason": skip_reason,
            })

        # ─── STEP 9: ATLAS — REPORT REFRESH WITH NEW VERSION ─
        log.info("\n📄 Step 9: Refreshing ATLAS report with incremented version...")
        t9 = time.time()
        current_version = db.get_current_report_version(run_id, target_employee_id)
        new_version = current_version + 1
        log.info(f"  Version: v{current_version} → v{new_version}")

        # ARCHIVE the current report BEFORE ATLAS overwrites it
        archived_v = db.archive_current_report(run_id, target_employee_id)
        if archived_v is not None:
            log.info(f"  ✅ Previous report preserved as 'atlas_v{archived_v}'")

        atlas_report = None
        atlas_may_have_saved = False

        if ATLAS_AVAILABLE:
            # --- 9a: Run ATLAS (it saves to DB inside build(), then prints stats) ---
            try:
                atlas_report = run_atlas(
                    run_id=run_id,
                    employee_id=target_employee_id,
                )
                atlas_may_have_saved = True
                log.info(f"  ✅ ATLAS report generated")
            except Exception as e:
                # ATLAS build() commits to DB BEFORE run_atlas() print statements.
                # So if the crash is in the prints (e.g. 'multiAxisConfig' KeyError),
                # the report IS in the DB — we just need to stamp it.
                log.warning(f"  ⚠️  ATLAS run_atlas() raised: {e}")
                log.warning(f"     (Report likely saved to DB — will attempt stamp anyway)")
                atlas_may_have_saved = True  # build() commits before prints

            # --- 9b: Record ATLAS cost if available ---
            if atlas_report and atlas_report.get("costSummary"):
                try:
                    costs.record_step("atlas", atlas_report["costSummary"])
                except Exception:
                    pass

            # --- 9c: ALWAYS stamp mutation metadata (even if run_atlas crashed) ---
            # This is an UPDATE on the existing row. If ATLAS didn't save, it's a no-op.
            if atlas_may_have_saved:
                try:
                    db.stamp_report_version(
                        run_id=run_id,
                        employee_id=target_employee_id,
                        new_version=new_version,
                        peer_name=peer_name,
                        peer_linkedin_url=peer_linkedin_url,
                    )
                    log.info(f"  ✅ ATLAS report v{new_version} stamped as mutation_1_add_peer")
                except Exception as stamp_err:
                    log.error(f"  ❌ stamp_report_version failed: {stamp_err}")
        else:
            log.warning("  ⚠️  ATLAS module not available — skipping report refresh")

        steps_completed.append({
            "step": "atlas_report_refresh",
            "status": "ok" if atlas_report else ("stamped" if atlas_may_have_saved else "failed"),
            "report_version": new_version,
            "elapsed_s": round(time.time() - t9, 2),
        })

        # ─── DONE ────────────────────────────────────────────
        total_elapsed = round(time.time() - costs.started_at, 2)
        peer_count_after = db.get_peer_count(run_id)

        log.info("\n" + "=" * 70)
        log.info("✅ MUTATION 1 COMPLETE")
        log.info(f"   Run: {run_id} (v{new_version})")
        log.info(f"   Peer added: {peer_name} ({peer_employee_id})")
        log.info(f"   Peers in run: {peer_count_after}")
        log.info(f"   Gaps changed: {gaps_changed}")
        log.info(f"   Agent 5 ran: {agent5_ran}")
        log.info(f"   Total time: {total_elapsed}s")
        cost_summary = costs.summary()
        log.info(f"   Total cost: ${cost_summary['total_cost_usd']:.4f}")
        log.info("=" * 70)

        db.close()
        return {
            "success": True,
            "idempotent": False,
            "run_id": run_id,
            "run_version": new_version,
            "peer_employee_id": peer_employee_id,
            "peer_name": peer_name,
            "peer_title": peer_title,
            "peer_linkedin_url": peer_linkedin_url,
            "target_employee_id": target_employee_id,
            "target_name": target_name,
            "peers_in_run": peer_count_after,
            "skills_extracted": cipher_result.get("skills_extracted", 0),
            "gaps_found": fractal_result.get("gaps_found", 0),
            "critical_gaps_changed": gaps_changed,
            "new_critical_gaps": sorted(new_gaps),
            "removed_critical_gaps": sorted(removed_gaps),
            "agent5_ran": agent5_ran,
            "steps_completed": steps_completed,
            "costs": cost_summary,
        }

    except Exception as e:
        log.error(f"❌ MUTATION 1 FAILED: {e}", exc_info=True)
        db.rollback()
        db.close()
        return {
            "success": False,
            "error": str(e),
            "run_id": run_id,
            "steps_completed": steps_completed,
            "costs": costs.summary(),
        }


# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Mutation 1 — Manual Peer Add",
    description=(
        "Appends a new comparison peer into an existing run_id workspace, "
        "then selectively recomputes only downstream artifacts affected by "
        "the peer-set change. Tracks each mutation as an incremented "
        "run_version snapshot."
    ),
    version="1.0.0",
)


# ── Debug middleware — logs EVERY request ─────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class DebugMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        log.info(f"🌐 INCOMING: {request.method} {request.url.path} (full: {request.url})")
        response = await call_next(request)
        log.info(f"🌐 RESPONSE: {response.status_code} for {request.method} {request.url.path}")
        return response


app.add_middleware(DebugMiddleware)


# ── Root endpoint — confirms the server is alive ─────────────
@app.get("/")
def root():
    """Root endpoint — shows all available routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {
        "service": "mutation_1_add_peer",
        "status": "ok",
        "available_routes": routes,
        "hint": "POST to /mutation/add-peer or /add-peer with {run_id, peer_linkedin_url}",
    }


# ── Startup event — print all registered routes ──────────────
@app.on_event("startup")
def print_routes():
    log.info("=" * 60)
    log.info("🚀 MUTATION 1 SERVER STARTED — REGISTERED ROUTES:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            log.info(f"   {list(route.methods)} {route.path}")
    log.info("=" * 60)


# ── Request / Response models ─────────────────────────────────

class AddPeerRequest(BaseModel):
    run_id: str = Field(..., description="UUID of the existing run")
    peer_linkedin_url: str = Field(..., description="LinkedIn profile URL of the new peer")
    peer_title: Optional[str] = Field(None, description="Optional job title override (auto-detected from scrape if omitted)")
    peer_name: Optional[str] = Field(None, description="Optional name override (auto-detected from scrape/URL if omitted)")
    skip_agent5: bool = Field(False, description="If true, skip course regeneration even if gaps changed")


class StepInfo(BaseModel):
    step: str
    status: Optional[str] = None
    elapsed_s: Optional[float] = None


class AddPeerResponse(BaseModel):
    success: bool
    idempotent: bool = False
    message: Optional[str] = None
    error: Optional[str] = None
    run_id: str
    run_version: Optional[int] = None
    peer_employee_id: Optional[str] = None
    peer_name: Optional[str] = None
    peer_title: Optional[str] = None
    peer_linkedin_url: Optional[str] = None
    target_employee_id: Optional[str] = None
    target_name: Optional[str] = None
    peers_in_run: Optional[int] = None
    skills_extracted: Optional[int] = None
    gaps_found: Optional[int] = None
    critical_gaps_changed: Optional[bool] = None
    new_critical_gaps: Optional[List[str]] = None
    removed_critical_gaps: Optional[List[str]] = None
    agent5_ran: Optional[bool] = None
    steps_completed: List[Dict[str, Any]] = []
    costs: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/mutation/add-peer", response_model=AddPeerResponse)
@app.post("/add-peer", response_model=AddPeerResponse, include_in_schema=False)
def api_add_peer(req: AddPeerRequest):
    """
    Add a comparison peer to an existing run and recompute downstream
    artifacts (skills → gaps → courses → report).
    """
    if not req.run_id or not req.run_id.strip():
        raise HTTPException(status_code=400, detail="run_id is required")
    if not req.peer_linkedin_url or "linkedin.com" not in req.peer_linkedin_url:
        raise HTTPException(status_code=400, detail="A valid LinkedIn URL is required")

    try:
        result = run_mutation_1(
            run_id=req.run_id.strip(),
            peer_linkedin_url=req.peer_linkedin_url.strip(),
            peer_title=req.peer_title,
            peer_name=req.peer_name,
            skip_agent5=req.skip_agent5,
        )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            # Return 404 for "not found" type errors, 500 for others
            if "not found" in error_msg.lower() or "no target" in error_msg.lower():
                raise HTTPException(status_code=404, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        return AddPeerResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unhandled error in add-peer endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/mutation/health")
@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "agent": "mutation_1_add_peer",
        "version": "1.0.0",
        "modules": {
            "shade": SHADE_AVAILABLE,
            "cipher": CIPHER_AVAILABLE,
            "fractal": FRACTAL_AVAILABLE,
            "spider": SPIDER_AVAILABLE,
            "atlas": ATLAS_AVAILABLE,
        },
    }


@app.get("/mutation/run-info/{run_id}")
@app.get("/run-info/{run_id}", include_in_schema=False)
def api_run_info(run_id: str):
    """Quick read-only view of a run's peer set (for debugging)."""
    db = MutationDB()
    try:
        run_info = db.get_run(run_id)
        if not run_info:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        target = db.get_target_employee_for_run(run_id)
        peer_count = db.get_peer_count(run_id)

        version = 0
        if target:
            version = db.get_current_report_version(run_id, str(target["employee_id"]))

        return {
            "run_id": run_id,
            "status": run_info.get("status"),
            "target": {
                "employee_id": str(target["employee_id"]) if target else None,
                "name": target["full_name"] if target else None,
                "title": target["current_title"] if target else None,
            } if target else None,
            "peer_count": peer_count,
            "report_version": version,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mutation 1 — Manual Peer Add")
    parser.add_argument("--run-id", required=False, help="UUID of the existing run")
    parser.add_argument("--peer-url", required=False, help="LinkedIn profile URL of the new peer")
    parser.add_argument("--peer-title", default=None, help="Optional job title")
    parser.add_argument("--peer-name", default=None, help="Optional name")
    parser.add_argument("--skip-agent5", action="store_true", help="Skip course regeneration")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8010, help="Server port")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        print("\n" + "=" * 60)
        print("🚀 MUTATION 1 — Manual Peer Add Server")
        print("=" * 60)
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Docs: http://{args.host}:{args.port}/docs")
        print(f"   Root: http://{args.host}:{args.port}/")
        print()
        print("   Available routes:")
        print("     POST /mutation/add-peer  (main endpoint)")
        print("     POST /add-peer           (alias)")
        print("     GET  /mutation/health")
        print("     GET  /health             (alias)")
        print("     GET  /mutation/run-info/{run_id}")
        print("     GET  /                   (route listing)")
        print()
        print(f"   Modules loaded:")
        print(f"     SHADE:   {SHADE_AVAILABLE}")
        print(f"     CIPHER:  {CIPHER_AVAILABLE}")
        print(f"     FRACTAL: {FRACTAL_AVAILABLE}")
        print(f"     SPIDER:  {SPIDER_AVAILABLE}")
        print(f"     ATLAS:   {ATLAS_AVAILABLE}")
        print()
        print("   ⚠️  If you're integrating into an existing FastAPI app, use:")
        print("       from mutation_1_add_peer import app as mutation_app")
        print("       main_app.mount('/mutation', mutation_app)")
        print("     OR:")
        print("       from mutation_1_add_peer import api_add_peer, AddPeerRequest")
        print("       main_app.post('/mutation/add-peer')(api_add_peer)")
        print("=" * 60)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.run_id or not args.peer_url:
        parser.error("--run-id and --peer-url are required (or use --serve)")

    result = run_mutation_1(
        run_id=args.run_id,
        peer_linkedin_url=args.peer_url,
        peer_title=args.peer_title,
        peer_name=args.peer_name,
        skip_agent5=args.skip_agent5,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()