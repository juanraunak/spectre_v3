"""
Mutation 3 — Inject Skill Gap
===============================
A run-scoped mutation that programmatically injects a new Critical skill gap
into spectre.employee_skill_gaps for the primary employee, then triggers
a deterministic cascade through the downstream pipeline:

Pipeline:
  1. Validate run + resolve target employee
  2. Normalize + deduplicate the injected skill gap
  3. Upsert skill into spectre.skills (if new)
  4. Insert gap row into spectre.employee_skill_gaps  (Critical, with reasoning)
  5. SPIDER (Agent 5)  → course regeneration incorporating the new gap
  6. ATLAS  (Agent 6)  → full report refresh with incremented run_version

The injected gap is indistinguishable from organically computed gaps at the
database layer — same table, same schema, same importance enum.  Downstream
agents simply re-read the gap set and react.

Endpoint:
  POST /mutation/inject-gap

Usage:
  python mutation_3_inject_gap.py --serve --port 8012
  curl -X POST http://localhost:8012/mutation/inject-gap \
       -H "Content-Type: application/json" \
       -d '{"run_id":"...","skill_gap_name":"AI-Assisted Development"}'

Requires co-located agent modules:
  - fractal_4.py           (FRACTAL — for DB_CONFIG + helpers, optional)
  - spectre_spider_5.py    (SPIDER  — Agent 5)
  - atlas_6.py             (ATLAS   — Agent 6)
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
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# AGENT IMPORTS
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("🔌 MUTATION 3 — LOADING AGENT MODULES...")
print("=" * 60)

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

# ── HARD FAIL CHECK ──────────────────────────────────────────
_REQUIRED_AGENTS = {
    "SPIDER": SPIDER_AVAILABLE,
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
    format="%(asctime)s - MUTATION3 - %(levelname)s - %(message)s",
)
log = logging.getLogger("mutation3")

log.info("=" * 60)
log.info("📦 mutation_3_inject_gap.py LOADED")
log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# COST TRACKER
# ═══════════════════════════════════════════════════════════════

GPT4O_INPUT_PER_1M = 2.50
GPT4O_OUTPUT_PER_1M = 10.00
GOOGLE_CSE_COST_PER_QUERY = 0.005


@dataclass
class MutationCostTracker:
    """Accumulates costs for the entire mutation pipeline."""

    gpt_input_tokens: int = 0
    gpt_output_tokens: int = 0
    gpt_calls: int = 0
    google_queries: int = 0
    started_at: float = field(default_factory=time.time)
    step_costs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_step(self, step_name: str, cost_data: Dict[str, Any]) -> None:
        self.step_costs[step_name] = cost_data
        gpt = cost_data.get("gpt", cost_data)
        self.gpt_input_tokens += int(
            gpt.get("input_tokens", 0)
            or gpt.get("prompt_tokens", 0)
            or gpt.get("inputTokens", 0)
            or gpt.get("total_input_tokens", 0)
            or 0
        )
        self.gpt_output_tokens += int(
            gpt.get("output_tokens", 0)
            or gpt.get("completion_tokens", 0)
            or gpt.get("outputTokens", 0)
            or gpt.get("total_output_tokens", 0)
            or 0
        )
        self.gpt_calls += int(
            gpt.get("calls", 0)
            or gpt.get("llm_calls", 0)
            or gpt.get("totalCalls", 0)
            or 0
        )
        self.google_queries += int(
            cost_data.get("google_cse", {}).get("queries", 0)
            or cost_data.get("google_searches", 0)
            or 0
        )

    def summary(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self.started_at, 2)
        input_cost = (self.gpt_input_tokens / 1_000_000) * GPT4O_INPUT_PER_1M
        output_cost = (self.gpt_output_tokens / 1_000_000) * GPT4O_OUTPUT_PER_1M
        gpt_total = input_cost + output_cost
        google_cost = self.google_queries * GOOGLE_CSE_COST_PER_QUERY
        total = gpt_total + google_cost

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
            "google_cse": {
                "queries": self.google_queries,
                "cost_usd": round(google_cost, 6),
            },
            "step_breakdown": self.step_costs,
        }


# ═══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════

def _get_conn():
    dsn = os.getenv("SPECTRE_DB_URL", DEFAULT_DSN)
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn


class Mutation3DB:
    """All DB reads and writes specific to Mutation 3."""

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

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

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
        then falls back to first employee in the run.
        """
        target_roles = ["primary", "seed", "target"]

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT re.role_in_run::text AS role
                FROM spectre.run_employees re
                WHERE re.run_id = %s
            """, (run_id,))
            existing_roles = [r["role"] for r in cur.fetchall()]
            log.info(f"  Roles found in run: {existing_roles}")

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

            # Fallback: first employee
            log.warning("  No standard target role found. Using first employee in run.")
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

    # ── Skill resolution ────────────────────────────────────
    def resolve_or_create_skill(self, skill_name: str, category: str = "injected") -> str:
        """
        Find an existing skill by name or create a new one.
        Returns the skill_id.
        """
        with self.conn.cursor() as cur:
            # Try to find existing
            cur.execute(
                "SELECT skill_id FROM spectre.skills WHERE lower(name) = lower(%s) LIMIT 1",
                (skill_name,),
            )
            row = cur.fetchone()
            if row:
                log.info(f"  ✅ Existing skill found: {row['skill_id']}")
                return str(row["skill_id"])

            # Create new skill
            new_id = str(uuid.uuid4())
            metadata = json.dumps({
                "source": "mutation_3_inject_gap",
                "injected_at": datetime.now(timezone.utc).isoformat(),
                "description": f"Skill injected via Mutation 3: {skill_name}",
            })
            cur.execute("""
                INSERT INTO spectre.skills (skill_id, name, category, metadata_json, raw_json, created_at)
                VALUES (%s, %s, %s, %s::jsonb, '{}'::jsonb, %s)
                ON CONFLICT DO NOTHING
                RETURNING skill_id
            """, (new_id, skill_name, category, metadata, datetime.now(timezone.utc)))
            row = cur.fetchone()
            if row:
                log.info(f"  ✅ Created new skill: {row['skill_id']}")
                return str(row["skill_id"])

            # Race condition: someone else created it between our SELECT and INSERT
            cur.execute(
                "SELECT skill_id FROM spectre.skills WHERE lower(name) = lower(%s) LIMIT 1",
                (skill_name,),
            )
            row = cur.fetchone()
            return str(row["skill_id"]) if row else new_id

    # ── Idempotency: check if gap already exists ────────────
    def gap_already_exists(self, run_id: str, employee_id: str, skill_gap_name: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM spectre.employee_skill_gaps
                WHERE run_id = %s AND employee_id = %s
                  AND lower(skill_gap_name) = lower(%s)
                LIMIT 1
            """, (run_id, employee_id, skill_gap_name))
            return cur.fetchone() is not None

    # ── Map importance to valid enum ────────────────────────
    def resolve_importance(self, preferred: str = "Critical") -> str:
        """Pick a valid skill_importance value that the DB accepts."""
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT pg_get_constraintdef(oid) FROM pg_constraint
                    WHERE conname = 'employee_skill_gaps_skill_importance_check'
                """)
                row = cur.fetchone()
                if row:
                    vals = re.findall(r"'([^']+)'::text", row[0] if isinstance(row, tuple) else row.get("pg_get_constraintdef", ""))
                    if vals:
                        log.info(f"  Allowed importance values: {vals}")
                        # Try exact match first
                        for v in vals:
                            if v.lower() == preferred.lower():
                                return v
                        # Fallback to first "critical"-like value
                        for v in vals:
                            if "critical" in v.lower() or "high" in v.lower():
                                return v
                        return vals[0]
            except Exception as e:
                log.warning(f"  Could not read importance constraint: {e}")
        return preferred

    # ── Inject the skill gap ────────────────────────────────
    def inject_skill_gap(
        self,
        run_id: str,
        employee_id: str,
        skill_id: str,
        skill_gap_name: str,
        skill_importance: str,
        gap_reasoning: str,
        competitor_companies: Optional[List[str]] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert a new skill gap row into spectre.employee_skill_gaps.
        Returns True if inserted, False if conflict.
        """
        comp_json = json.dumps(competitor_companies or [], ensure_ascii=False)
        raw_json = json.dumps(
            {
                **(raw_metadata or {}),
                "gap_source": "mutation_3_injection",
                "gap_tier": skill_importance,
                "overall_relevance": 0.95,
                "injected_at": datetime.now(timezone.utc).isoformat(),
                "analysis_method": "manual_strategic_injection",
            },
            ensure_ascii=False,
        )

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_skill_gaps (
                    run_id, employee_id, skill_id, skill_gap_name,
                    skill_importance, gap_reasoning, competitor_companies,
                    raw_json, created_by_agent, model_name, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s::jsonb,
                    %s::jsonb, 'mutation_3', 'strategic_injection', %s
                )
            """, (
                run_id, employee_id, skill_id, skill_gap_name,
                skill_importance, gap_reasoning, comp_json,
                raw_json, datetime.now(timezone.utc),
            ))
            return cur.rowcount > 0

    # ── Gap snapshot ────────────────────────────────────────
    def get_critical_gaps(self, run_id: str, employee_id: str) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT skill_gap_name
                FROM spectre.employee_skill_gaps
                WHERE run_id = %s AND employee_id = %s
                  AND skill_importance IN ('Critical', 'Important', 'critical', 'important')
                ORDER BY skill_gap_name
            """, (run_id, employee_id))
            return sorted([r["skill_gap_name"] for r in cur.fetchall()])

    def get_all_gaps(self, run_id: str, employee_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT skill_gap_name, skill_importance, gap_reasoning,
                       created_by_agent, created_at
                FROM spectre.employee_skill_gaps
                WHERE run_id = %s AND employee_id = %s
                ORDER BY
                  CASE skill_importance
                    WHEN 'Critical' THEN 1 WHEN 'Important' THEN 2
                    WHEN 'Nice-to-have' THEN 3 ELSE 4 END,
                  skill_gap_name
            """, (run_id, employee_id))
            return [dict(r) for r in cur.fetchall()]

    # ── Report versioning (same pattern as Mutation 1) ──────
    def get_current_report_version(self, run_id: str, employee_id: str) -> int:
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

    def archive_current_report(self, run_id: str, employee_id: str) -> Optional[int]:
        """Copy the current 'atlas' row to 'atlas_v{N}' before ATLAS overwrites."""
        with self.conn.cursor() as cur:
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

            report_json_val = row["report_json"]
            if isinstance(report_json_val, (dict, list)):
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

    def stamp_report_version(
        self,
        run_id: str,
        employee_id: str,
        new_version: int,
        skill_gap_name: str,
    ) -> None:
        """Stamp mutation metadata on top of ATLAS's freshly written report."""
        mutation_summary = f"mutation_3:inject_gap:{skill_gap_name}"
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE spectre.employee_reports
                SET report_version      = %s,
                    created_by_mutation  = 'mutation_3_inject_gap',
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

    def get_peer_count(self, run_id: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS cnt
                FROM spectre.run_employees
                WHERE run_id = %s AND role_in_run::text NOT IN ('primary', 'seed', 'target')
            """, (run_id,))
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0


# ═══════════════════════════════════════════════════════════════
# CORE MUTATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_mutation_3(
    run_id: str,
    skill_gap_name: str,
    gap_reasoning: Optional[str] = None,
    skill_importance: str = "Critical",
    competitor_companies: Optional[List[str]] = None,
    skip_agent5: bool = False,
    skip_atlas: bool = False,
) -> Dict[str, Any]:
    """
    Execute Mutation 3: inject a skill gap and cascade through the pipeline.

    Steps:
      1. Validate run + find target employee
      2. Resolve or create the skill in spectre.skills
      3. Inject the gap into spectre.employee_skill_gaps
      4. Agent 5 (Spider) — course regeneration
      5. Agent 6 (Atlas)  — report refresh with new version

    Returns a result dict with success status, steps, and cost breakdown.
    """
    log.info("=" * 70)
    log.info("MUTATION 3 — Inject Skill Gap")
    log.info(f"  run_id:         {run_id}")
    log.info(f"  skill_gap_name: {skill_gap_name}")
    log.info(f"  importance:     {skill_importance}")
    log.info(f"  reasoning:      {gap_reasoning or '(auto-generated)'}")
    log.info("=" * 70)

    # ── HARD FAIL: refuse to run without required agents ──
    missing = []
    if not SPIDER_AVAILABLE:
        missing.append("SPIDER (spectre_spider_5)")
    if not ATLAS_AVAILABLE:
        missing.append("ATLAS (atlas_6)")
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
    db = Mutation3DB()

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
                f"Found {len(all_emps)} employee(s) but none matched target roles."
            )

        target_employee_id = str(target["employee_id"])
        target_name = target["full_name"]

        log.info(f"  ✅ Target: {target_name} ({target_employee_id})")
        steps_completed.append({
            "step": "validate_run",
            "status": "ok",
            "target_employee": target_name,
            "target_employee_id": target_employee_id,
            "elapsed_s": round(time.time() - t0, 2),
        })

        # ─── STEP 1: NORMALIZE + DEDUP ───────────────────────
        log.info("\n🔗 Step 1: Normalizing skill gap name and checking idempotency...")
        t1 = time.time()

        skill_gap_name = skill_gap_name.strip()
        if not skill_gap_name:
            raise ValueError("skill_gap_name cannot be empty")

        already_exists = db.gap_already_exists(run_id, target_employee_id, skill_gap_name)
        if already_exists:
            log.info(f"  ⚠️  Gap '{skill_gap_name}' already exists for this run+employee. Idempotent — skipping injection.")
            # Even if already present, still cascade through Agent 5 + Atlas
            # to ensure the report reflects the latest state.
            log.info("  ℹ️  Will still cascade through Agent 5 + Atlas for freshness.")

        steps_completed.append({
            "step": "normalize_and_dedup",
            "status": "already_exists" if already_exists else "ok",
            "skill_gap_name": skill_gap_name,
            "elapsed_s": round(time.time() - t1, 2),
        })

        # ─── STEP 2: RESOLVE OR CREATE SKILL ─────────────────
        log.info("\n🧩 Step 2: Resolving skill in spectre.skills...")
        t2 = time.time()

        skill_id = db.resolve_or_create_skill(
            skill_name=skill_gap_name,
            category="mutation-3-injected",
        )
        db.commit()
        log.info(f"  ✅ skill_id: {skill_id}")

        steps_completed.append({
            "step": "resolve_skill",
            "status": "ok",
            "skill_id": skill_id,
            "elapsed_s": round(time.time() - t2, 2),
        })

        # ─── STEP 3: INJECT THE GAP ──────────────────────────
        log.info("\n💉 Step 3: Injecting skill gap into spectre.employee_skill_gaps...")
        t3 = time.time()

        # Resolve importance to valid DB enum value
        resolved_importance = db.resolve_importance(skill_importance)
        log.info(f"  Importance: '{skill_importance}' → '{resolved_importance}'")

        # Build reasoning if not provided
        if not gap_reasoning:
            gap_reasoning = (
                f"Strategically injected via Mutation 3. "
                f"'{skill_gap_name}' has been identified as a critical capability deficit "
                f"requiring immediate upskilling to maintain competitive positioning "
                f"in the current market landscape."
            )

        injected = False
        if not already_exists:
            injected = db.inject_skill_gap(
                run_id=run_id,
                employee_id=target_employee_id,
                skill_id=skill_id,
                skill_gap_name=skill_gap_name,
                skill_importance=resolved_importance,
                gap_reasoning=gap_reasoning,
                competitor_companies=competitor_companies,
                raw_metadata={
                    "mutation": "mutation_3_inject_gap",
                    "requested_importance": skill_importance,
                    "resolved_importance": resolved_importance,
                },
            )
            db.commit()
            log.info(f"  ✅ Gap injected: {injected}")
        else:
            log.info(f"  ⏭️  Skipped injection (already exists)")

        # Snapshot gaps after injection
        gaps_after = db.get_critical_gaps(run_id, target_employee_id)
        log.info(f"  ✅ Critical gaps now: {len(gaps_after)} → {gaps_after[:5]}")

        steps_completed.append({
            "step": "inject_gap",
            "status": "injected" if injected else "already_exists",
            "skill_gap_name": skill_gap_name,
            "importance": resolved_importance,
            "total_critical_gaps": len(gaps_after),
            "elapsed_s": round(time.time() - t3, 2),
        })

        # ─── STEP 4: AGENT 5 (SPIDER) — COURSE REGEN ────────
        agent5_ran = False
        if not skip_agent5 and SPIDER_AVAILABLE:
            log.info("\n🕷️ Step 4: Running Agent 5 (Spider) for course regeneration...")
            t4 = time.time()
            try:
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
                courses_gen = agent5_result.get("courses_generated", 0)
                log.info(f"  ✅ Agent 5 generated {courses_gen} course(s)")
            except Exception as e:
                log.error(f"  ❌ Agent 5 failed (non-fatal): {e}")

            steps_completed.append({
                "step": "agent5_course_regen",
                "status": "ok" if agent5_ran else "failed",
                "reason": "gap_injected",
                "courses_generated": agent5_result.get("courses_generated", 0) if agent5_ran else 0,
                "elapsed_s": round(time.time() - t4, 2),
            })
        else:
            skip_reason = "skip_flag" if skip_agent5 else "module_unavailable"
            log.info(f"\n⏭️ Step 4: Skipping Agent 5 ({skip_reason})")
            steps_completed.append({
                "step": "agent5_course_regen",
                "status": "skipped",
                "reason": skip_reason,
            })

        # ─── STEP 5: ATLAS — REPORT REFRESH ──────────────────
        atlas_report = None
        new_version = 0
        if not skip_atlas and ATLAS_AVAILABLE:
            log.info("\n📄 Step 5: Refreshing ATLAS report with incremented version...")
            t5 = time.time()

            current_version = db.get_current_report_version(run_id, target_employee_id)
            new_version = current_version + 1
            log.info(f"  Version: v{current_version} → v{new_version}")

            # Archive current report
            archived_v = db.archive_current_report(run_id, target_employee_id)
            if archived_v is not None:
                log.info(f"  ✅ Previous report preserved as 'atlas_v{archived_v}'")

            try:
                atlas_report = run_atlas(
                    run_id=run_id,
                    employee_id=target_employee_id,
                )
                if atlas_report and atlas_report.get("costSummary"):
                    costs.record_step("atlas", atlas_report["costSummary"])

                # Stamp mutation version metadata
                db.stamp_report_version(
                    run_id=run_id,
                    employee_id=target_employee_id,
                    new_version=new_version,
                    skill_gap_name=skill_gap_name,
                )
                log.info(f"  ✅ ATLAS report v{new_version} saved and stamped")
            except Exception as e:
                log.error(f"  ❌ ATLAS failed: {e}")

            steps_completed.append({
                "step": "atlas_report_refresh",
                "status": "ok" if atlas_report else "failed",
                "report_version": new_version,
                "elapsed_s": round(time.time() - t5, 2),
            })
        else:
            skip_reason = "skip_flag" if skip_atlas else "module_unavailable"
            log.info(f"\n⏭️ Step 5: Skipping ATLAS ({skip_reason})")
            steps_completed.append({
                "step": "atlas_report_refresh",
                "status": "skipped",
                "reason": skip_reason,
            })

        # ─── DONE ────────────────────────────────────────────
        total_elapsed = round(time.time() - costs.started_at, 2)
        all_gaps = db.get_all_gaps(run_id, target_employee_id)

        log.info("\n" + "=" * 70)
        log.info("✅ MUTATION 3 COMPLETE")
        log.info(f"   Run: {run_id} (v{new_version})")
        log.info(f"   Injected gap: {skill_gap_name} [{resolved_importance}]")
        log.info(f"   Target: {target_name} ({target_employee_id})")
        log.info(f"   Total gaps: {len(all_gaps)} ({len(gaps_after)} critical)")
        log.info(f"   Agent 5 ran: {agent5_ran}")
        log.info(f"   Atlas ran: {atlas_report is not None}")
        log.info(f"   Total time: {total_elapsed}s")
        cost_summary = costs.summary()
        log.info(f"   Total cost: ${cost_summary['total_cost_usd']:.4f}")
        log.info("=" * 70)

        db.close()
        return {
            "success": True,
            "run_id": run_id,
            "run_version": new_version,
            "target_employee_id": target_employee_id,
            "target_name": target_name,
            "injected_gap": {
                "skill_gap_name": skill_gap_name,
                "skill_id": skill_id,
                "importance": resolved_importance,
                "reasoning": gap_reasoning,
                "was_new": injected,
                "already_existed": already_exists,
            },
            "total_gaps": len(all_gaps),
            "critical_gaps": gaps_after,
            "agent5_ran": agent5_ran,
            "atlas_ran": atlas_report is not None,
            "steps_completed": steps_completed,
            "costs": cost_summary,
        }

    except Exception as e:
        log.error(f"❌ MUTATION 3 FAILED: {e}", exc_info=True)
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
    title="Mutation 3 — Inject Skill Gap",
    description=(
        "Injects a new Critical skill gap into an existing run, then cascades "
        "through Agent 5 (course generation) and Agent 6 (ATLAS report refresh) "
        "to produce a new versioned report reflecting the injected deficit. "
        "The gap becomes indistinguishable from organically computed gaps."
    ),
    version="1.0.0",
)


# ── Debug middleware ──────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class DebugMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        log.info(f"🌐 INCOMING: {request.method} {request.url.path}")
        response = await call_next(request)
        log.info(f"🌐 RESPONSE: {response.status_code} for {request.method} {request.url.path}")
        return response


app.add_middleware(DebugMiddleware)


# ── Root ──────────────────────────────────────────────────────
@app.get("/")
def root():
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({"path": route.path, "methods": list(route.methods)})
    return {
        "service": "mutation_3_inject_gap",
        "status": "ok",
        "available_routes": routes,
        "hint": "POST to /mutation/inject-gap with {run_id, skill_gap_name}",
    }


@app.on_event("startup")
def print_routes():
    log.info("=" * 60)
    log.info("🚀 MUTATION 3 SERVER STARTED — REGISTERED ROUTES:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            log.info(f"   {list(route.methods)} {route.path}")
    log.info("=" * 60)


# ── Request / Response models ─────────────────────────────────

class InjectGapRequest(BaseModel):
    run_id: str = Field(..., description="UUID of the existing run")
    skill_gap_name: str = Field(..., description="Name of the skill gap to inject (e.g. 'AI-Assisted Development')")
    gap_reasoning: Optional[str] = Field(
        None,
        description="Custom reasoning for why this gap matters. Auto-generated if omitted.",
    )
    skill_importance: str = Field(
        "Critical",
        description="Importance level: 'Critical', 'Important', or 'Nice-to-have'. Defaults to Critical.",
    )
    competitor_companies: Optional[List[str]] = Field(
        None,
        description="Optional list of competitor companies where this skill is prevalent.",
    )
    skip_agent5: bool = Field(False, description="If true, skip course regeneration")
    skip_atlas: bool = Field(False, description="If true, skip ATLAS report refresh")


class InjectGapResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    run_id: str
    run_version: Optional[int] = None
    target_employee_id: Optional[str] = None
    target_name: Optional[str] = None
    injected_gap: Optional[Dict[str, Any]] = None
    total_gaps: Optional[int] = None
    critical_gaps: Optional[List[str]] = None
    agent5_ran: Optional[bool] = None
    atlas_ran: Optional[bool] = None
    steps_completed: List[Dict[str, Any]] = []
    costs: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/mutation/inject-gap", response_model=InjectGapResponse)
@app.post("/inject-gap", response_model=InjectGapResponse, include_in_schema=False)
def api_inject_gap(req: InjectGapRequest):
    """
    Inject a skill gap into an existing run and cascade through
    Agent 5 (courses) → Agent 6 (ATLAS report).

    The injected gap is stored as a first-class row in
    spectre.employee_skill_gaps with the same schema as organically
    computed gaps. Downstream agents simply re-read the gap set.
    """
    if not req.run_id or not req.run_id.strip():
        raise HTTPException(status_code=400, detail="run_id is required")
    if not req.skill_gap_name or not req.skill_gap_name.strip():
        raise HTTPException(status_code=400, detail="skill_gap_name is required")

    try:
        result = run_mutation_3(
            run_id=req.run_id.strip(),
            skill_gap_name=req.skill_gap_name.strip(),
            gap_reasoning=req.gap_reasoning,
            skill_importance=req.skill_importance,
            competitor_companies=req.competitor_companies,
            skip_agent5=req.skip_agent5,
            skip_atlas=req.skip_atlas,
        )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            if "not found" in error_msg.lower() or "no target" in error_msg.lower():
                raise HTTPException(status_code=404, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        return InjectGapResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unhandled error in inject-gap endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/mutation/health")
@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok",
        "agent": "mutation_3_inject_gap",
        "version": "1.0.0",
        "modules": {
            "spider": SPIDER_AVAILABLE,
            "atlas": ATLAS_AVAILABLE,
        },
    }


@app.get("/mutation/run-gaps/{run_id}")
@app.get("/run-gaps/{run_id}", include_in_schema=False)
def api_run_gaps(run_id: str):
    """Read-only view of a run's current skill gaps (for debugging)."""
    db = Mutation3DB()
    try:
        run_info = db.get_run(run_id)
        if not run_info:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        target = db.get_target_employee_for_run(run_id)
        if not target:
            raise HTTPException(status_code=404, detail=f"No target employee in run: {run_id}")

        target_id = str(target["employee_id"])
        all_gaps = db.get_all_gaps(run_id, target_id)
        critical_gaps = db.get_critical_gaps(run_id, target_id)
        version = db.get_current_report_version(run_id, target_id)

        return {
            "run_id": run_id,
            "target": {
                "employee_id": target_id,
                "name": target["full_name"],
                "title": target.get("current_title"),
            },
            "report_version": version,
            "total_gaps": len(all_gaps),
            "critical_gap_count": len(critical_gaps),
            "critical_gaps": critical_gaps,
            "all_gaps": [
                {
                    "skill_gap_name": g["skill_gap_name"],
                    "importance": g["skill_importance"],
                    "agent": g.get("created_by_agent"),
                }
                for g in all_gaps
            ],
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mutation 3 — Inject Skill Gap")
    parser.add_argument("--run-id", required=False, help="UUID of the existing run")
    parser.add_argument("--skill-gap", required=False, help="Skill gap name to inject")
    parser.add_argument("--importance", default="Critical", help="Skill importance (Critical/Important/Nice-to-have)")
    parser.add_argument("--reasoning", default=None, help="Custom gap reasoning")
    parser.add_argument("--skip-agent5", action="store_true", help="Skip course regeneration")
    parser.add_argument("--skip-atlas", action="store_true", help="Skip ATLAS report refresh")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8012, help="Server port")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        print("\n" + "=" * 60)
        print("🚀 MUTATION 3 — Inject Skill Gap Server")
        print("=" * 60)
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Docs: http://{args.host}:{args.port}/docs")
        print(f"   Root: http://{args.host}:{args.port}/")
        print()
        print("   Available routes:")
        print("     POST /mutation/inject-gap  (main endpoint)")
        print("     POST /inject-gap           (alias)")
        print("     GET  /mutation/health")
        print("     GET  /health               (alias)")
        print("     GET  /mutation/run-gaps/{run_id}")
        print("     GET  /                     (route listing)")
        print()
        print(f"   Modules loaded:")
        print(f"     SPIDER:  {SPIDER_AVAILABLE}")
        print(f"     ATLAS:   {ATLAS_AVAILABLE}")
        print()
        print("   ⚠️  Integration into existing FastAPI app:")
        print("       from mutation_3_inject_gap import app as mutation3_app")
        print("       main_app.mount('/mutation', mutation3_app)")
        print("     OR:")
        print("       from mutation_3_inject_gap import api_inject_gap, InjectGapRequest")
        print("       main_app.post('/mutation/inject-gap')(api_inject_gap)")
        print("=" * 60)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.run_id or not args.skill_gap:
        parser.error("--run-id and --skill-gap are required (or use --serve)")

    result = run_mutation_3(
        run_id=args.run_id,
        skill_gap_name=args.skill_gap,
        gap_reasoning=args.reasoning,
        skill_importance=args.importance,
        skip_agent5=args.skip_agent5,
        skip_atlas=args.skip_atlas,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()