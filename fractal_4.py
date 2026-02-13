"""
Agent 4 — DB-backed Skill Gap Analyzer
=======================================
Flow:
  1. Accept a run_id and/or employee_id
  2. Query run_employees → find TARGET + COMPETITOR employees
  3. Pull skills from employee_skills + skills tables
  4. Pull employee metadata from employees + companies
  5. Run skill-gap logic (normalize, frequency, heuristic/LLM scoring)
  6. Write results to employee_skill_gaps

Usage (CLI):
  python agent4_db_skill_gap.py --run_id <uuid>
  python agent4_db_skill_gap.py --run_id <uuid> --no_llm   # heuristic only
  python agent4_db_skill_gap.py --employee_id <uuid>
  python agent4_db_skill_gap.py --run_id <uuid> --employee_id <uuid>

Usage (API):
  uvicorn agent4_db_skill_gap:app --host 0.0.0.0 --port 8004
  POST /analyze  {"run_id": "...", "employee_id": "...", "use_llm": true}

Usage (serve via CLI):
  python agent4_db_skill_gap.py --serve --port 8004
"""

import os
import re
import json
import logging
import argparse
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

try:
    from cost_tracker import get_cost_tracker
    tracker = get_cost_tracker()
except Exception:
    tracker = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent4_db")

# ---------------------------------------------------------------------------
# DB Connection
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("SPECTRE_DB_HOST", "monsterdb.postgres.database.azure.com"),
    "port": int(os.getenv("SPECTRE_DB_PORT", 5432)),
    "dbname": os.getenv("SPECTRE_DB_NAME", "postgres"),
    "user": os.getenv("SPECTRE_DB_USER", ""),
    "password": os.getenv("SPECTRE_DB_PASSWORD", ""),
    "sslmode": os.getenv("SPECTRE_DB_SSLMODE", "require"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EmployeeProfile:
    employee_id: str
    full_name: str
    current_title: str
    company_id: Optional[str]
    company_name: Optional[str]
    role_in_run: str  # 'target' or 'competitor'
    skills: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillGapResult:
    employee_id: str
    skill_id: Optional[str]
    skill_gap_name: str
    skill_importance: str
    gap_reasoning: str
    competitor_companies: List[str]
    raw_json: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunCost:
    """Tracks LLM token usage and cost for a single run."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_name: str = ""
    cost_usd: float = 0.0

    # Default pricing (GPT-4o) — override via env vars
    INPUT_COST_PER_M: float = float(os.getenv("LLM_INPUT_COST_PER_M", "2.50"))
    OUTPUT_COST_PER_M: float = float(os.getenv("LLM_OUTPUT_COST_PER_M", "10.00"))

    def record_usage(self, prompt_tok: int, completion_tok: int, model: str = ""):
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += prompt_tok + completion_tok
        self.llm_calls += 1
        if model:
            self.model_name = model
        self.cost_usd = (
            (self.prompt_tokens / 1_000_000) * self.INPUT_COST_PER_M +
            (self.completion_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "model_name": self.model_name,
            "cost_usd": round(self.cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# DB Reader
# ---------------------------------------------------------------------------
class RunDataReader:

    def __init__(self, conn):
        self.conn = conn

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT run_id, scope, status, target_company_id,
                       config_json, started_at, ended_at
                FROM spectre.runs WHERE run_id = %s
            """, (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_run_employees(self, run_id: str) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    re.employee_id, re.role_in_run::text AS role_in_run,
                    re.source_company_id, e.full_name, e.current_title,
                    c.name AS company_name
                FROM spectre.run_employees re
                JOIN spectre.employees e ON e.employee_id = re.employee_id
                LEFT JOIN spectre.companies c ON c.company_id = re.source_company_id
                WHERE re.run_id = %s
                ORDER BY re.role_in_run::text, e.full_name
            """, (run_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_all_employee_skills_for_run(self, run_id: str) -> Dict[str, List[Dict[str, Any]]]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    es.employee_id, es.skill_id,
                    s.name AS skill_name, s.category AS skill_category,
                    es.skill_confidence, es.level
                FROM spectre.employee_skills es
                JOIN spectre.skills s ON s.skill_id = es.skill_id
                WHERE es.run_id = %s
                ORDER BY es.employee_id, es.skill_confidence DESC NULLS LAST
            """, (run_id,))
            rows = cur.fetchall()

        by_emp: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            eid = str(r["employee_id"])
            by_emp.setdefault(eid, []).append(dict(r))
        return by_emp

    def get_skill_id_by_name(self, skill_name: str) -> Optional[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT skill_id FROM spectre.skills WHERE lower(name) = lower(%s) LIMIT 1",
                (skill_name,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def bulk_get_skill_ids(self, skill_names: List[str]) -> Dict[str, str]:
        if not skill_names:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT skill_id, lower(name) AS lname
                FROM spectre.skills WHERE lower(name) = ANY(%s)
            """, ([n.lower() for n in skill_names],))
            return {row[1]: str(row[0]) for row in cur.fetchall()}

    def find_runs_for_employee(self, employee_id: str) -> List[str]:
        """Find all run_ids where this employee appears."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT run_id FROM spectre.run_employees
                WHERE employee_id = %s
                ORDER BY run_id
            """, (employee_id,))
            return [str(r[0]) for r in cur.fetchall()]

    def build_profiles(self, run_id: str, force_target_id: Optional[str] = None) -> Tuple[Optional[EmployeeProfile], List[EmployeeProfile]]:
        """
        Build profiles for a run. If force_target_id is given, that employee
        becomes the target regardless of role_in_run in DB.
        """
        run_emps = self.get_run_employees(run_id)
        if not run_emps:
            log.error(f"No employees found for run_id={run_id}")
            return None, []

        all_skills = self.get_all_employee_skills_for_run(run_id)
        target: Optional[EmployeeProfile] = None
        competitors: List[EmployeeProfile] = []

        for row in run_emps:
            eid = str(row["employee_id"])
            profile = EmployeeProfile(
                employee_id=eid,
                full_name=row["full_name"] or "",
                current_title=row["current_title"] or "",
                company_id=str(row["source_company_id"]) if row["source_company_id"] else None,
                company_name=row["company_name"] or "",
                role_in_run=row["role_in_run"] or "",
                skills=all_skills.get(eid, []),
            )

            # Decide target: explicit employee_id override > DB role_in_run
            if force_target_id:
                if eid == force_target_id:
                    profile.role_in_run = "primary"
                    target = profile
                else:
                    competitors.append(profile)
            elif profile.role_in_run == "primary":
                target = profile
            else:
                competitors.append(profile)

        if not target and not force_target_id:
            # Last resort: treat first employee as target if roles aren't set
            log.warning(f"No TARGET role in run_id={run_id}. Roles found: {[r['role_in_run'] for r in run_emps]}")
            log.warning("Treating first employee as target.")
            all_profiles = [target] if target else []
            all_profiles.extend(competitors)
            if all_profiles:
                target = all_profiles[0]
                target.role_in_run = "primary"
                competitors = all_profiles[1:]

        if not target:
            log.error(f"Could not determine target employee for run_id={run_id}")

        log.info(f"Run {run_id}: target={'YES' if target else 'NO'}, competitors={len(competitors)}")
        return target, competitors


# ---------------------------------------------------------------------------
# Skill Gap Engine
# ---------------------------------------------------------------------------
class FractalGapEngine:

    SKILL_ALIASES = {
        "ms excel": "excel", "microsoft excel": "excel",
        "power point": "powerpoint", "nlp": "natural language processing",
        "gen ai": "generative ai", "py": "python", "js": "javascript",
    }

    SKILL_STOPWORDS = {
        "ms office", "microsoft office", "office", "excel (basic)",
        "proficient in ms office", "communication", "good communication skills",
        "hardworking", "team player", "self starter", "self-starter",
        "open to work", "seeking opportunities",
    }

    SKILL_BLACKLIST = {
        "linkedin", "resume", "curriculum vitae", "cv", "email", "phone", "contact",
    }

    BFSI_WHITELIST = {
        "credit underwriting", "gold loans", "gold loan", "collections",
        "risk modeling", "risk modelling", "credit risk", "fraud detection",
        "lending", "loan origination", "nbfc", "microfinance", "secured lending",
        "unsecured lending", "collections strategy", "portfolio management",
        "asset-liability management", "alm", "treasury", "channel finance",
        "co-lending", "digital lending", "upi", "payments", "fintech",
    }

    GENERIC_TECH_WHITELIST = {
        "python", "java", "javascript", "typescript", "kubernetes", "docker",
        "aws", "azure", "gcp", "microservices", "cloud computing",
        "ci cd", "data engineering", "machine learning", "data science",
        "sql", "nosql", "kafka", "spark",
    }

    MIN_ROLE_SIM = 0.30
    MIN_FREQ_RATIO = 0.03
    MIN_FREQ_COUNT = 1
    MIN_OVERALL_RELEVANCE = 0.10
    TOP_PRIMARY_SKILLS = 40

    def __init__(self, use_llm: bool = True, azure_config: Optional[Dict[str, str]] = None):
        self.use_llm = use_llm
        self.client = None
        self.deployment_id = None
        self.run_cost = RunCost()
        if use_llm:
            self._setup_azure(azure_config or {})

    def _setup_azure(self, cfg: Dict[str, str]):
        if not AzureOpenAI:
            log.warning("openai package not installed; LLM disabled.")
            return
        try:
            self.client = AzureOpenAI(
                api_key=(cfg.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY", "")).strip(),
                azure_endpoint=(cfg.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT", "")).strip().rstrip("/"),
                api_version=(cfg.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")).strip(),
            )
            self.deployment_id = (cfg.get("deployment_id") or os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")).strip()
            log.info(f"Azure OpenAI initialized (deployment={self.deployment_id})")
        except Exception as e:
            log.warning(f"Azure init failed: {e}")
            self.client = None

    @staticmethod
    def normalize_skill(skill: str) -> str:
        if not skill or not isinstance(skill, str):
            return ""
        s = re.sub(r"[^\w\s+#.]", " ", skill.lower()).strip()
        s = re.sub(r"\s+", " ", s)
        return FractalGapEngine.SKILL_ALIASES.get(s, s)

    @staticmethod
    def normalize_role(role: str) -> str:
        if not role:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", role.lower())).strip()

    @staticmethod
    def role_similarity(a: str, b: str) -> float:
        a_n, b_n = FractalGapEngine.normalize_role(a), FractalGapEngine.normalize_role(b)
        if not a_n or not b_n:
            return 0.0
        return SequenceMatcher(None, a_n, b_n).ratio()

    def is_skill_allowed(self, s: str) -> bool:
        s_clean = self.normalize_skill(s)
        if not s_clean or len(s_clean) <= 3:
            return False
        if s_clean in self.SKILL_BLACKLIST or s_clean in self.SKILL_STOPWORDS:
            return False
        return True

    @staticmethod
    def get_role_family(role: str) -> str:
        r = re.sub(r"[^a-z0-9]+", " ", (role or "").lower()).strip()
        families = {
            "exec": ["ceo", "chief executive", "founder", "co founder", "managing director"],
            "product": ["product", "pm", "program manager", "owner"],
            "data": ["data", "ml", "ai", "scientist"],
            "engineering": ["engineer", "developer", "devops", "architect"],
            "risk": ["risk", "credit", "collections"],
            "marketing": ["marketing", "growth", "demand gen", "brand"],
            "sales": ["sales", "business development", "bd"],
            "finance": ["finance", "fp&a", "controller", "accounting"],
            "ops": ["operations", "ops"],
        }
        for family, keywords in families.items():
            if any(k in r for k in keywords):
                return family
        return "generic"

    @staticmethod
    def infer_persona(role: str) -> str:
        r = (role or "").lower()
        if any(k in r for k in ["ceo", "chief executive", "founder", "co-founder", "managing director"]):
            return "ceo"
        if any(k in r for k in ["cto", "chief technology", "vp engineering", "head of engineering"]):
            return "cto"
        if any(k in r for k in ["product head", "head of product", "vp product", "chief product"]):
            return "product_head"
        if any(k in r for k in ["cmo", "marketing head", "growth"]):
            return "marketing"
        return "generic"

    @staticmethod
    def heuristic_person_relevance(persona: str, skill: str) -> float:
        s = (skill or "").lower()
        if persona == "ceo":
            ceo_high = ["governance", "strategic planning", "strategy", "capital",
                        "investor relations", "risk management", "board", "p l", "p&l"]
            ceo_low = ["python", "javascript", "react", "node js", "java", "c++",
                       "html", "css", "kubernetes", "docker"]
            if any(k in s for k in ceo_high):
                return 0.95
            if any(k in s for k in ceo_low):
                return 0.2
            return 0.65
        return 0.6

    def extract_skills(self, profile: EmployeeProfile) -> Set[str]:
        result = set()
        for sk in profile.skills:
            normed = self.normalize_skill(sk.get("skill_name", ""))
            if normed and self.is_skill_allowed(normed):
                result.add(normed)
        return result

    def compute_gaps(self, target: EmployeeProfile, competitors: List[EmployeeProfile]) -> List[SkillGapResult]:
        target_skills = self.extract_skills(target)
        target_role = target.current_title or ""
        persona = self.infer_persona(target_role)
        role_family = self.get_role_family(target_role)

        log.info(f"Target: {target.full_name} | role={target_role} | "
                 f"persona={persona} | family={role_family} | skills={len(target_skills)}")

        # Filter competitors by role similarity
        filtered_comps = [
            c for c in competitors
            if self.role_similarity(target_role, c.current_title or "") >= self.MIN_ROLE_SIM
        ]
        if not filtered_comps and competitors:
            log.warning("Role filtering removed all competitors; using all of them.")
            filtered_comps = competitors

        log.info(f"Competitors: {len(filtered_comps)} (of {len(competitors)} total)")

        # Aggregate competitor skills
        comp_skill_freq: Dict[str, int] = {}
        comp_skill_companies: Dict[str, Set[str]] = {}
        comp_skill_holders: Dict[str, List[str]] = {}

        for c in filtered_comps:
            c_skills = self.extract_skills(c)
            company = c.company_name or "unknown"
            for s in c_skills:
                comp_skill_freq[s] = comp_skill_freq.get(s, 0) + 1
                comp_skill_companies.setdefault(s, set()).add(company)
                comp_skill_holders.setdefault(s, []).append(c.full_name)

        missing = sorted(set(comp_skill_freq.keys()) - target_skills)
        log.info(f"Competitor skill universe: {len(comp_skill_freq)} | "
                 f"Target has: {len(target_skills)} | Raw missing: {len(missing)}")

        if not missing:
            log.info("No skill gaps found — target covers all competitor skills.")
            return []

        # Score each missing skill
        competitor_count = len(filtered_comps)
        scored: List[Tuple[str, float, Dict[str, Any]]] = []

        for s in missing:
            freq_count = comp_skill_freq.get(s, 0)
            freq_ratio = freq_count / competitor_count if competitor_count else 0.0

            if freq_count < self.MIN_FREQ_COUNT or freq_ratio < self.MIN_FREQ_RATIO:
                continue

            person_rel = self.heuristic_person_relevance(persona, s)
            overall = 0.45 * freq_ratio + 0.35 * person_rel + 0.20 * min(1.0, freq_ratio * 2)
            overall = max(0.0, min(1.0, overall))

            if overall < self.MIN_OVERALL_RELEVANCE:
                continue

            if overall >= 0.7:
                tier = "Critical"
            elif overall >= 0.5:
                tier = "Important"
            else:
                tier = "Nice-to-have"

            companies = sorted(comp_skill_companies.get(s, set()))
            meta = {
                "frequency_count": freq_count,
                "frequency_ratio": round(freq_ratio, 3),
                "person_relevance": round(person_rel, 3),
                "overall_relevance": round(overall, 3),
                "gap_tier": tier,
                "role_family": role_family,
                "persona": persona,
                "competitor_count": competitor_count,
                "example_companies": companies[:5],
                "example_holders": comp_skill_holders.get(s, [])[:5],
            }
            scored.append((s, overall, meta))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:self.TOP_PRIMARY_SKILLS]

        if self.client and self.use_llm:
            scored = self._llm_enhance_gaps(target, scored)

        # Build results
        results: List[SkillGapResult] = []
        for skill_name, _, meta in scored:
            companies = meta.get("example_companies", [])
            reasoning = (
                f"Present in {meta['frequency_count']}/{competitor_count} competitors "
                f"(ratio={meta['frequency_ratio']:.0%}). "
                f"Relevance={meta['overall_relevance']:.2f}."
            )
            if companies:
                reasoning += f" Found at: {', '.join(companies[:3])}."

            results.append(SkillGapResult(
                employee_id=target.employee_id,
                skill_id=None,
                skill_gap_name=skill_name,
                skill_importance=meta["gap_tier"],
                gap_reasoning=reasoning,
                competitor_companies=companies,
                raw_json=meta,
            ))

        log.info(f"Final gaps: {len(results)} "
                 f"(Critical={sum(1 for r in results if r.skill_importance == 'Critical')}, "
                 f"Important={sum(1 for r in results if r.skill_importance == 'Important')}, "
                 f"Nice-to-have={sum(1 for r in results if r.skill_importance == 'Nice-to-have')})")
        return results

    def _llm_enhance_gaps(
        self, target: EmployeeProfile, scored: List[Tuple[str, float, Dict[str, Any]]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not scored:
            return scored

        skills_payload = [
            {"skill": s, "frequency_ratio": m["frequency_ratio"],
             "heuristic_relevance": m["overall_relevance"], "tier": m["gap_tier"]}
            for s, _, m in scored[:20]
        ]

        prompt = f"""You are an expert talent analyst. Return ONLY valid JSON.

Target: {target.full_name} - {target.current_title}
Current skills count: {len(target.skills)}

Evaluate these {len(skills_payload)} candidate skill gaps:
{json.dumps(skills_payload, ensure_ascii=False)}

For EACH skill, decide:
1. keep: true/false (is this a REAL gap?)
2. gap_tier: "Critical" | "Important" | "Nice-to-have"
3. adjusted_relevance: 0.0-1.0

Return ONLY this JSON:
{{
  "skills": {{
    "skill_name": {{
      "keep": true,
      "gap_tier": "Critical",
      "adjusted_relevance": 0.8
    }}
  }}
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )

            if tracker:
                usage = getattr(resp, "usage", None)
                if usage:
                    tracker.track_gpt_tokens(
                        int(getattr(usage, "prompt_tokens", 0)),
                        int(getattr(usage, "completion_tokens", 0)),
                    )

            # Record per-run cost
            usage = getattr(resp, "usage", None)
            if usage:
                self.run_cost.record_usage(
                    int(getattr(usage, "prompt_tokens", 0)),
                    int(getattr(usage, "completion_tokens", 0)),
                    model=self.deployment_id or "",
                )

            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw.replace("```json", "").replace("```", "").strip())
            llm_skills = data.get("skills", {})

            enhanced = []
            for skill_name, relevance, meta in scored:
                llm_info = llm_skills.get(skill_name, {})
                if not llm_info.get("keep", True):
                    continue

                if "gap_tier" in llm_info:
                    meta["gap_tier"] = llm_info["gap_tier"]
                if "adjusted_relevance" in llm_info:
                    try:
                        adj = float(llm_info["adjusted_relevance"])
                        if 0.0 <= adj <= 1.0:
                            relevance = 0.5 * relevance + 0.5 * adj
                            meta["overall_relevance"] = round(relevance, 3)
                    except (ValueError, TypeError):
                        pass

                meta["llm_enhanced"] = True
                enhanced.append((skill_name, relevance, meta))

            log.info(f"LLM kept {len(enhanced)}/{len(scored)} gaps")
            return enhanced

        except Exception as e:
            log.warning(f"LLM enhancement failed: {e}; using heuristic scores")
            return scored


# ---------------------------------------------------------------------------
# DB Writer
# ---------------------------------------------------------------------------
class GapWriter:

    def __init__(self, conn):
        self.conn = conn
        self._allowed_importance = self._read_allowed_importance()

    def _read_allowed_importance(self) -> List[str]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT pg_get_constraintdef(oid) FROM pg_constraint
                    WHERE conname = 'employee_skill_gaps_skill_importance_check'
                """)
                row = cur.fetchone()
                if row:
                    vals = re.findall(r"'([^']+)'::text", row[0])
                    if vals:
                        log.info(f"DB allowed skill_importance values: {vals}")
                        return vals
        except Exception as e:
            log.warning(f"Could not read check constraint: {e}")
        return []

    def _map_importance(self, tier: str) -> str:
        allowed = self._allowed_importance
        if not allowed:
            return tier

        t = tier.lower().strip()
        for v in allowed:
            if v.lower() == t:
                return v

        tier_rank = {
            "critical": 0, "high": 0, "must-have": 0,
            "important": 1, "medium": 1, "should-have": 1,
            "nice-to-have": 2, "low": 2, "optional": 2, "nice_to_have": 2,
        }
        input_rank = tier_rank.get(t, 1)
        best, best_dist = None, 999
        for v in allowed:
            dist = abs(tier_rank.get(v.lower(), 1) - input_rank)
            if dist < best_dist:
                best, best_dist = v, dist

        if best:
            return best

        log.warning(f"Could not map importance '{tier}' to {allowed}; using '{allowed[0]}'")
        return allowed[0]

    def resolve_skill_ids(self, results: List[SkillGapResult]) -> List[SkillGapResult]:
        skill_names = list({r.skill_gap_name for r in results})
        reader = RunDataReader(self.conn)
        existing = reader.bulk_get_skill_ids(skill_names)

        missing_names = [n for n in skill_names if n.lower() not in existing]
        if missing_names:
            log.info(f"Creating {len(missing_names)} new skill entries in spectre.skills")
            with self.conn.cursor() as cur:
                for name in missing_names:
                    new_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO spectre.skills (skill_id, name, category, metadata_json, raw_json, created_at)
                        VALUES (%s, %s, %s, '{}'::jsonb, '{}'::jsonb, %s)
                        ON CONFLICT DO NOTHING RETURNING skill_id
                    """, (new_id, name, "auto-detected", datetime.now(timezone.utc)))
                    row = cur.fetchone()
                    if row:
                        existing[name.lower()] = str(row[0])
                    else:
                        fetched = reader.get_skill_id_by_name(name)
                        if fetched:
                            existing[name.lower()] = fetched
            self.conn.commit()

        for r in results:
            r.skill_id = existing.get(r.skill_gap_name.lower())
        return results

    def write_gaps(self, run_id: str, results: List[SkillGapResult],
                   agent_name: str = "agent4_db", model_name: str = "heuristic+llm"):
        if not results:
            log.info("No gaps to write.")
            return

        results = self.resolve_skill_ids(results)
        now = datetime.now(timezone.utc)
        rows = [
            (run_id, r.employee_id, r.skill_id, r.skill_gap_name,
             self._map_importance(r.skill_importance), r.gap_reasoning,
             json.dumps(r.competitor_companies, ensure_ascii=False),
             json.dumps(r.raw_json, ensure_ascii=False),
             agent_name, model_name, now)
            for r in results
        ]

        with self.conn.cursor() as cur:
            employee_id = results[0].employee_id
            cur.execute("""
                DELETE FROM spectre.employee_skill_gaps
                WHERE run_id = %s AND employee_id = %s
            """, (run_id, employee_id))
            deleted = cur.rowcount
            if deleted:
                log.info(f"Cleared {deleted} previous gap rows for this run+employee")

            psycopg2.extras.execute_batch(cur, """
                INSERT INTO spectre.employee_skill_gaps
                    (run_id, employee_id, skill_id, skill_gap_name,
                     skill_importance, gap_reasoning, competitor_companies,
                     raw_json, created_by_agent, model_name, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
            """, rows)

        self.conn.commit()
        log.info(f"Wrote {len(rows)} skill gaps (run_id={run_id}, employee={results[0].employee_id})")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def resolve_run_ids(run_id: Optional[str], employee_id: Optional[str]) -> List[str]:
    """
    Resolve which run_ids to process:
      - run_id only       → [run_id]
      - employee_id only  → all runs where employee is target
      - both              → [run_id]
    """
    if run_id:
        return [run_id]
    if employee_id:
        conn = get_connection()
        try:
            runs = RunDataReader(conn).find_runs_for_employee(employee_id)
            if not runs:
                raise RuntimeError(f"No runs found where employee {employee_id} is target")
            return runs
        finally:
            conn.close()
    raise ValueError("At least one of run_id or employee_id is required")


def run_agent4(run_id: str, use_llm: bool = True, employee_id: Optional[str] = None,
               azure_config: Optional[Dict] = None) -> Tuple[List[SkillGapResult], RunCost]:
    log.info(f"{'=' * 60}")
    log.info(f"Agent 4 — Skill Gap Analysis")
    log.info(f"Run ID: {run_id} | Employee ID: {employee_id or 'auto'}")
    log.info(f"LLM: {'enabled' if use_llm else 'disabled'}")
    log.info(f"{'=' * 60}")

    conn = get_connection()
    try:
        reader = RunDataReader(conn)
        run_info = reader.get_run_info(run_id)
        if not run_info:
            raise RuntimeError(f"Run not found: {run_id}")
        log.info(f"Run status: {run_info.get('status')}, scope: {run_info.get('scope')}")

        target, competitors = reader.build_profiles(run_id, force_target_id=employee_id)
        if not target:
            raise RuntimeError(f"No target employee in run {run_id}")
        if not competitors:
            raise RuntimeError(f"No competitor employees in run {run_id}")

        log.info(f"Target: {target.full_name} — {target.current_title} "
                 f"@ {target.company_name} ({len(target.skills)} skills)")
        for i, c in enumerate(competitors, 1):
            log.info(f"  Comp #{i}: {c.full_name} — {c.current_title} "
                     f"@ {c.company_name} ({len(c.skills)} skills)")

        engine = FractalGapEngine(use_llm=use_llm, azure_config=azure_config)
        gap_results = engine.compute_gaps(target, competitors)
        run_cost = engine.run_cost

        if gap_results:
            log.info(f"--- Top gaps for {target.full_name} ---")
            for i, g in enumerate(gap_results[:10], 1):
                log.info(f"  {i}. [{g.skill_importance}] {g.skill_gap_name}")

        # Log cost summary
        if run_cost.llm_calls > 0:
            log.info(f"LLM cost: {run_cost.llm_calls} call(s), "
                     f"{run_cost.total_tokens} tokens, ${run_cost.cost_usd:.6f}")
        else:
            log.info("LLM cost: $0 (heuristic only)")

        writer = GapWriter(conn)
        writer.write_gaps(
            run_id=run_id,
            results=gap_results,
            agent_name="agent4_db",
            model_name="fractal_heuristic" + ("+llm" if use_llm else ""),
        )

        log.info(f"Agent 4 complete for run {run_id}")
        return gap_results, run_cost
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async adapter (orchestrator compatibility)
# ---------------------------------------------------------------------------
async def run(context: Dict[str, Any]) -> Dict[str, Any]:
    inputs = context.get("inputs", {})
    run_id = inputs.get("run_id")
    employee_id = inputs.get("employee_id")
    if not run_id and not employee_id:
        raise ValueError("run_id or employee_id is required in context.inputs")

    use_llm = inputs.get("use_llm", True)
    azure_config = inputs.get("azure_config")
    db_cfg = inputs.get("db_config", {})
    if db_cfg:
        DB_CONFIG.update({k: v for k, v in db_cfg.items() if v})

    run_ids = resolve_run_ids(run_id, employee_id)
    total_gaps = 0
    total_cost = RunCost()
    for rid in run_ids:
        results, cost = run_agent4(rid, use_llm=use_llm, employee_id=employee_id, azure_config=azure_config)
        total_gaps += len(results) if results else 0
        total_cost.record_usage(cost.prompt_tokens, cost.completion_tokens, cost.model_name)

    context.setdefault("agents", {})
    context["agents"]["agent4"] = {
        "run_ids": run_ids,
        "gaps_found": total_gaps,
        "cost": total_cost.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return context


def run_sync(context: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(run(context))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Agent 4 — Skill Gap Analyzer", version="1.0.0")


class AnalyzeRequest(BaseModel):
    run_id: Optional[str] = None
    employee_id: Optional[str] = None
    use_llm: bool = True


class GapItem(BaseModel):
    skill_gap_name: str
    skill_importance: str
    gap_reasoning: str
    competitor_companies: List[str]
    overall_relevance: Optional[float] = None


class CostInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    model_name: str = ""
    cost_usd: float = 0.0


class AnalyzeResponse(BaseModel):
    run_ids_processed: List[str]
    total_gaps: int
    gaps: Dict[str, List[GapItem]]
    cost: CostInfo


@app.post("/analyze", response_model=AnalyzeResponse)
def api_analyze(req: AnalyzeRequest):
    """
    Run skill gap analysis. Accepts:
      - run_id only       → analyze that specific run
      - employee_id only  → find all runs where employee is target, analyze each
      - both              → analyze the given run_id
    """
    if not req.run_id and not req.employee_id:
        raise HTTPException(status_code=400, detail="At least one of run_id or employee_id is required")

    try:
        run_ids = resolve_run_ids(req.run_id, req.employee_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    all_gaps: Dict[str, List[GapItem]] = {}
    total = 0
    total_cost = RunCost()
    for rid in run_ids:
        try:
            results, cost = run_agent4(rid, use_llm=req.use_llm, employee_id=req.employee_id)
            total_cost.record_usage(cost.prompt_tokens, cost.completion_tokens, cost.model_name)
            items = [
                GapItem(
                    skill_gap_name=r.skill_gap_name,
                    skill_importance=r.skill_importance,
                    gap_reasoning=r.gap_reasoning,
                    competitor_companies=r.competitor_companies,
                    overall_relevance=r.raw_json.get("overall_relevance"),
                )
                for r in results
            ]
            all_gaps[rid] = items
            total += len(items)
        except Exception as e:
            log.error(f"Failed processing run {rid}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing run {rid}: {e}")

    return AnalyzeResponse(
        run_ids_processed=run_ids,
        total_gaps=total,
        gaps=all_gaps,
        cost=CostInfo(**total_cost.to_dict()),
    )


@app.get("/health")
def health():
    return {"status": "ok", "agent": "agent4_db_skill_gap"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Agent 4 — DB Skill Gap Analyzer")
    p.add_argument("--run_id", default=None, help="UUID of the run to analyze")
    p.add_argument("--employee_id", default=None, help="UUID of the employee to analyze")
    p.add_argument("--no_llm", action="store_true", help="Disable LLM-enhanced scoring")
    p.add_argument("--db_host", default=None, help="Override DB host")
    p.add_argument("--db_user", default=None, help="Override DB user")
    p.add_argument("--db_password", default=None, help="Override DB password")
    p.add_argument("--db_name", default=None, help="Override DB name")
    p.add_argument("--serve", action="store_true", help="Start FastAPI server")
    p.add_argument("--port", type=int, default=8004, help="Port for FastAPI server")
    args = p.parse_args()

    if args.db_host:
        DB_CONFIG["host"] = args.db_host
    if args.db_user:
        DB_CONFIG["user"] = args.db_user
    if args.db_password:
        DB_CONFIG["password"] = args.db_password
    if args.db_name:
        DB_CONFIG["dbname"] = args.db_name

    if args.serve:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        if not args.run_id and not args.employee_id:
            p.error("At least one of --run_id or --employee_id is required")
        run_ids = resolve_run_ids(args.run_id, args.employee_id)
        total_cost = RunCost()
        for rid in run_ids:
            _, cost = run_agent4(rid, use_llm=not args.no_llm, employee_id=args.employee_id)
            total_cost.record_usage(cost.prompt_tokens, cost.completion_tokens, cost.model_name)
        if total_cost.llm_calls > 0:
            log.info(f"Total cost across {len(run_ids)} run(s): "
                     f"{total_cost.llm_calls} LLM call(s), "
                     f"{total_cost.total_tokens} tokens, "
                     f"${total_cost.cost_usd:.6f}")
        else:
            log.info(f"Total cost across {len(run_ids)} run(s): $0 (heuristic only)")