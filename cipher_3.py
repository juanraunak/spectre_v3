"""
Agent 3 — Database-powered skills extraction with FastAPI.

Modes:
  1. Single employee: extract skills for one employee_id
  2. Batch run: extract skills for all employees in a run_id

Endpoints:
  POST /extract/employee  — body: { "employee_id": "...", "run_id": "..." (optional) }
  POST /extract/run       — body: { "run_id": "..." }

CLI (kept for testing):
  python agent3_db.py --employee_id <UUID>
  python agent3_db.py --run_id <UUID>
"""

import json
import os
import uuid
import time
import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None

# ── Config ───────────────────────────────────────────────────
MAX_SKILLS = 15
MIN_SKILLS_RETRY_THRESHOLD = 5

DB_CONFIG = {
    "host": os.getenv("SPECTRE_DB_HOST", "monsterdb.postgres.database.azure.com"),
    "port": int(os.getenv("SPECTRE_DB_PORT", "5432")),
    "database": os.getenv("SPECTRE_DB_NAME", "postgres"),
    "user": os.getenv("SPECTRE_DB_USER", ""),
    "password": os.getenv("SPECTRE_DB_PASSWORD", ""),
}

# GPT-4o pricing per 1M tokens (adjust if your Azure contract differs)
COST_PER_1M_INPUT_TOKENS = 2.50
COST_PER_1M_OUTPUT_TOKENS = 10.00

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent3_db")


# ── Cost Tracker ─────────────────────────────────────────────
@dataclass
class RunCostTracker:
    """Tracks token usage and cost for a single agent run."""
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    started_at: float = field(default_factory=time.time)

    def record(self, input_tok: int, output_tok: int):
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.llm_calls += 1

    @property
    def input_cost(self) -> float:
        return self.input_tokens * COST_PER_1M_INPUT_TOKENS / 1_000_000

    @property
    def output_cost(self) -> float:
        return self.output_tokens * COST_PER_1M_OUTPUT_TOKENS / 1_000_000

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.started_at
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost_usd": round(self.input_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
            "elapsed_seconds": round(elapsed, 2),
        }


# ── LLM Prompts ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Extract professional skills from the given JSON. Use only the provided evidence; "
    "do not guess tools/languages unless explicitly named. Prefer recent roles; "
    "apply time-decay (ended >7y ×0.70; 3–7y ×0.85; current ×1.0). Output JSON only."
)

USER_INSTRUCTIONS = (
    "You are a senior CV & skill-mapping expert.\n"
    "Extract professional skills from the given JSON profile. "
    "Use only the provided evidence; do not invent tools, domains, or methods.\n\n"
    "Use these fields as evidence:\n"
    "- position\n"
    "- current_company.title and current_company.name\n"
    "- company (top-level)\n"
    "- experience.title, experience.company, experience.location, "
    "experience.description, experience.description_html, experience.start_date, experience.end_date\n"
    "- education.title, education.field, education.degree\n"
    "- certifications.title, honors_and_awards.title\n"
    "- about\n"
    "- courses.title\n"
    "- activity.title, activity.interaction\n"
    "- posts.title, posts.subtitle, posts.text\n"
    "- recommendations.text\n"
    "- match_info.match_rationale\n"
    "- linkedin_skills.name, top_skills.name (treat as hints; keep only if supported by experience/roles)\n\n"
    "Return up to 15 canonical, CV-style skills with scores and one short evidence snippet each.\n"
    "Make each skill a short but specific phrase (2–6 words) that includes domain and context.\n"
    "Examples of good skill labels:\n"
    '- "Regional sales & collections leadership"\n'
    '- "Gold loan risk assessment & underwriting"\n'
    '- "ATL & BTL marketing campaigns"\n'
    '- "Retail branch operations management"\n'
    '- "Customer acquisition & cross-selling"\n'
    'Avoid overly generic labels like just "Leadership", "Communication", "Teamwork".\n\n'
    "Categories: functional | domain | methods | tools/platforms | soft.\n\n"
    "Schema:\n"
    "{\n"
    '  "employee_id": "string",\n'
    '  "name": "string",\n'
    '  "company": "string",\n'
    '  "title": "string",\n'
    '  "skills": [\n'
    "    {\n"
    '      "name":"string",\n'
    '      "category":"functional|domain|methods|tools/platforms|soft",\n'
    '      "score":0.0,\n'
    '      "evidence": {\n'
    '         "field":"position|current_company.title|...",\n'
    '         "snippet":"verbatim text from that field"\n'
    "      }\n"
    "    }\n"
    "  ],\n"
    '  "languages":[{"name":"string","proficiency":"string"}]\n'
    "}\n\n"
    "Rules:\n"
    "- Every skill MUST include evidence.field and evidence.snippet copied verbatim from the provided JSON.\n"
    "- Use the most specific label that reflects the evidence.\n"
    "- If no evidence exists for a candidate skill, omit it.\n"
    "- Prefer recent roles; apply time-decay (ended >7y ×0.70; 3–7y ×0.85; current ×1.0).\n"
    "- LinkedIn skills / top_skills are secondary hints only.\n"
    "- Max 15 skills.\n\n"
)

RETRY_APPEND = (
    "\n\nThe previous response returned too few evidenced skills. "
    "Return additional skills only if you can provide verbatim snippets from the provided JSON fields. "
    "Keep the same schema and rules. Max 15 skills total."
)


# ── Database Manager ─────────────────────────────────────────
class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conn = None

    def connect(self):
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**self.config)
            logger.info("Database connected")
        return self.conn

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def fetch_employees_for_run(self, run_id: str) -> List[str]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT DISTINCT employee_id FROM spectre.run_employees WHERE run_id = %s",
                (run_id,),
            )
            return [str(r["employee_id"]) for r in cur.fetchall()]

    def fetch_employee_data(self, employee_id: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT employee_id, full_name, linkedin_url, canonical_linkedin_id,
                          current_company_id, current_title, location, headline,
                          metadata_json, raw_json, profile_cache_text
                   FROM spectre.employees WHERE employee_id = %s""",
                (employee_id,),
            )
            emp = cur.fetchone()
            if not emp:
                logger.warning(f"Employee not found: {employee_id}")
                return None

            cur.execute(
                """SELECT run_id, employee_id, data_origin, details_json, raw_json,
                          created_by_agent, model_name, v_location, v_company, v_title,
                          v_about, v_name, v_city, v_country_code, v_employee_id,
                          v_connections, v_followers, v_position, v_experience,
                          v_languages, v_education, v_activity, v_courses, v_posts,
                          rp_last_name, rp_first_name, rp_banner_image, rp_avatar,
                          rp_url, rp_linkedin_profile_id
                   FROM spectre.employee_details
                   WHERE employee_id = %s ORDER BY created_at DESC LIMIT 1""",
                (employee_id,),
            )
            details = cur.fetchone()

        return self._merge(emp, details)

    # ── merge helpers ────────────────────────────────────────
    def _merge(self, emp: Dict, details: Optional[Dict]) -> Dict[str, Any]:
        merged = {
            "employee_id": str(emp.get("employee_id", "")),
            "name": emp.get("full_name", ""),
            "linkedin_url": emp.get("linkedin_url", ""),
            "canonical_linkedin_id": emp.get("canonical_linkedin_id", ""),
            "current_company_id": str(emp.get("current_company_id", "")),
            "title": emp.get("current_title", ""),
            "position": emp.get("current_title", ""),
            "location": emp.get("location", ""),
            "headline": emp.get("headline", ""),
        }

        if details:
            merged["about"] = details.get("v_about") or ""
            merged["city"] = details.get("v_city") or ""
            merged["country_code"] = details.get("v_country_code") or ""
            merged["connections"] = details.get("v_connections") or ""
            merged["followers"] = details.get("v_followers") or ""

            for key in ("v_experience", "v_languages", "v_education", "v_activity", "v_courses", "v_posts"):
                merged[key.removeprefix("v_")] = _safe_json(details.get(key))

            merged.update(details.get("details_json") or {})
            merged["company"] = details.get("v_company") or merged.get("company", "")

        # Estimate duration_months where missing
        for exp in merged.get("experience") or []:
            if isinstance(exp, dict) and "duration_months" not in exp:
                exp["duration_months"] = _estimate_months(exp.get("start_date"), exp.get("end_date"))

        merged.setdefault("current_company", {
            "name": merged.get("company", ""),
            "title": merged.get("title", ""),
        })
        return merged

    # ── save skills ──────────────────────────────────────────
    def save_employee_skills(self, employee_id: str, run_id: str, skills_data: List[Dict],
                             created_by_agent: str = "agent3_db", model_name: str = "gpt-4o"):
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                for skill in skills_data:
                    name = (skill.get("name") or "").strip()
                    if not name:
                        continue

                    skill_id = self._ensure_skill(cur, name, skill.get("category", "functional"), skill)

                    cur.execute(
                        """INSERT INTO spectre.employee_skills
                           (run_id, employee_id, skill_id, skill_confidence, level,
                            rationale_json, raw_json, created_by_agent, model_name, created_at)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (run_id, employee_id, skill_id) DO UPDATE SET
                             skill_confidence = EXCLUDED.skill_confidence,
                             level            = EXCLUDED.level,
                             rationale_json   = EXCLUDED.rationale_json,
                             raw_json         = EXCLUDED.raw_json,
                             created_by_agent = EXCLUDED.created_by_agent,
                             model_name       = EXCLUDED.model_name,
                             created_at       = EXCLUDED.created_at""",
                        (run_id, employee_id, skill_id,
                         skill.get("score", 0.0), skill.get("level", "proficient"),
                         Json(skill.get("evidence", {})), Json(skill),
                         created_by_agent, model_name, datetime.utcnow()),
                    )
            conn.commit()
            logger.info(f"Saved {len(skills_data)} skills for {employee_id}")
        except Exception:
            conn.rollback()
            raise

    def _ensure_skill(self, cur, name: str, category: str, metadata: Dict) -> str:
        cur.execute("SELECT skill_id FROM spectre.skills WHERE LOWER(name) = LOWER(%s)", (name,))
        row = cur.fetchone()
        if row:
            return str(row[0])

        skill_id = str(uuid.uuid4())
        cur.execute(
            """INSERT INTO spectre.skills (skill_id, name, category, metadata_json, raw_json, created_at)
               VALUES (%s,%s,%s,%s,%s,%s) RETURNING skill_id""",
            (skill_id, name, category, Json({"extracted_count": 1}), Json(metadata), datetime.utcnow()),
        )
        return skill_id


# ── Skill Normalizer ─────────────────────────────────────────
class SkillNormalizer:
    def __init__(self, client, deployment_id: str, cost: RunCostTracker):
        self.client = client
        self.deployment_id = deployment_id
        self.cost = cost
        self.cache: Dict[str, str] = {}

    def normalize(self, raw_list: List[str]) -> List[str]:
        if not raw_list:
            return []
        self._normalize_batch(raw_list)
        seen, out = set(), []
        for s in raw_list:
            n = self.cache.get(s, s)
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _normalize_batch(self, skills: List[str]):
        to_send = [s for s in skills if s and s not in self.cache]
        if not to_send:
            return

        skills_text = "\n".join(f"- {s}" for s in to_send)
        prompt = (
            "You are a skill normalization expert. Map variants to canonical labels.\n\n"
            "Rules:\n"
            "1. Merge only near-duplicate variants with the same meaning\n"
            "2. Keep languages/frameworks/tools exact\n"
            "3. Preserve descriptive qualifiers and context for business/soft skills\n"
            "4. Title Case, remove meaningless prefixes\n"
            "5. Prefer concise 2–6 word phrases\n"
            '6. Return each mapping as JSON: { "original": "Normalized" }\n\n'
            f"Skills to normalize:\n{skills_text}\n\n"
            "Return ONLY valid JSON."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "You normalize skill labels. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            self._track(resp)
            mapping = json.loads(resp.choices[0].message.content or "{}")
            for orig, norm in mapping.items():
                if isinstance(orig, str) and isinstance(norm, str):
                    self.cache[orig.strip()] = norm.strip()
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            for s in to_send:
                self.cache[s] = " ".join(s.strip().split())

    def _track(self, resp):
        usage = getattr(resp, "usage", None)
        if usage:
            self.cost.record(
                int(getattr(usage, "prompt_tokens", 0)),
                int(getattr(usage, "completion_tokens", 0)),
            )


# ── Skill Extractor ──────────────────────────────────────────
class SkillExtractor:
    def __init__(self, azure_config: Dict[str, str] = None, db_config: Dict[str, str] = None):
        if not AzureOpenAI:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        self.db = DatabaseManager(db_config or DB_CONFIG)

        cfg = azure_config or {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "deployment_id": os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o"),
        }
        self.deployment_id = cfg["deployment_id"]

        missing = [k for k in ("api_key", "endpoint", "api_version") if not cfg.get(k)]
        if missing:
            raise RuntimeError(f"Azure OpenAI config missing: {', '.join(missing)}")

        self.client = AzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            azure_endpoint=cfg["endpoint"],
        )
        logger.info("Azure OpenAI client ready")

    # ── public API ───────────────────────────────────────────
    def process_employee(self, employee_id: str, run_id: str = None) -> Dict[str, Any]:
        run_id = run_id or str(uuid.uuid4())
        cost = RunCostTracker()

        logger.info(f"Processing employee {employee_id} (run {run_id})")

        data = self.db.fetch_employee_data(employee_id)
        if not data:
            return {"success": False, "error": "Employee not found",
                    "employee_id": employee_id, "cost": cost.summary()}

        skills = self._extract(data, cost)
        skills = self._normalize(skills, cost)

        self.db.save_employee_skills(employee_id, run_id, skills,
                                     model_name=self.deployment_id)

        return {
            "success": True,
            "employee_id": employee_id,
            "employee_name": data.get("name"),
            "run_id": run_id,
            "skills_extracted": len(skills),
            "skills": skills,
            "cost": cost.summary(),
        }

    def process_run(self, run_id: str) -> Dict[str, Any]:
        cost = RunCostTracker()
        logger.info(f"Batch processing run {run_id}")

        employee_ids = self.db.fetch_employees_for_run(run_id)
        if not employee_ids:
            return {"success": False, "error": "No employees found for run",
                    "run_id": run_id, "employees_processed": 0, "cost": cost.summary()}

        results, successful, failed = [], 0, 0

        for idx, eid in enumerate(employee_ids, 1):
            logger.info(f"[{idx}/{len(employee_ids)}] Processing {eid}")
            try:
                r = self._process_single_in_batch(eid, run_id, cost)
                results.append(r)
                if r["success"]:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                results.append({"success": False, "employee_id": eid, "error": str(e)})
                logger.error(f"Error processing {eid}: {e}")

        logger.info(f"Run {run_id} complete: {successful} ok, {failed} failed")

        return {
            "success": True,
            "run_id": run_id,
            "employees_total": len(employee_ids),
            "employees_successful": successful,
            "employees_failed": failed,
            "results": results,
            "cost": cost.summary(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def close(self):
        self.db.close()

    # ── internals ────────────────────────────────────────────
    def _process_single_in_batch(self, employee_id: str, run_id: str,
                                  cost: RunCostTracker) -> Dict[str, Any]:
        """Process one employee using a shared cost tracker (for batch runs)."""
        data = self.db.fetch_employee_data(employee_id)
        if not data:
            return {"success": False, "employee_id": employee_id, "error": "Not found"}

        skills = self._extract(data, cost)
        skills = self._normalize(skills, cost)

        self.db.save_employee_skills(employee_id, run_id, skills,
                                     model_name=self.deployment_id)
        return {
            "success": True,
            "employee_id": employee_id,
            "employee_name": data.get("name"),
            "skills_extracted": len(skills),
        }

    def _extract(self, data: Dict, cost: RunCostTracker) -> List[Dict]:
        user_msg = USER_INSTRUCTIONS + json.dumps(data, ensure_ascii=False)
        vitals_lc = _build_vitals_blob(data).lower()

        def call(msg: str) -> List[Dict]:
            resp = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                temperature=0,
                top_p=1,
                response_format={"type": "json_object"},
            )
            usage = getattr(resp, "usage", None)
            if usage:
                cost.record(
                    int(getattr(usage, "prompt_tokens", 0)),
                    int(getattr(usage, "completion_tokens", 0)),
                )

            text = (resp.choices[0].message.content or "").strip()
            try:
                obj = json.loads(text)
                raw = obj.get("skills", []) if isinstance(obj, dict) else []
            except Exception:
                return []

            grounded = []
            for item in raw:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                snippet = (item.get("evidence", {}).get("snippet") or "").strip().lower()
                if snippet and len(snippet) >= 3 and snippet in vitals_lc:
                    grounded.append(item)
            return grounded[:MAX_SKILLS]

        try:
            skills = call(user_msg)
            if len(skills) < MIN_SKILLS_RETRY_THRESHOLD:
                logger.info(f"Only {len(skills)} grounded skills, retrying")
                retry = call(user_msg + RETRY_APPEND)
                seen, merged = set(), []
                for s in skills + retry:
                    key = s.get("name", "").lower()
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(s)
                skills = merged[:MAX_SKILLS]

            logger.info(f"Extracted {len(skills)} grounded skills")
            return skills
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return []

    def _normalize(self, skills: List[Dict], cost: RunCostTracker) -> List[Dict]:
        normalizer = SkillNormalizer(self.client, self.deployment_id, cost)
        raw_names = [s["name"] for s in skills if s.get("name")]
        normed = normalizer.normalize(raw_names)
        out = []
        for i, s in enumerate(skills):
            if i < len(normed):
                s["name"] = normed[i]
                out.append(s)
        return out


# ── Helpers ──────────────────────────────────────────────────
def _safe_json(val):
    if not val:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


def _estimate_months(start: Optional[str], end: Optional[str]) -> int:
    try:
        if not end or (isinstance(end, str) and "present" in end.lower()):
            ey, em = 2025, 1
        else:
            ey, em = int(str(end).split()[-1]), 6

        if not start:
            return 12
        sy, sm = int(str(start).split()[-1]), 6
        return max(1, (ey - sy) * 12 + (em - sm))
    except Exception:
        return 12


def _build_vitals_blob(data: Dict) -> str:
    parts = [
        data.get("position", ""), data.get("title", ""),
        data.get("company", ""), data.get("about", ""), data.get("headline", ""),
    ]
    cc = data.get("current_company") or {}
    parts += [cc.get("title", ""), cc.get("name", "")]

    for exp in data.get("experience") or []:
        if isinstance(exp, dict):
            parts += [exp.get("title", ""), exp.get("company", ""),
                      exp.get("description", ""), exp.get("description_html", "")]

    for ed in data.get("education") or []:
        if isinstance(ed, dict):
            parts += [ed.get("title", ""), ed.get("field", ""), ed.get("degree", "")]

    for field_name in ("certifications", "honors_and_awards", "courses", "activity", "posts"):
        for item in data.get(field_name) or []:
            if isinstance(item, dict):
                parts += [item.get("title", ""), item.get("text", "")]

    return " ".join(p for p in parts if p)


# ── FastAPI App ──────────────────────────────────────────────
if FastAPI:
    app = FastAPI(title="Agent 3 — Skill Extraction", version="2.0")

    class EmployeeRequest(BaseModel):
        employee_id: str
        run_id: Optional[str] = None

    class RunRequest(BaseModel):
        run_id: str

    @app.post("/skills/employee")
    def extract_employee(req: EmployeeRequest):
        extractor = SkillExtractor()
        try:
            result = extractor.process_employee(req.employee_id, req.run_id)
            if not result["success"]:
                raise HTTPException(status_code=404, detail=result["error"])
            return result
        finally:
            extractor.close()

    @app.post("/skills/run")
    def extract_run(req: RunRequest):
        extractor = SkillExtractor()
        try:
            result = extractor.process_run(req.run_id)
            if not result["success"]:
                raise HTTPException(status_code=404, detail=result["error"])
            return result
        finally:
            extractor.close()

    @app.get("/health")
    def health():
        return {"status": "ok", "agent": "agent3_db", "version": "2.0"}

else:
    app = None


# ── CLI ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Agent 3 — Skill extraction")
    parser.add_argument("--employee_id", help="Process single employee by UUID")
    parser.add_argument("--run_id", help="Process all employees in a run")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    args = parser.parse_args()

    if args.serve:
        if not app:
            print("FastAPI not installed. Run: pip install fastapi uvicorn")
            return
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if not args.employee_id and not args.run_id:
        parser.error("one of --employee_id or --run_id is required (or use --serve)")

    extractor = SkillExtractor()
    try:
        if args.employee_id:
            result = extractor.process_employee(args.employee_id)
        else:
            result = extractor.process_run(args.run_id)
        print(json.dumps(result, indent=2))
    finally:
        extractor.close()


if __name__ == "__main__":
    main()