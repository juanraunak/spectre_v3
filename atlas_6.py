# -*- coding: utf-8 -*-
"""
Atlas v2 — Spectre Heatmap / Quadrant / Gap-Action Engine
==========================================================

Refactored from Atlas v1 with three major changes:

  1.  ALL reads come from the PostgreSQL DB (spectre schema) — zero JSON file I/O.
  2.  Clusters, X-axis, Y-axis, and peer comparisons are **fully dynamic**
      based on the target's actual role / field / seniority — works for a
      fresh-grad software engineer up to a Fortune-500 CEO.
  3.  Output is a **structured JSON** (not a text prompt) saved to
      `spectre.employee_reports`.

Usage:
    # FastAPI server
    uvicorn atlas_v2:app --host 0.0.0.0 --port 8000 --reload

    # CLI
    python atlas_v2.py --run-id <UUID> --employee-id <UUID>

Required env vars (or pass directly):
    SPECTRE_DB_HOST, SPECTRE_DB_PORT, SPECTRE_DB_NAME,
    SPECTRE_DB_USER, SPECTRE_DB_PASSWORD,
    AZURE_OPENAI_API_KEY          (optional — enables GPT clustering + summaries)
    AZURE_OPENAI_ENDPOINT         (default: https://notedai.openai.azure.com)
    AZURE_OPENAI_DEPLOYMENT_ID    (default: gpt-4o)
    AZURE_OPENAI_API_VERSION      (default: 2024-06-01)
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────
# CONFIG / GLOBALS
# ────────────────────────────────────────────────────────────────

DEBUG = True


def debug(msg: str) -> None:
    if DEBUG:
        print(msg)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ════════════════════════════════════════════════════════════════
#  1.  DATABASE LAYER
# ════════════════════════════════════════════════════════════════

class SpectreDB:
    """
    Thin read-only façade over the spectre Postgres schema.
    Every method returns plain Python dicts/lists — no ORM.
    """

    def __init__(
        self,
        host: str = "",
        port: int = 5432,
        dbname: str = "",
        user: str = "",
        password: str = "",
    ):
        _DEFAULT_DSN = (
            "postgresql://monsteradmin:M0nsteradmin@"
            "monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require"
        )

        if host:
            self.conn_params = dict(
                host=host, port=port,
                dbname=dbname or "postgres",
                user=user or "monsteradmin",
                password=password or "M0nsteradmin",
                sslmode="require",
            )
        else:
            dsn = _env("SPECTRE_DB_URL") or _DEFAULT_DSN
            self.conn_params = dict(dsn=dsn)

        self._conn: Optional[psycopg2.extensions.connection] = None
        debug(f"[DB] Connection mode: {'individual params' if host else 'DSN URL'}")

    # ── connection management ──────────────────────────────────
    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
        return self._conn

    def _q(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _q1(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self._q(sql, params)
        return rows[0] if rows else None

    def _execute(self, sql: str, params: tuple = ()) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── EMPLOYEE (target) ─────────────────────────────────────
    def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        return self._q1(
            """
            SELECT e.*, c.name AS company_name, c.industry AS company_industry,
                   c.business_model, c.description AS company_description
              FROM spectre.employees e
              LEFT JOIN spectre.companies c ON c.company_id = e.current_company_id
             WHERE e.employee_id = %s
            """,
            (employee_id,),
        )

    # ── EMPLOYEES IN A RUN ────────────────────────────────────
    def get_run_employees(self, run_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT re.role_in_run,
                   e.employee_id, e.full_name, e.current_title, e.headline,
                   c.name AS company_name, c.industry AS company_industry
              FROM spectre.run_employees re
              JOIN spectre.employees    e ON e.employee_id = re.employee_id
              LEFT JOIN spectre.companies c ON c.company_id = e.current_company_id
             WHERE re.run_id = %s
             ORDER BY re.role_in_run, e.full_name
            """,
            (run_id,),
        )

    def get_target_employee_for_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._q1(
            """
            SELECT e.employee_id, e.full_name, e.current_title, e.headline,
                   c.name AS company_name, c.industry AS company_industry,
                   c.business_model, c.description AS company_description
              FROM spectre.run_employees re
              JOIN spectre.employees    e ON e.employee_id = re.employee_id
              LEFT JOIN spectre.companies c ON c.company_id = e.current_company_id
             WHERE re.run_id = %s
               AND re.role_in_run = 'primary'
             LIMIT 1
            """,
            (run_id,),
        )

    # ── MATCHES (MIRAGE competitors) ──────────────────────────
    def get_matches_for_employee(
        self, run_id: str, employee_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT em.matched_employee_id, em.matched_name, em.matched_title,
                   em.matched_company_name, em.match_score,
                   em.match_type, em.rationale_json
              FROM spectre.employee_matches em
             WHERE em.run_id = %s
               AND em.employee_id = %s
             ORDER BY em.match_score DESC
             LIMIT %s
            """,
            (run_id, employee_id, limit),
        )

    # ── SKILLS (CIPHER) ──────────────────────────────────────
    def get_skills_for_employee(
        self, run_id: str, employee_id: str
    ) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT s.skill_id, s.name AS skill_name, s.category,
                   es.skill_confidence, es.level, es.rationale_json
              FROM spectre.employee_skills es
              JOIN spectre.skills s ON s.skill_id = es.skill_id
             WHERE es.run_id = %s
               AND es.employee_id = %s
             ORDER BY es.skill_confidence DESC NULLS LAST, s.name
            """,
            (run_id, employee_id),
        )

    def get_skills_for_employee_any_run(
        self, employee_id: str
    ) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT DISTINCT ON (s.skill_id)
                   s.skill_id, s.name AS skill_name, s.category,
                   es.skill_confidence, es.level, es.rationale_json
              FROM spectre.employee_skills es
              JOIN spectre.skills s ON s.skill_id = es.skill_id
             WHERE es.employee_id = %s
             ORDER BY s.skill_id, es.created_at DESC
            """,
            (employee_id,),
        )

    # ── SKILL GAPS (FRACTAL) ─────────────────────────────────
    def get_skill_gaps_for_employee(
        self, run_id: str, employee_id: str
    ) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT sg.skill_id, sg.skill_gap_name, sg.skill_importance,
                   sg.gap_reasoning, sg.competitor_companies, sg.raw_json
              FROM spectre.employee_skill_gaps sg
             WHERE sg.run_id = %s
               AND sg.employee_id = %s
             ORDER BY
               CASE sg.skill_importance
                 WHEN 'Critical'    THEN 1
                 WHEN 'Important'   THEN 2
                 WHEN 'Nice-to-have' THEN 3
                 ELSE 4
               END,
               sg.skill_gap_name
            """,
            (run_id, employee_id),
        )

    # ── COURSES (SPIDER) ─────────────────────────────────────
    def get_courses_for_employee(self, employee_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT ec.course_id, ec.course_name, ec.raw_json
              FROM spectre.employee_courses ec
             WHERE ec.employee_id = %s
             ORDER BY ec.created_at DESC
            """,
            (employee_id,),
        )

    # ── EMPLOYEE DETAILS ──────────────────────────────────────
    def get_employee_details(
        self, run_id: str, employee_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._q1(
            """
            SELECT *
              FROM spectre.employee_details ed
             WHERE ed.run_id = %s
               AND ed.employee_id = %s
             LIMIT 1
            """,
            (run_id, employee_id),
        )

    # ── WRITE: SAVE REPORT ────────────────────────────────────
    def save_employee_report(
        self,
        run_id: str,
        employee_id: str,
        report_json: dict,
        report_type: str = "atlas",
        created_by_agent: str = "atlas_v2",
        model_name: str = "gpt-4o",
        report_version: str = "1",
        created_by_mutation: str = "original",
        mutation_summary: str = "original report",
    ) -> None:
        now = datetime.now(timezone.utc)
        self._execute(
            """
            INSERT INTO spectre.employee_reports
                   (run_id, employee_id, report_json, report_type,
                    created_by_agent, model_name, created_at, updated_at,
                    report_version, created_by_mutation, mutation_summary,
                    version_created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, employee_id, report_type)
            DO UPDATE SET report_json        = EXCLUDED.report_json,
                          created_by_agent   = EXCLUDED.created_by_agent,
                          model_name         = EXCLUDED.model_name,
                          updated_at         = EXCLUDED.updated_at,
                          report_version     = EXCLUDED.report_version,
                          created_by_mutation = EXCLUDED.created_by_mutation,
                          mutation_summary   = EXCLUDED.mutation_summary,
                          version_created_at = EXCLUDED.version_created_at
            """,
            (
                run_id, employee_id,
                json.dumps(report_json, ensure_ascii=False),
                report_type, created_by_agent, model_name, now, now,
                report_version, created_by_mutation, mutation_summary, now,
            ),
        )
        debug(f"[DB] ✅ Saved report type='{report_type}' v{report_version} for employee={employee_id}")


# ════════════════════════════════════════════════════════════════
#  2.  AZURE GPT HELPER
# ════════════════════════════════════════════════════════════════

class AzureGPT:
    """Minimal Azure OpenAI chat wrapper with per-run cost tracking."""

    # ── Pricing per 1M tokens (USD) — update when model changes ──
    # GPT-4o (2024-06-01): $2.50 input / $10.00 output per 1M tokens
    # GPT-4o-mini:          $0.15 input / $0.60 output per 1M tokens
    MODEL_PRICING = {
        "gpt-4o":      {"input_per_1m": 2.50,  "output_per_1m": 10.00},
        "gpt-4o-mini": {"input_per_1m": 0.15,  "output_per_1m": 0.60},
        "gpt-4":       {"input_per_1m": 30.00, "output_per_1m": 60.00},
        "gpt-35-turbo": {"input_per_1m": 0.50, "output_per_1m": 1.50},
    }

    def __init__(self, api_key: str = ""):
        self.api_key = (
            api_key
            or _env("AZURE_OPENAI_API_KEY")
            or "2be1544b3dc14327b60a870fe8b94f35"
        )
        self.endpoint = _env("AZURE_OPENAI_ENDPOINT") or "https://notedai.openai.azure.com"
        self.deployment = _env("AZURE_OPENAI_DEPLOYMENT_ID") or "gpt-4o"
        self.api_version = _env("AZURE_OPENAI_API_VERSION") or "2024-06-01"
        debug(f"[GPT] Key={'✅' if self.api_key else '❌'} | {self.endpoint}/deployments/{self.deployment}")

        # ── Per-run usage accumulators ──
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.call_log: List[Dict[str, Any]] = []

    def reset_usage(self) -> None:
        """Reset token counters — call at the start of each run."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.call_log = []

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _record_usage(self, usage: Dict[str, Any], label: str = "") -> None:
        """Accumulate token counts from an API response's usage block."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_calls += 1
        self.call_log.append({
            "call_number": self.total_calls,
            "label": label,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        debug(
            f"[GPT] 📊 Call #{self.total_calls} ({label}): "
            f"in={prompt_tokens} out={completion_tokens} "
            f"| Running total: in={self.total_input_tokens} out={self.total_output_tokens}"
        )

    def get_cost_summary(self) -> Dict[str, Any]:
        """Return a full cost breakdown for the current run."""
        pricing = self.MODEL_PRICING.get(self.deployment, self.MODEL_PRICING["gpt-4o"])
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input_per_1m"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output_per_1m"]
        total_cost = input_cost + output_cost

        return {
            "model": self.deployment,
            "totalCalls": self.total_calls,
            "inputTokens": self.total_input_tokens,
            "outputTokens": self.total_output_tokens,
            "totalTokens": self.total_input_tokens + self.total_output_tokens,
            "pricing": {
                "inputPer1MTokens_USD": pricing["input_per_1m"],
                "outputPer1MTokens_USD": pricing["output_per_1m"],
            },
            "cost": {
                "inputCost_USD": round(input_cost, 6),
                "outputCost_USD": round(output_cost, 6),
                "totalCost_USD": round(total_cost, 6),
            },
            "callLog": self.call_log,
        }

    def chat(
        self, system: str, user: str,
        max_tokens: int = 1500, temperature: float = 0.1,
        label: str = "",
    ) -> str:
        if not self.available:
            return ""
        try:
            import requests
            url = (
                f"{self.endpoint}/openai/deployments/{self.deployment}"
                f"/chat/completions?api-version={self.api_version}"
            )
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json", "api-key": self.api_key},
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=90,
            )
            if resp.status_code >= 400:
                debug(f"[GPT] ❌ {resp.status_code}: {resp.text[:200]}")
                return ""
            data = resp.json()
            # Track usage
            if "usage" in data:
                self._record_usage(data["usage"], label=label)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            debug(f"[GPT] ❌ {e}")
            return ""

    def chat_json(
        self, system: str, user: str,
        max_tokens: int = 2000, temperature: float = 0.1,
        label: str = "",
    ) -> Optional[Any]:
        raw = self.chat(system, user, max_tokens, temperature, label=label)
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            )
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            debug(f"[GPT] JSON parse error: {e}\nRaw: {raw[:300]}")
            return None


# ════════════════════════════════════════════════════════════════
#  3.  DYNAMIC CLUSTER + AXIS ENGINE
# ════════════════════════════════════════════════════════════════

def _format_person_label(name: str, title: str = "", company: str = "") -> str:
    parts = []
    if title:
        parts.append(title)
    if company:
        parts.append(company)
    suffix = ", ".join(parts)
    return f"{name} ({suffix})" if suffix else name


def _infer_career_domain(title: str, headline: str = "", industry: str = "") -> str:
    blob = f"{title} {headline} {industry}".lower()
    domain_keywords = {
        "engineering": [
            "software", "engineer", "developer", "devops", "sre", "backend",
            "frontend", "full stack", "fullstack", "cloud", "infrastructure",
            "platform", "architect", "sde", "mobile dev",
        ],
        "data": [
            "data scientist", "data engineer", "machine learning", "ml ",
            "ai ", "analytics", "data analyst", "deep learning", "nlp",
        ],
        "product": [
            "product manager", "product lead", "product owner", "product director",
            "product head", "pm ", "growth product",
        ],
        "design": [
            "designer", "ux ", "ui ", "design lead", "creative director",
        ],
        "sales": [
            "sales", "account executive", "business development", "bdr ",
            "sdr ", "account manager", "revenue", "partnerships",
        ],
        "marketing": [
            "marketing", "brand", "content", "growth", "seo", "sem ",
            "demand gen", "communications",
        ],
        "finance": [
            "finance", "cfo", "controller", "accounting", "treasury",
            "fp&a", "investor", "investment",
        ],
        "hr": [
            "hr ", "human resources", "people", "talent", "recruiter",
            "recruiting", "chro", "culture",
        ],
        "operations": [
            "operations", "supply chain", "logistics", "procurement",
            "manufacturing", "coo",
        ],
        "executive": [
            "ceo", "cto", "cmo", "coo", "cfo", "cpo", "chief",
            "founder", "co-founder", "president", "managing director",
            "general manager", "vp ", "vice president", "svp ", "evp ",
            "director", "head of", "board member",
        ],
    }

    best_domain = "general"
    best_score = 0
    for domain, kws in domain_keywords.items():
        score = sum(1 for kw in kws if kw in blob)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def generate_dynamic_clusters_and_axes(
    gpt: AzureGPT,
    target_name: str, target_title: str, target_company: str,
    target_industry: str, target_headline: str,
    all_skills: List[str], career_domain: str,
) -> Dict[str, Any]:
    debug(f"\n[DYNAMIC] Generating clusters for domain='{career_domain}', "
          f"title='{target_title}', skills={len(all_skills)}")

    skills_unique = sorted(set(s.strip() for s in all_skills if s.strip()))[:400]
    skills_text = "\n".join(f"- {s}" for s in skills_unique)

    system_msg = f"""You are an expert career-development analyst.

Your job: given a TARGET person's role, industry, and a pool of skills from them
and their peer competitors, produce TWO meaningful comparison axes and 8-12
skill clusters.

CRITICAL RULES:
- The axes and clusters must be RELEVANT to this person's actual career domain.
  For a software engineer → axes might be "Technical Depth" vs "Engineering Leadership & Delivery".
  For a CEO → axes might be "Strategic Vision & Growth" vs "Operational & Financial Mastery".
  For a salesperson → axes might be "Sales Execution & Client Management" vs "Market Strategy & Revenue Growth".
  For an HR professional → axes might be "People & Culture Strategy" vs "Talent Operations & Compliance".
  ADAPT to the person. DO NOT default to generic "Product/Tech" vs "GTM/Scale".
- Every cluster must contain ONLY skills from the provided skill list. Do not invent skills.
- Each skill should appear in exactly one cluster.
- Aim for 8-12 clusters, each with 3-30 skills.
- Assign each cluster to axis "x", "y", or "neither".
- Ensure at least 2 clusters on X and 2 on Y.

Return ONLY valid JSON (no markdown fences), in this exact shape:
{{
  "x_axis": {{"label": "...", "description": "..."}},
  "y_axis": {{"label": "...", "description": "..."}},
  "clusters": {{
    "Cluster Name": {{
      "axis": "x",
      "skills": ["skill1", "skill2"]
    }},
    ...
  }}
}}"""

    user_msg = f"""TARGET:
  Name: {target_name}
  Title: {target_title}
  Company: {target_company}
  Industry: {target_industry}
  Headline: {target_headline}
  Career Domain: {career_domain}

SKILL POOL ({len(skills_unique)} skills from target + peers):
{skills_text}"""

    result = gpt.chat_json(system_msg, user_msg, max_tokens=3000, temperature=0.1, label="dynamic_clusters")

    if result and isinstance(result, dict) and "clusters" in result:
        clusters = result.get("clusters", {})
        has_x = any(c.get("axis") == "x" for c in clusters.values())
        has_y = any(c.get("axis") == "y" for c in clusters.values())
        if has_x and has_y and len(clusters) >= 3:
            debug(f"[DYNAMIC] ✅ GPT returned {len(clusters)} clusters")
            return result

    debug("[DYNAMIC] ⚠️ GPT unavailable or bad response — using fallback engine")
    return _fallback_dynamic_clusters(skills_unique, career_domain, target_title)


def _fallback_dynamic_clusters(
    all_skills: List[str], career_domain: str, target_title: str,
) -> Dict[str, Any]:
    axis_configs = {
        "engineering": {
            "x_axis": {"label": "Technical Depth & Architecture",
                       "description": "Core engineering skills, system design, coding proficiency"},
            "y_axis": {"label": "Engineering Leadership & Delivery",
                       "description": "Team leadership, project execution, cross-functional impact"},
        },
        "data": {
            "x_axis": {"label": "Data & ML Technical Depth",
                       "description": "ML/AI algorithms, data pipelines, statistical modeling"},
            "y_axis": {"label": "Data Strategy & Business Impact",
                       "description": "Translating data into business value, stakeholder management"},
        },
        "product": {
            "x_axis": {"label": "Product Craft & Technical Acumen",
                       "description": "Product discovery, specs, technical fluency, analytics"},
            "y_axis": {"label": "Go-to-Market & Growth",
                       "description": "Launch strategy, growth, customer insights, market expansion"},
        },
        "sales": {
            "x_axis": {"label": "Sales Execution & Client Management",
                       "description": "Pipeline management, closing, account growth, CRM mastery"},
            "y_axis": {"label": "Market Strategy & Revenue Leadership",
                       "description": "Territory planning, forecasting, team leadership, partnerships"},
        },
        "marketing": {
            "x_axis": {"label": "Marketing Craft & Channels",
                       "description": "Content, SEO/SEM, brand, creative, campaign execution"},
            "y_axis": {"label": "Growth Strategy & Revenue Impact",
                       "description": "Demand gen, funnel optimization, market positioning"},
        },
        "finance": {
            "x_axis": {"label": "Financial Analysis & Reporting",
                       "description": "FP&A, modeling, compliance, accounting standards"},
            "y_axis": {"label": "Strategic Finance & Business Partnering",
                       "description": "Capital allocation, investor relations, M&A, advisory"},
        },
        "hr": {
            "x_axis": {"label": "People Operations & Compliance",
                       "description": "HRIS, payroll, policy, labor law, benefits admin"},
            "y_axis": {"label": "Talent Strategy & Culture",
                       "description": "Employer branding, L&D, org design, DE&I, engagement"},
        },
        "operations": {
            "x_axis": {"label": "Operational Execution & Process",
                       "description": "Supply chain, logistics, lean, quality, vendor management"},
            "y_axis": {"label": "Strategic Planning & Scale",
                       "description": "Capacity planning, P&L ownership, transformation, KPIs"},
        },
        "executive": {
            "x_axis": {"label": "Functional / Domain Expertise",
                       "description": "Deep expertise in core business functions"},
            "y_axis": {"label": "Strategic Leadership & Scale",
                       "description": "Vision, board governance, M&A, org scaling, GTM strategy"},
        },
        "design": {
            "x_axis": {"label": "Design Craft & Technical Skills",
                       "description": "UI/UX, prototyping, visual design, user research"},
            "y_axis": {"label": "Design Leadership & Strategy",
                       "description": "Design systems, team management, product strategy influence"},
        },
    }

    config = axis_configs.get(career_domain, axis_configs["executive"])

    keyword_buckets = {
        "Technical Foundations":       {"axis": "x", "kw": [
            "python", "java", "sql", "javascript", "typescript", "c++", "go", "rust",
            "html", "css", "react", "node", "api", "git", "docker", "kubernetes",
            "aws", "azure", "gcp", "linux", "database", "system design", "architecture",
            "microservices", "ci/cd", "testing", "debugging", "algorithms",
        ]},
        "Data & Analytics":            {"axis": "x", "kw": [
            "data", "analytics", "machine learning", "ai", "deep learning", "nlp",
            "statistics", "tableau", "power bi", "excel", "sql", "etl", "pipeline",
            "tensorflow", "pytorch", "modeling", "visualization", "r ",
        ]},
        "Product & Innovation":        {"axis": "x", "kw": [
            "product", "roadmap", "user research", "a/b testing", "mvp", "feature",
            "backlog", "sprint", "agile", "scrum", "kanban", "jira", "discovery",
            "prototype", "wireframe", "design thinking",
        ]},
        "Sales & Revenue":             {"axis": "y", "kw": [
            "sales", "revenue", "pipeline", "quota", "crm", "salesforce", "hubspot",
            "account management", "client", "customer", "negotiation", "closing",
            "prospecting", "cold call", "demo",
        ]},
        "Marketing & Growth":          {"axis": "y", "kw": [
            "marketing", "brand", "seo", "sem", "content", "social media", "campaign",
            "demand gen", "lead gen", "email marketing", "growth", "conversion",
            "funnel", "acquisition", "retention",
        ]},
        "Strategy & Business":         {"axis": "y", "kw": [
            "strategy", "strategic", "business development", "partnership", "market",
            "competitive", "swot", "go-to-market", "gtm", "expansion", "p&l",
            "business model", "stakeholder",
        ]},
        "Leadership & Management":     {"axis": "y", "kw": [
            "leadership", "management", "team", "mentoring", "coaching", "hiring",
            "performance", "culture", "cross-functional", "executive", "board",
            "decision", "influence", "delegation", "vision",
        ]},
        "Finance & Operations":        {"axis": "x", "kw": [
            "finance", "budget", "forecast", "accounting", "audit", "compliance",
            "operations", "logistics", "supply chain", "procurement", "inventory",
            "process", "lean", "six sigma", "cost",
        ]},
        "Communication & Soft Skills": {"axis": "neither", "kw": [
            "communication", "presentation", "writing", "public speaking",
            "negotiation", "collaboration", "empathy", "adaptability",
            "time management", "problem solving", "critical thinking",
            "interpersonal", "conflict resolution",
        ]},
    }

    clusters: Dict[str, Dict[str, Any]] = {}
    assigned_skills: set = set()

    for cluster_name, bucket in keyword_buckets.items():
        matched = []
        for skill in all_skills:
            if skill.lower() in assigned_skills:
                continue
            if any(kw in skill.lower() for kw in bucket["kw"]):
                matched.append(skill)
                assigned_skills.add(skill.lower())
        if matched:
            clusters[cluster_name] = {"axis": bucket["axis"], "skills": sorted(matched)}

    remaining = [s for s in all_skills if s.lower() not in assigned_skills]
    if remaining:
        clusters["Other Skills"] = {"axis": "neither", "skills": sorted(remaining)}

    axes_present = {v["axis"] for v in clusters.values()}
    names = list(clusters.keys())
    if "x" not in axes_present and names:
        clusters[names[0]]["axis"] = "x"
    if "y" not in axes_present and len(names) > 1:
        clusters[names[-1]]["axis"] = "y"

    return {"x_axis": config["x_axis"], "y_axis": config["y_axis"], "clusters": clusters}


# ════════════════════════════════════════════════════════════════
#  4.  CORE ATLAS ENGINE — heatmap, quadrant, gaps
# ════════════════════════════════════════════════════════════════

class AtlasEngine:
    """Pure computation: takes DB data → produces heatmap, quadrant, gaps."""

    @staticmethod
    def build_heatmap(
        participants: List[Dict[str, Any]],
        clusters: Dict[str, Dict[str, Any]],
        skills_by_person: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        matrix = []
        for cluster_name, cluster_data in clusters.items():
            cluster_skills_lower = {s.lower() for s in cluster_data.get("skills", [])}
            values: Dict[str, int] = {}
            for person in participants:
                eid = person["employee_id"]
                pid = person["id"]
                person_skills_lower = {s.lower() for s in skills_by_person.get(eid, [])}
                match_count = len(cluster_skills_lower & person_skills_lower)
                if match_count == 0:
                    score = 0
                elif match_count <= 2:
                    score = 1
                elif match_count <= 4:
                    score = 2
                else:
                    score = 3
                values[pid] = score
            matrix.append({"cluster": cluster_name, "values": values})
        return matrix

    @staticmethod
    def compute_quadrant(
        participants: List[Dict[str, Any]],
        heatmap_matrix: List[Dict[str, Any]],
        clusters: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        x_clusters = [n for n, c in clusters.items() if c.get("axis") == "x"]
        y_clusters = [n for n, c in clusters.items() if c.get("axis") == "y"]
        heatmap_lookup = {row["cluster"]: row["values"] for row in heatmap_matrix}
        x_max = len(x_clusters) * 3 if x_clusters else 1
        y_max = len(y_clusters) * 3 if y_clusters else 1

        quadrant = []
        for person in participants:
            pid = person["id"]
            x_sum = sum(heatmap_lookup.get(cn, {}).get(pid, 0) for cn in x_clusters)
            y_sum = sum(heatmap_lookup.get(cn, {}).get(pid, 0) for cn in y_clusters)
            x = round((x_sum / x_max) * 10, 1)
            y = round((y_sum / y_max) * 10, 1)

            if x >= 5 and y >= 5:
                qlabel = "Upper-Right"
            elif x < 5 and y >= 5:
                qlabel = "Upper-Left"
            elif x >= 5 and y < 5:
                qlabel = "Lower-Right"
            else:
                qlabel = "Lower-Left"

            rat_parts = []
            if x >= 6 and y >= 6:
                rat_parts.append("Strong across both dimensions — well-rounded profile.")
            elif x >= 6:
                hit = [cn for cn in x_clusters if heatmap_lookup.get(cn, {}).get(pid, 0) >= 2]
                rat_parts.append(f"Strong on X-axis depth ({', '.join(hit[:2]) or 'multiple clusters'}).")
            elif y >= 6:
                hit = [cn for cn in y_clusters if heatmap_lookup.get(cn, {}).get(pid, 0) >= 2]
                rat_parts.append(f"Strong on Y-axis breadth ({', '.join(hit[:2]) or 'multiple clusters'}).")
            if x < 4 and x < 6:
                rat_parts.append("Limited X-axis depth — development opportunity.")
            if y < 4 and y < 6:
                rat_parts.append("Y-axis execution lighter — room to grow.")
            if not rat_parts:
                rat_parts.append("Emerging capabilities across both dimensions.")

            quadrant.append({
                "personId": pid, "personName": person["name"],
                "x": x, "y": y,
                "quadrantLabel": qlabel, "rationale": " ".join(rat_parts),
            })
        return quadrant

    @staticmethod
    def compute_gaps(
        target_pid: str, heatmap_matrix: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        gaps = []
        for row in heatmap_matrix:
            values = row["values"]
            target_score = values.get(target_pid, 0)
            comp_scores = [v for k, v in values.items() if k != target_pid]
            max_comp = max(comp_scores) if comp_scores else 0
            gap = max_comp - target_score
            if gap > 0:
                gaps.append({
                    "cluster": row["cluster"],
                    "targetScore": target_score,
                    "maxScore": max_comp,
                    "gap": gap,
                    "gapPercent": round((gap / 3) * 100, 1),
                })
        gaps.sort(key=lambda g: g["gap"], reverse=True)
        return gaps


# ════════════════════════════════════════════════════════════════
#  5.  GPT-ENHANCED SECTIONS
# ════════════════════════════════════════════════════════════════

def generate_executive_summary(
    gpt: AzureGPT, target_label: str, career_domain: str,
    quadrant_entry: Dict[str, Any], gaps: List[Dict[str, Any]],
    skill_gap_rows: List[Dict[str, Any]],
) -> str:
    if not gpt.available:
        return (
            f"{target_label} shows a mix of strengths and development areas "
            f"across the assessed skill clusters. Key gaps exist in "
            f"{', '.join(g['cluster'] for g in gaps[:3])} relative to peer benchmarks."
        )
    payload = {
        "target": target_label, "career_domain": career_domain,
        "quadrant": quadrant_entry, "top_gaps": gaps[:5],
        "critical_skill_gaps": [
            {"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
             "reason": sg["gap_reasoning"]}
            for sg in skill_gap_rows[:8]
        ],
    }
    text = gpt.chat(
        system=(
            "Write a crisp 2-paragraph executive summary for a skill-gap and "
            "upskilling report. Plain English, no buzzwords. "
            "Reference the person's career domain and quadrant position."
        ),
        user=json.dumps(payload, ensure_ascii=False, indent=2),
        max_tokens=400,
        label="executive_summary",
    )
    return text.strip() if text else ""


def generate_gap_actions(
    gpt: AzureGPT, target_label: str, career_domain: str,
    gaps: List[Dict[str, Any]], skill_gap_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if gpt.available and gaps:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "gaps": gaps[:6],
            "skill_gaps_detail": [
                {"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
                 "reason": sg["gap_reasoning"]}
                for sg in skill_gap_rows[:10]
            ],
        }
        system = (
            "You are an executive coach. For each gap cluster, return 2-4 concrete "
            "actions (who/what/when) and a timeline bucket (0-3 months, 3-9 months, "
            "or 9-18 months). Adapt actions to the person's career domain.\n\n"
            "Return ONLY JSON array:\n"
            '[{"cluster":"...","gap":3,"actions":["..."],"timeline":"0-3 months"}, ...]'
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=1500, label="gap_actions")
        if isinstance(result, list) and result:
            return result

    return [
        {
            "cluster": g["cluster"], "gap": g["gap"],
            "actions": [
                f"Complete a focused learning module on '{g['cluster']}'.",
                f"Apply {g['cluster']} concepts in 2-3 real work initiatives.",
                f"Seek mentorship from a peer who scores higher in {g['cluster']}.",
            ],
            "timeline": "0-3 months" if g["gap"] >= 2 else "3-9 months",
        }
        for g in gaps[:6]
    ]


def generate_course_plans(
    gpt: AzureGPT, target_label: str, career_domain: str,
    skill_gap_rows: List[Dict[str, Any]], existing_courses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    top_gaps = skill_gap_rows[:6]
    if gpt.available and top_gaps:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "skill_gaps": [
                {"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
                 "reason": sg["gap_reasoning"]}
                for sg in top_gaps
            ],
            "existing_courses": [
                {"name": c.get("course_name", ""), "id": str(c.get("course_id", ""))}
                for c in existing_courses[:5]
            ],
        }
        system = (
            "You are a curriculum designer. For each skill gap, design a 3-level "
            "course ladder (beginner / intermediate / advanced). Each level has:\n"
            "- modules: 2-4 short module titles\n"
            "- outcome: what the learner can do after this level\n"
            "- successMetrics: 2-3 measurable indicators\n"
            "- estimatedHours: number\n\n"
            "Adapt content to the person's career domain. Use ONLY the provided data.\n"
            "Return ONLY a JSON array of objects with keys: "
            "skill, importance, reason, levels (with beginner/intermediate/advanced)."
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=2500, label="course_plans")
        if isinstance(result, list) and result:
            return result

    plans = []
    for sg in top_gaps:
        skill = sg["skill_gap_name"]
        plans.append({
            "skill": skill,
            "importance": sg["skill_importance"],
            "reason": sg["gap_reasoning"],
            "levels": {
                "beginner": {
                    "modules": [f"{skill} Foundations", f"Core Concepts of {skill}"],
                    "outcome": f"Understand key concepts and vocabulary of {skill}.",
                    "successMetrics": [
                        f"Complete introductory assessment on {skill}",
                        "Document key learnings in a personal wiki",
                    ],
                    "estimatedHours": 4,
                },
                "intermediate": {
                    "modules": [f"Applying {skill}", f"{skill} Case Studies"],
                    "outcome": f"Apply {skill} in real work scenarios with guidance.",
                    "successMetrics": [
                        f"Complete 2 practical exercises applying {skill}",
                        "Present findings to manager",
                    ],
                    "estimatedHours": 8,
                },
                "advanced": {
                    "modules": [f"Advanced {skill}", f"Leading with {skill}"],
                    "outcome": f"Lead initiatives requiring deep {skill} expertise.",
                    "successMetrics": [
                        f"Drive a cross-functional project leveraging {skill}",
                        "Mentor a junior colleague on this skill",
                    ],
                    "estimatedHours": 12,
                },
            },
        })
    return plans


# ════════════════════════════════════════════════════════════════
#  5b. HELPER BUILDERS
# ════════════════════════════════════════════════════════════════

def build_person_skills_map(
    participants: List[Dict[str, Any]], skills_by_person: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    return {
        p["name"]: sorted(skills_by_person.get(p["employee_id"], []))
        for p in participants
    }


def build_cluster_skill_map(
    clusters: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    return {cname: sorted(cdata.get("skills", [])) for cname, cdata in clusters.items()}


def build_cluster_evidence_by_person(
    participants: List[Dict[str, Any]],
    clusters: Dict[str, Dict[str, Any]],
    skills_by_person: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for p in participants:
        eid = p["employee_id"]
        person_skills_lower = {s.lower(): s for s in skills_by_person.get(eid, [])}
        person_evidence: Dict[str, Any] = {}
        for cname, cdata in clusters.items():
            cluster_skills_lower = {s.lower() for s in cdata.get("skills", [])}
            matched_lower = cluster_skills_lower & set(person_skills_lower.keys())
            if matched_lower:
                matched_original = sorted(person_skills_lower[sl] for sl in matched_lower)
                person_evidence[cname] = {
                    "matchCount": len(matched_original),
                    "matchedSkills": matched_original,
                }
        evidence[p["name"]] = person_evidence
    return evidence


def build_heatmap_markdown(
    heatmap_matrix: List[Dict[str, Any]], participants: List[Dict[str, Any]],
) -> str:
    if not heatmap_matrix or not participants:
        return ""
    pid_to_name = {p["id"]: p["name"] for p in participants}
    pids = [p["id"] for p in participants]
    header_cells = ["Cluster"] + [pid_to_name.get(pid, pid) for pid in pids]
    header_line = "| " + " | ".join(header_cells) + " |"
    sep_line = "|" + "|".join(["---"] + ["---:" for _ in pids]) + "|"
    rows = [header_line, sep_line]
    for row in heatmap_matrix:
        cells = [row["cluster"]] + [str(row["values"].get(pid, 0)) for pid in pids]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


HEATMAP_LEGEND = (
    "0 = no/low evidence (lightest) · "
    "1 = basic (light) · "
    "2 = solid (medium) · "
    "3 = strong (darkest)"
)


def build_quadrant_description(
    quadrant: List[Dict[str, Any]], x_axis_def: Dict[str, str],
    y_axis_def: Dict[str, str], target_label: str,
) -> str:
    lines = [
        f"Scatter plot with X-axis = \"{x_axis_def.get('label', 'Dimension A')}\" "
        f"(0-10) and Y-axis = \"{y_axis_def.get('label', 'Dimension B')}\" (0-10).",
        "Draw a vertical dashed line at X = 5 and a horizontal dashed line at Y = 5 "
        "to create four quadrants:",
        "  • Upper-Right (X≥5, Y≥5): Strong on both dimensions — well-rounded.",
        "  • Upper-Left  (X<5, Y≥5): Strong Y-axis breadth, weaker X-axis depth.",
        "  • Lower-Right (X≥5, Y<5): Strong X-axis depth, lighter Y-axis execution.",
        "  • Lower-Left  (X<5, Y<5): Developing on both — largest growth opportunity.",
        "",
        "Plotted points:",
    ]
    for q in quadrant:
        marker = "★" if target_label in q.get("personName", "") else "●"
        lines.append(f"  {marker} {q['personName']} → ({q['x']}, {q['y']}) [{q['quadrantLabel']}]")
    target_q = next((q for q in quadrant if target_label in q.get("personName", "")), None)
    if target_q:
        lines.append("")
        lines.append(f"Target positioning rationale: {target_q.get('rationale', 'N/A')}")
    return "\n".join(lines)


def generate_presentation_pack(
    gpt: AzureGPT, target_label: str, career_domain: str,
    executive_summary: str, gaps: List[Dict[str, Any]],
    pros_cons: Dict[str, List[str]], gap_actions: List[Dict[str, Any]],
    quadrant_entry: Dict[str, Any], heatmap_matrix: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if gpt.available:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "executive_summary": executive_summary,
            "quadrant": quadrant_entry, "top_gaps": gaps[:5],
            "pros": pros_cons.get("pros", []),
            "cons": pros_cons.get("cons", []),
            "gap_actions": gap_actions[:5],
            "cluster_count": len(heatmap_matrix),
        }
        system = (
            "You are a management consulting slide-deck designer.\n"
            "Create a 10-14 slide executive presentation pack.\n"
            "For each slide provide: title, 3-5 bullet points, and 1-2 speaker notes.\n"
            "Adapt the narrative to the person's career domain.\n"
            "Cover: context/intro, heatmap highlights, quadrant positioning, "
            "key gaps, pros & cons, upskilling plan, 90-day roadmap, next steps.\n\n"
            "Return ONLY JSON:\n"
            '{"slides":[{"title":"...","bullets":["..."],"speakerNotes":["..."]}]}'
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=2500, label="presentation_pack")
        if isinstance(result, dict) and "slides" in result:
            return result

    return {
        "slides": [
            {"title": "Executive Overview", "bullets": [
                f"Target: {target_label}", f"Career domain: {career_domain}",
                f"Peers assessed: {len(heatmap_matrix[0]['values']) - 1 if heatmap_matrix else 'N/A'}",
                "Assessment powered by Spectre Atlas v2",
            ], "speakerNotes": ["Open with who we assessed and why.", "Set context for the career domain and peer set."]},
            {"title": "Executive Summary", "bullets": [
                executive_summary[:200] + "..." if len(executive_summary) > 200 else executive_summary
            ], "speakerNotes": ["Summarise the key takeaway in one sentence."]},
            {"title": "Skill Heatmap Highlights", "bullets": [
                f"{len(heatmap_matrix)} clusters assessed on 0-3 scale",
                "Colour intensity = evidence strength",
                "Focus on red (gap) cells vs green (strength) cells",
            ], "speakerNotes": ["Walk the audience through the heatmap row by row.", "Highlight the 2-3 largest gaps."]},
            {"title": "Quadrant Positioning", "bullets": [
                f"X = {quadrant_entry.get('x', 'N/A')}, Y = {quadrant_entry.get('y', 'N/A')}",
                f"Quadrant: {quadrant_entry.get('quadrantLabel', 'N/A')}",
                quadrant_entry.get("rationale", ""),
            ], "speakerNotes": ["Explain what X and Y axes represent for this career domain.", "Point out where target sits relative to peers."]},
            {"title": "Key Gaps Identified", "bullets": [
                f"{g['cluster']}: gap of {g['gap']} (target {g['targetScore']} vs best {g['maxScore']})"
                for g in gaps[:5]
            ] or ["No significant cluster gaps identified"], "speakerNotes": ["Walk through top 3 gaps and why they matter."]},
            {"title": "Strengths (Pros)", "bullets": [
                f"{p['title']}: {p['description']}" if isinstance(p, dict) else str(p)
                for p in pros_cons.get("pros", [{"title": "N/A", "description": ""}])[:5]
            ], "speakerNotes": ["Celebrate what's working before diving into improvement areas."]},
            {"title": "Development Areas (Cons)", "bullets": [
                f"{c['title']}: {c['description']}" if isinstance(c, dict) else str(c)
                for c in pros_cons.get("cons", [{"title": "N/A", "description": ""}])[:5]
            ], "speakerNotes": ["Frame these as growth opportunities, not weaknesses."]},
            {"title": "Upskilling Actions", "bullets": [
                f"{a['cluster']}: {a['actions'][0]}" if a.get("actions") else a.get("cluster", "")
                for a in gap_actions[:5]
            ] or ["See detailed course plan"], "speakerNotes": ["Highlight the most impactful action per cluster.", "Reference the full course plan for details."]},
            {"title": "90-Day Priority Roadmap", "bullets": [
                "Month 1: Complete foundational modules for top 2 gaps",
                "Month 2: Apply learning in real projects + seek mentor",
                "Month 3: Present progress review + adjust plan",
            ], "speakerNotes": ["This is the quick-win phase — show momentum in 90 days."]},
            {"title": "Next Steps & Recommendations", "bullets": [
                "Assign accountability partner / mentor",
                "Schedule 30/60/90-day check-ins",
                "Re-run Atlas assessment at 6 months to measure progress",
                "Align upskilling with upcoming business objectives",
            ], "speakerNotes": ["Close with clear ownership and timeline.", "Offer to deep-dive on any section."]},
        ]
    }


def generate_roadmap(
    gpt: AzureGPT, target_label: str, career_domain: str,
    gap_actions: List[Dict[str, Any]], course_plans: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    if gpt.available:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "gap_actions": gap_actions[:6],
            "course_plans": [
                {"skill": cp.get("skill", ""), "importance": cp.get("importance", "")}
                for cp in course_plans[:6]
            ],
            "gaps": gaps[:6],
        }
        system = (
            "Create a consolidated 90/180-day upskilling roadmap.\n"
            "Split into two phases: 0-90 days (quick wins + foundations) "
            "and 90-180 days (deeper application + leadership).\n"
            "Each phase should have 4-8 specific, actionable items.\n"
            "Adapt to the career domain. Use only the provided data.\n\n"
            "Return ONLY JSON:\n"
            '{"0_to_90_days":["action1","action2",...],"90_to_180_days":["action1",...]}'
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=800, label="roadmap")
        if isinstance(result, dict) and "0_to_90_days" in result:
            return result

    phase1, phase2 = [], []
    for i, ga in enumerate(gap_actions[:4]):
        cluster = ga.get("cluster", "Unknown")
        actions = ga.get("actions", [])
        timeline = ga.get("timeline", "")
        if "0-3" in timeline or i < 2:
            phase1.extend(f"[{cluster}] {a}" for a in actions[:2])
        else:
            phase2.extend(f"[{cluster}] {a}" for a in actions[:2])

    for cp in course_plans[:3]:
        skill = cp.get("skill", "Unknown")
        levels = cp.get("levels", {})
        if levels.get("beginner", {}).get("modules"):
            phase1.append(f"Complete: {levels['beginner']['modules'][0]} ({skill})")
        if levels.get("intermediate", {}).get("modules"):
            phase2.append(f"Complete: {levels['intermediate']['modules'][0]} ({skill})")

    if not phase1:
        phase1 = ["Identify top 2 skill gaps and enrol in beginner modules",
                   "Set up monthly check-ins with manager",
                   "Review peer benchmarks from heatmap"]
    if not phase2:
        phase2 = ["Apply intermediate learnings in 2-3 real projects",
                   "Present progress to leadership team",
                   "Plan advanced-level modules for next cycle"]
    return {"0_to_90_days": phase1, "90_to_180_days": phase2}


def extract_course_chapters(
    course_rows: List[Dict[str, Any]], skill_gap_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gap_skills = {
        sg["skill_gap_name"].lower(): sg
        for sg in skill_gap_rows if sg.get("skill_gap_name")
    }
    courses_out: List[Dict[str, Any]] = []

    for cr in course_rows:
        course_id = str(cr.get("course_id", ""))
        course_name = cr.get("course_name", "Untitled Course")
        raw = cr.get("raw_json") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}

        chapters: List[Dict[str, Any]] = []
        raw_chapters = (
            raw.get("chapters") or raw.get("topics") or raw.get("modules")
            or raw.get("sections") or raw.get("syllabus")
        )
        if isinstance(raw_chapters, list):
            for ch in raw_chapters:
                if isinstance(ch, dict):
                    ch_title = (
                        ch.get("title") or ch.get("chapter_name") or ch.get("topic_name")
                        or ch.get("name") or ch.get("module_name") or "Untitled Chapter"
                    )
                    lessons_raw = (
                        ch.get("lessons") or ch.get("topics") or ch.get("subtopics")
                        or ch.get("items") or ch.get("content") or []
                    )
                    lessons: List[str] = []
                    if isinstance(lessons_raw, list):
                        for lsn in lessons_raw:
                            if isinstance(lsn, str):
                                lessons.append(lsn)
                            elif isinstance(lsn, dict):
                                lessons.append(
                                    lsn.get("title") or lsn.get("name")
                                    or lsn.get("lesson_name") or lsn.get("topic") or str(lsn)
                                )
                    chapters.append({"title": ch_title, "lessons": lessons})
                elif isinstance(ch, str):
                    chapters.append({"title": ch, "lessons": []})

        if not chapters:
            topic_list = raw.get("topic_names") or raw.get("lesson_names") or []
            if isinstance(topic_list, list) and topic_list:
                chapters.append({"title": course_name, "lessons": [str(t) for t in topic_list if t]})

        duration = raw.get("duration") or raw.get("total_duration") or raw.get("estimated_time") or ""
        if isinstance(duration, (int, float)):
            duration = f"{int(duration)} hours"
        elif not duration:
            total_lessons = sum(len(ch.get("lessons", [])) for ch in chapters)
            duration = f"~{total_lessons * 30} min" if total_lessons else "Self-paced"

        linked_gap = ""
        gap_score = 0
        course_lower = course_name.lower()
        for gap_name, gap_data in gap_skills.items():
            if gap_name in course_lower or any(w in course_lower for w in gap_name.split() if len(w) > 3):
                linked_gap = gap_data["skill_gap_name"]
                importance = (gap_data.get("skill_importance") or "").lower()
                gap_score = 100 if importance == "critical" else 66 if importance == "important" else 33
                break

        course_url = (
            raw.get("url") or raw.get("course_url") or raw.get("link")
            or f"https://mynoted.com/course/{course_id}"
        )

        video_resources: List[Dict[str, str]] = []
        raw_videos = (
            raw.get("videoResources") or raw.get("videos")
            or raw.get("video_links") or raw.get("video_resources") or []
        )
        if isinstance(raw_videos, list):
            for vid in raw_videos:
                if isinstance(vid, dict):
                    video_resources.append({
                        "title": vid.get("title") or vid.get("name") or "Video",
                        "url": vid.get("url") or vid.get("link") or "",
                        "duration": vid.get("duration") or "",
                    })
                elif isinstance(vid, str):
                    video_resources.append({"title": "Video", "url": vid, "duration": ""})

        courses_out.append({
            "courseId": course_id, "courseName": course_name,
            "duration": str(duration), "gapScore": gap_score,
            "skillGapLinked": linked_gap, "url": course_url,
            "chapters": chapters, "videoResources": video_resources,
        })
    return courses_out


# ════════════════════════════════════════════════════════════════
#  5c. UI-SHAPE GENERATORS
# ════════════════════════════════════════════════════════════════

def generate_peer_descriptions(
    gpt: AzureGPT, participants: List[Dict[str, Any]],
    skills_by_person: Dict[str, List[str]], career_domain: str,
    quadrant: List[Dict[str, Any]],
) -> Dict[str, str]:
    quad_lookup = {q["personId"]: q for q in quadrant}
    peer_data = []
    for p in participants:
        q = quad_lookup.get(p["id"], {})
        peer_data.append({
            "id": p["id"], "name": p.get("name", ""),
            "title": p.get("title", ""), "company": p.get("company", ""),
            "isTarget": p.get("isTarget", False),
            "xScore": q.get("x", 0), "yScore": q.get("y", 0),
            "quadrant": q.get("quadrantLabel", ""),
            "topSkills": skills_by_person.get(p["employee_id"], [])[:8],
        })

    if gpt.available and peer_data:
        system = (
            "You generate one-line professional profile descriptions (max 15 words) "
            "for each person based on their title, skills, and quadrant position.\n"
            f"Career domain context: {career_domain}. Adapt language accordingly.\n"
            "For engineers: focus on tech stack, depth, specialization.\n"
            "For executives: focus on strategic scope, industry, leadership style.\n"
            "For sales: focus on market segments, revenue impact, client types.\n\n"
            "Return ONLY JSON: {\"p_id\": \"description\", ...}"
        )
        result = gpt.chat_json(system, json.dumps(peer_data), max_tokens=1000, label="peer_descriptions")
        if isinstance(result, dict) and result:
            debug(f"[PEERS] ✅ GPT generated {len(result)} descriptions")
            return result

    descriptions: Dict[str, str] = {}
    for p in participants:
        q = quad_lookup.get(p["id"], {})
        rationale = q.get("rationale", "")
        if rationale:
            descriptions[p["id"]] = rationale.split(".")[0][:80] + "."
        else:
            descriptions[p["id"]] = f"{p.get('title', 'Professional')} with emerging skill profile."
    return descriptions


def build_swot_structured(
    gpt: AzureGPT, target_label: str, career_domain: str,
    quadrant_entry: Dict[str, Any], gaps: List[Dict[str, Any]],
    heatmap_matrix: List[Dict[str, Any]], target_pid: str,
    skill_gap_rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    strong_clusters, weak_clusters = [], []
    for row in heatmap_matrix:
        target_score = row["values"].get(target_pid, 0)
        comp_scores = [v for k, v in row["values"].items() if k != target_pid]
        avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 0
        if target_score >= 2 and target_score >= avg_comp:
            strong_clusters.append({"cluster": row["cluster"], "score": target_score})
        elif target_score < avg_comp:
            weak_clusters.append({"cluster": row["cluster"], "score": target_score, "peerAvg": round(avg_comp, 1)})

    if gpt.available:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "quadrant": quadrant_entry,
            "strong_clusters": strong_clusters, "weak_clusters": weak_clusters,
            "gaps": gaps[:5],
            "skill_gaps": [
                {"skill": sg["skill_gap_name"], "importance": sg["skill_importance"]}
                for sg in skill_gap_rows[:6]
            ],
        }
        system = (
            "Generate 3-5 pros and 3-5 cons for this person vs their peer set.\n"
            "Each item MUST have a short 'title' (3-6 words) and a 'description' (1-2 sentences).\n"
            f"Adapt to career domain: {career_domain}.\n\n"
            "Return ONLY JSON:\n"
            '{"pros":[{"title":"...","description":"..."}],'
            '"cons":[{"title":"...","description":"..."}]}'
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=1000, label="swot_pros_cons")
        if isinstance(result, dict) and "pros" in result:
            if result["pros"] and isinstance(result["pros"][0], dict) and "title" in result["pros"][0]:
                return result

    pros = [
        {"title": f"Strong in {c['cluster']}", "description": f"Scores {c['score']}/3, at or above peer average."}
        for c in strong_clusters[:4]
    ] or [{"title": "Foundational Skills Present", "description": "Core skills detected across clusters."}]
    cons = [
        {"title": f"Gap in {c['cluster']}", "description": f"Scores {c['score']}/3 vs peer avg {c['peerAvg']}."}
        for c in weak_clusters[:4]
    ] or [{"title": "Broad Development Needed", "description": "Gaps detected across multiple clusters."}]
    return {"pros": pros, "cons": cons}


def build_gaps_with_description_and_category(
    gaps: List[Dict[str, Any]], skill_gap_rows: List[Dict[str, Any]],
    clusters: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    reasoning_by_skill = {}
    for sg in skill_gap_rows:
        name = (sg.get("skill_gap_name") or "").lower()
        reasoning_by_skill[name] = sg.get("gap_reasoning") or ""

    enriched = []
    for g in gaps:
        cluster_name = g["cluster"]
        cluster_data = clusters.get(cluster_name, {})
        axis = cluster_data.get("axis", "neither")
        category = "technical" if axis == "x" else "leadership" if axis == "y" else "general"

        description = ""
        for cs in (s.lower() for s in cluster_data.get("skills", [])):
            if cs in reasoning_by_skill and reasoning_by_skill[cs]:
                description = reasoning_by_skill[cs]
                break
        if not description:
            description = (
                f"Target scores {g['targetScore']}/3 vs best peer at "
                f"{g['maxScore']}/3 in {cluster_name}. "
                f"This represents a {g['gapPercent']}% gap to close."
            )
        enriched.append({**g, "description": description, "category": category})
    return enriched


def _enrich_gap_actions(
    gap_actions: List[Dict[str, Any]], gaps_enriched: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gap_lookup = {g["cluster"]: g for g in gaps_enriched}
    return [
        {
            **ga,
            "description": gap_lookup.get(ga.get("cluster", ""), {}).get(
                "description", f"Development area in {ga.get('cluster', '')}."
            ),
            "category": gap_lookup.get(ga.get("cluster", ""), {}).get("category", "general"),
        }
        for ga in gap_actions
    ]


# ════════════════════════════════════════════════════════════════
#  6.  REPORT BUILDER
# ════════════════════════════════════════════════════════════════

class AtlasReportBuilder:
    """
    Orchestrates the full Atlas pipeline:
      DB reads → dynamic clusters → heatmap → quadrant → gaps →
      GPT enhancements → structured JSON → save to employee_reports.
    """

    def __init__(self, db: SpectreDB, gpt: AzureGPT, run_id: str, employee_id: str, max_peers: int = 10):
        self.db = db
        self.gpt = gpt
        self.run_id = run_id
        self.employee_id = employee_id
        self.max_peers = max_peers

    def build(self) -> Dict[str, Any]:
        # ─── 0. RESET GPT USAGE COUNTERS ─────────────────────
        self.gpt.reset_usage()
        # ─── 1. LOAD TARGET ──────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 1: Loading target from DB ═══")
        target_row = self.db.get_employee(self.employee_id)
        if not target_row:
            raise RuntimeError(f"Employee {self.employee_id} not found in DB")

        target_name = target_row["full_name"] or "Unknown"
        target_title = target_row.get("current_title") or ""
        target_company = target_row.get("company_name") or ""
        target_industry = target_row.get("company_industry") or ""
        target_headline = target_row.get("headline") or ""
        target_label = _format_person_label(target_name, target_title, target_company)
        career_domain = _infer_career_domain(target_title, target_headline, target_industry)
        debug(f"[ATLAS] Target: {target_label} | domain: {career_domain}")

        # ─── 2. LOAD PEERS ───────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 2: Loading peers from DB ═══")
        match_rows = self.db.get_matches_for_employee(self.run_id, self.employee_id, limit=self.max_peers)
        debug(f"[ATLAS] Found {len(match_rows)} matches in DB")

        target_pid = f"p_{target_name.lower().replace(' ', '_')[:20]}"
        participants: List[Dict[str, Any]] = [{
            "id": target_pid, "employee_id": self.employee_id,
            "name": target_label, "isTarget": True,
            "company": target_company, "title": target_title,
        }]

        seen_ids = {self.employee_id}
        for m in match_rows:
            m_eid = str(m.get("matched_employee_id") or "")
            if m_eid in seen_ids or not m_eid:
                continue
            seen_ids.add(m_eid)
            m_name = m.get("matched_name") or "Unknown"
            m_title = m.get("matched_title") or ""
            m_company = m.get("matched_company_name") or ""
            m_label = _format_person_label(m_name, m_title, m_company)
            m_pid = f"p_{m_name.lower().replace(' ', '_')[:20]}"
            counter = 2
            base_pid = m_pid
            while any(p["id"] == m_pid for p in participants):
                m_pid = f"{base_pid}_{counter}"
                counter += 1
            participants.append({
                "id": m_pid, "employee_id": m_eid, "name": m_label,
                "isTarget": False, "company": m_company, "title": m_title,
                "matchScore": m.get("match_score"),
            })
        debug(f"[ATLAS] Total participants: {len(participants)}")

        # ─── 3. LOAD SKILLS ──────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 3: Loading skills from DB ═══")
        skills_by_person: Dict[str, List[str]] = {}
        all_skills_pool: List[str] = []
        for p in participants:
            eid = p["employee_id"]
            skill_rows = self.db.get_skills_for_employee(self.run_id, eid)
            if not skill_rows:
                skill_rows = self.db.get_skills_for_employee_any_run(eid)
            skill_names = [r["skill_name"] for r in skill_rows if r.get("skill_name")]
            skills_by_person[eid] = skill_names
            all_skills_pool.extend(skill_names)
            debug(f"  {p['name'][:30]}: {len(skill_names)} skills")
        all_skills_pool = list(set(all_skills_pool))
        debug(f"[ATLAS] Total unique skills in pool: {len(all_skills_pool)}")

        # ─── 4. DYNAMIC CLUSTERS ─────────────────────────────
        debug("\n[ATLAS] ═══ STEP 4: Dynamic cluster generation ═══")
        cluster_result = generate_dynamic_clusters_and_axes(
            gpt=self.gpt, target_name=target_name, target_title=target_title,
            target_company=target_company, target_industry=target_industry,
            target_headline=target_headline, all_skills=all_skills_pool,
            career_domain=career_domain,
        )
        x_axis_def = cluster_result.get("x_axis", {"label": "Dimension A", "description": ""})
        y_axis_def = cluster_result.get("y_axis", {"label": "Dimension B", "description": ""})
        clusters = cluster_result.get("clusters", {})
        debug(f"[ATLAS] Clusters: {len(clusters)} | X: {x_axis_def['label']} | Y: {y_axis_def['label']}")

        # ─── 5. HEATMAP ──────────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 5: Building heatmap ═══")
        heatmap_matrix = AtlasEngine.build_heatmap(participants, clusters, skills_by_person)

        # ─── 6. QUADRANT ─────────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 6: Computing quadrant ═══")
        quadrant = AtlasEngine.compute_quadrant(participants, heatmap_matrix, clusters)

        # ─── 7. GAPS ─────────────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 7: Computing gaps ═══")
        gaps = AtlasEngine.compute_gaps(target_pid, heatmap_matrix)

        # ─── 8. SKILL GAPS FROM FRACTAL ──────────────────────
        debug("\n[ATLAS] ═══ STEP 8: Loading skill gaps from DB ═══")
        skill_gap_rows = self.db.get_skill_gaps_for_employee(self.run_id, self.employee_id)
        debug(f"[ATLAS] Skill gaps from FRACTAL: {len(skill_gap_rows)}")

        # ─── 9. COURSES FROM SPIDER ──────────────────────────
        debug("\n[ATLAS] ═══ STEP 9: Loading courses from DB ═══")
        course_rows = self.db.get_courses_for_employee(self.employee_id)
        debug(f"[ATLAS] Courses: {len(course_rows)}")

        primary_course_link = None
        if course_rows:
            first = course_rows[0]
            raw = first.get("raw_json") or {}
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            course_url = (
                raw.get("url") or raw.get("course_url") or raw.get("link")
                or f"https://mynoted.com/course/{first.get('course_id', '')}"
            )
            primary_course_link = {"label": first.get("course_name") or "Open Course", "url": course_url}

        # ─── 10. GPT ENHANCEMENTS ────────────────────────────
        debug("\n[ATLAS] ═══ STEP 10: GPT enhancements ═══")
        target_quad = next(
            (q for q in quadrant if q["personId"] == target_pid),
            {"x": 0, "y": 0, "quadrantLabel": "Lower-Left", "rationale": ""},
        )

        executive_summary = generate_executive_summary(
            self.gpt, target_label, career_domain, target_quad, gaps, skill_gap_rows,
        )
        gap_actions = generate_gap_actions(
            self.gpt, target_label, career_domain, gaps, skill_gap_rows,
        )
        course_plans = generate_course_plans(
            self.gpt, target_label, career_domain, skill_gap_rows, course_rows,
        )

        debug("[ATLAS] Generating structured SWOT...")
        pros_cons = build_swot_structured(
            self.gpt, target_label, career_domain, target_quad,
            gaps, heatmap_matrix, target_pid, skill_gap_rows,
        )

        debug("[ATLAS] Generating peer descriptions...")
        peer_descriptions = generate_peer_descriptions(
            self.gpt, participants, skills_by_person, career_domain, quadrant,
        )

        debug("[ATLAS] Extracting course chapters from DB...")
        recommended_courses = extract_course_chapters(course_rows, skill_gap_rows)

        # ─── 11. ASSEMBLE JSON ───────────────────────────────
        debug("\n[ATLAS] ═══ STEP 11: Assembling final JSON ═══")

        quad_lookup = {q["personId"]: q for q in quadrant}

        def _axis_to_field_name(label: str) -> str:
            words = re.sub(r'[^a-zA-Z0-9\s]', '', label).split()
            if not words:
                return "score"
            return words[0].lower() + "".join(w.capitalize() for w in words[1:]) + "Score"

        x_score_field = _axis_to_field_name(x_axis_def.get("label", "xAxis"))
        y_score_field = _axis_to_field_name(y_axis_def.get("label", "yAxis"))
        debug(f"[ATLAS] Dynamic score fields: x={x_score_field}, y={y_score_field}")

        peers_json = []
        for p in participants:
            pid = p["id"]
            q = quad_lookup.get(pid, {})
            peer_entry = {
                "id": pid, "name": p["name"], "displayName": p["name"],
                "isTarget": p.get("isTarget", False), "isUser": p.get("isTarget", False),
                "company": p.get("company", ""), "title": p.get("title", ""),
                "xScore": q.get("x", 0.0), "yScore": q.get("y", 0.0),
                x_score_field: q.get("x", 0.0), y_score_field: q.get("y", 0.0),
                "description": peer_descriptions.get(pid, ""),
            }
            if p.get("matchScore"):
                peer_entry["matchScore"] = p["matchScore"]
            peers_json.append(peer_entry)

        cluster_def = {
            "axes": {"x": x_axis_def, "y": y_axis_def},
            "clusters": [
                {"name": cname, "axis": cdata.get("axis", "neither"), "skills": cdata.get("skills", [])}
                for cname, cdata in clusters.items()
            ],
        }

        skill_gaps_json = [
            {
                "skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
                "reasoning": sg["gap_reasoning"],
                "description": sg["gap_reasoning"] or f"Gap identified in {sg['skill_gap_name']}.",
                "category": (
                    "critical" if (sg.get("skill_importance") or "").lower() == "critical"
                    else "important" if (sg.get("skill_importance") or "").lower() == "important"
                    else "nice-to-have"
                ),
                "competitorCompanies": (
                    sg["competitor_companies"] if isinstance(sg.get("competitor_companies"), list) else []
                ),
            }
            for sg in skill_gap_rows
        ]

        gaps_enriched = build_gaps_with_description_and_category(gaps, skill_gap_rows, clusters)

        debug("[ATLAS] Generating presentation pack...")
        presentation_pack = generate_presentation_pack(
            self.gpt, target_label, career_domain, executive_summary,
            gaps, pros_cons, gap_actions, target_quad, heatmap_matrix,
        )

        debug("[ATLAS] Generating 90/180-day roadmap...")
        roadmap = generate_roadmap(
            self.gpt, target_label, career_domain, gap_actions, course_plans, gaps,
        )

        report = {
            "meta": {
                "targetPerson": target_label,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "version": "2.3", "runId": self.run_id, "employeeId": self.employee_id,
                "careerDomain": career_domain,
                "axisScoreFields": {
                    "x": x_score_field, "y": y_score_field,
                    "xLabel": x_axis_def.get("label", ""), "yLabel": y_axis_def.get("label", ""),
                },
            },
            "targetPerson": target_label,
            "executiveSummary": executive_summary,
            "peers": peers_json,
            "personSkills": build_person_skills_map(participants, skills_by_person),
            "clusterDefinition": cluster_def,
            "clusterSkillMap": build_cluster_skill_map(clusters),
            "clusterEvidenceByPerson": build_cluster_evidence_by_person(participants, clusters, skills_by_person),
            "heatmapMatrix": heatmap_matrix,
            "heatmapMarkdownTable": build_heatmap_markdown(heatmap_matrix, participants),
            "heatmapLegend": HEATMAP_LEGEND,
            "quadrant": quadrant,
            "quadrantPlotDescription": build_quadrant_description(quadrant, x_axis_def, y_axis_def, target_label),
            "gaps": gaps_enriched,
            "skillGaps": skill_gaps_json,
            "gapActions": _enrich_gap_actions(gap_actions, gaps_enriched),
            "recommendedCourses": recommended_courses,
            "coursePlan": course_plans,
            "prosAndCons": pros_cons,
            "experienceSignals": [],
            "goalChips": [],
            "presentationPack": presentation_pack,
            "roadmap": roadmap,
            "primaryCourseLink": primary_course_link or {"label": "Open Course", "url": ""},
            "costSummary": self.gpt.get_cost_summary(),
        }

        # ─── 12. SAVE TO DB ──────────────────────────────────
        debug("\n[ATLAS] ═══ STEP 12: Saving to employee_reports ═══")
        self.db.save_employee_report(
            run_id=self.run_id, employee_id=self.employee_id,
            report_json=report, report_type="atlas",
            created_by_agent="atlas_v2",
            model_name="gpt-4o" if self.gpt.available else "fallback",
        )
        debug("[ATLAS] ✅ Atlas report complete and saved to DB")
        return report


# ════════════════════════════════════════════════════════════════
#  7.  CORE ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run_atlas(
    run_id: str,
    employee_id: str,
    azure_api_key: str = "",
    db_host: str = "",
    db_port: int = 5432,
    db_name: str = "",
    db_user: str = "",
    db_password: str = "",
    max_peers: int = 10,
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("ATLAS v2 — Dynamic Skill Intelligence Engine")
    print("=" * 70)

    db = SpectreDB(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password)
    gpt = AzureGPT(api_key=azure_api_key)

    try:
        builder = AtlasReportBuilder(db=db, gpt=gpt, run_id=run_id, employee_id=employee_id, max_peers=max_peers)
        report = builder.build()

        print(f"\n✅ Atlas report generated for: {report['targetPerson']}")
        print(f"   Peers: {len(report['peers']) - 1}")
        print(f"   Clusters: {len(report['clusterDefinition']['clusters'])}")
        print(f"   Gaps: {len(report['gaps'])}")
        print(f"   Skill Gaps: {len(report['skillGaps'])}")
        print(f"   Course Plans: {len(report['coursePlan'])}")
        print(f"   Recommended Courses (chapters): {len(report['recommendedCourses'])}")
        print(f"   Person Skills: {len(report['personSkills'])} people")
        print(f"   Cluster Evidence: {len(report['clusterEvidenceByPerson'])} people")
        print(f"   Presentation Slides: {len(report.get('presentationPack', {}).get('slides', []))}")
        print(f"   Roadmap phases: {len(report.get('roadmap', {}))}")
        print(f"   SWOT pros: {len(report.get('prosAndCons', {}).get('pros', []))}")
        print(f"   SWOT cons: {len(report.get('prosAndCons', {}).get('cons', []))}")

        cost = report.get("costSummary", {})
        cost_usd = cost.get("cost", {}).get("totalCost_USD", 0)
        print(f"\n   💰 GPT Calls: {cost.get('totalCalls', 0)}")
        print(f"   💰 Tokens: {cost.get('inputTokens', 0)} in / {cost.get('outputTokens', 0)} out / {cost.get('totalTokens', 0)} total")
        print(f"   💰 Estimated Cost: ${cost_usd:.4f} USD ({cost.get('model', 'unknown')})")

        return report
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════
#  8.  FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Atlas v2 — Spectre Skill Intelligence API",
    description="Dynamic heatmap / quadrant / gap-action engine for employee skill assessment.",
    version="2.3",
)


class AtlasRequest(BaseModel):
    """Request body for the Atlas report endpoint."""
    run_id: str = Field(..., description="Spectre run UUID")
    employee_id: Optional[str] = Field(None, description="Target employee UUID (auto-resolved from run if omitted)")
    max_peers: int = Field(10, ge=1, le=50, description="Max peer competitors to include")
    azure_api_key: Optional[str] = Field(None, description="Azure OpenAI API key (optional, overrides env)")
    db_host: Optional[str] = Field(None, description="DB host (optional, overrides env/default DSN)")
    db_port: int = Field(5432, description="DB port")
    db_name: Optional[str] = Field(None, description="DB name")
    db_user: Optional[str] = Field(None, description="DB user")
    db_password: Optional[str] = Field(None, description="DB password")


class AtlasResponse(BaseModel):
    """Wrapper response for the Atlas report."""
    success: bool
    target_person: str
    career_domain: str
    run_id: str
    employee_id: str
    cost_summary: Dict[str, Any] = Field(default_factory=dict, description="GPT token usage and cost breakdown")
    report: Dict[str, Any]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "atlas_v2", "version": "2.3"}


@app.post("/atlas/report", response_model=AtlasResponse)
def generate_atlas_report(req: AtlasRequest):
    """Generate a full Atlas skill-intelligence report for the given employee + run."""
    try:
        employee_id = req.employee_id

        # Auto-resolve employee_id from run if not provided
        if not employee_id:
            db = SpectreDB(
                host=req.db_host or "", port=req.db_port,
                dbname=req.db_name or "", user=req.db_user or "",
                password=req.db_password or "",
            )
            try:
                target_row = db.get_target_employee_for_run(req.run_id)
                if not target_row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No target employee found for run_id '{req.run_id}'. "
                               f"Either pass employee_id explicitly or ensure the run has a target.",
                    )
                employee_id = target_row["employee_id"]
                debug(f"[API] Auto-resolved employee_id={employee_id} from run_id={req.run_id}")
            finally:
                db.close()

        report = run_atlas(
            run_id=req.run_id,
            employee_id=employee_id,
            azure_api_key=req.azure_api_key or "",
            db_host=req.db_host or "",
            db_port=req.db_port,
            db_name=req.db_name or "",
            db_user=req.db_user or "",
            db_password=req.db_password or "",
            max_peers=req.max_peers,
        )
        return AtlasResponse(
            success=True,
            target_person=report.get("targetPerson", ""),
            career_domain=report.get("meta", {}).get("careerDomain", ""),
            run_id=req.run_id,
            employee_id=employee_id,
            cost_summary=report.get("costSummary", {}),
            report=report,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        debug(f"[API] ❌ Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/atlas/report")
def generate_atlas_report_get(
    run_id: str = Query(..., description="Spectre run UUID"),
    employee_id: Optional[str] = Query(None, description="Target employee UUID (auto-resolved from run if omitted)"),
    max_peers: int = Query(10, ge=1, le=50, description="Max peers"),
):
    """GET convenience endpoint — uses default DB/GPT config from env vars."""
    try:
        resolved_employee_id = employee_id

        # Auto-resolve employee_id from run if not provided
        if not resolved_employee_id:
            db = SpectreDB()
            try:
                target_row = db.get_target_employee_for_run(run_id)
                if not target_row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No target employee found for run_id '{run_id}'. "
                               f"Either pass employee_id explicitly or ensure the run has a target.",
                    )
                resolved_employee_id = target_row["employee_id"]
                debug(f"[API] Auto-resolved employee_id={resolved_employee_id} from run_id={run_id}")
            finally:
                db.close()

        report = run_atlas(run_id=run_id, employee_id=resolved_employee_id, max_peers=max_peers)
        return AtlasResponse(
            success=True,
            target_person=report.get("targetPerson", ""),
            career_domain=report.get("meta", {}).get("careerDomain", ""),
            run_id=run_id,
            employee_id=resolved_employee_id,
            cost_summary=report.get("costSummary", {}),
            report=report,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        debug(f"[API] ❌ Unhandled error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# ════════════════════════════════════════════════════════════════
#  9.  CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Atlas v2 — Spectre Skill Intelligence")
    parser.add_argument("--run-id", required=True, help="Spectre run UUID")
    parser.add_argument("--employee-id", required=True, help="Target employee UUID")
    parser.add_argument("--max-peers", type=int, default=10, help="Max peers (default 10)")
    parser.add_argument("--output", default="", help="Also save JSON to this file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug output")
    args = parser.parse_args()

    if args.quiet:
        DEBUG = False

    report = run_atlas(run_id=args.run_id, employee_id=args.employee_id, max_peers=args.max_peers)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Report also saved to: {args.output}")