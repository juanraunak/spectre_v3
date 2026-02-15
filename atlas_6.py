# -*- coding: utf-8 -*-
"""
Atlas v2.4 — Multi-Axis Spectre Heatmap / Quadrant / Gap-Action Engine
=======================================================================

Changes from v2.3:
  - 7 X-axes + 7 Y-axes generated per career domain (market-projected)
  - All 49 axis combinations pre-computed with per-person scores
  - Quadrant scoring is NOW INDEPENDENT of skill clusters
    → GPT evaluates each person directly on each axis (0-10)
  - Frontend dropdown can switch axes; "Recalculate Position" uses pre-computed data
  - Backward-compatible: default combo still populates legacy `quadrant` and `peers[].xScore/yScore`

Usage:
    uvicorn atlas_v2:app --host 0.0.0.0 --port 8000 --reload
    python atlas_v2.py --run-id <UUID> --employee-id <UUID>
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
NUM_AXES = 7  # 7 X-axes and 7 Y-axes → 49 combinations


def debug(msg: str) -> None:
    if DEBUG:
        print(msg)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ════════════════════════════════════════════════════════════════
#  1.  DATABASE LAYER
# ════════════════════════════════════════════════════════════════

class SpectreDB:
    """Thin read-only facade over the spectre Postgres schema."""

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

    def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        return self._q1(
            """
            SELECT e.*, c.name AS company_name, c.industry AS company_industry,
                   c.business_model, c.description AS company_description
              FROM spectre.employees e
              LEFT JOIN spectre.companies c ON c.company_id = e.current_company_id
             WHERE e.employee_id = %s
            """, (employee_id,),
        )

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
            """, (run_id,),
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
             WHERE re.run_id = %s AND re.role_in_run = 'primary'
             LIMIT 1
            """, (run_id,),
        )

    def get_matches_for_employee(self, run_id: str, employee_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT em.matched_employee_id, em.matched_name, em.matched_title,
                   em.matched_company_name, em.match_score,
                   em.match_type, em.rationale_json
              FROM spectre.employee_matches em
             WHERE em.run_id = %s AND em.employee_id = %s
             ORDER BY em.match_score DESC LIMIT %s
            """, (run_id, employee_id, limit),
        )

    def get_skills_for_employee(self, run_id: str, employee_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT s.skill_id, s.name AS skill_name, s.category,
                   es.skill_confidence, es.level, es.rationale_json
              FROM spectre.employee_skills es
              JOIN spectre.skills s ON s.skill_id = es.skill_id
             WHERE es.run_id = %s AND es.employee_id = %s
             ORDER BY es.skill_confidence DESC NULLS LAST, s.name
            """, (run_id, employee_id),
        )

    def get_skills_for_employee_any_run(self, employee_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT DISTINCT ON (s.skill_id)
                   s.skill_id, s.name AS skill_name, s.category,
                   es.skill_confidence, es.level, es.rationale_json
              FROM spectre.employee_skills es
              JOIN spectre.skills s ON s.skill_id = es.skill_id
             WHERE es.employee_id = %s
             ORDER BY s.skill_id, es.created_at DESC
            """, (employee_id,),
        )

    def get_skill_gaps_for_employee(self, run_id: str, employee_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT sg.skill_id, sg.skill_gap_name, sg.skill_importance,
                   sg.gap_reasoning, sg.competitor_companies, sg.raw_json
              FROM spectre.employee_skill_gaps sg
             WHERE sg.run_id = %s AND sg.employee_id = %s
             ORDER BY
               CASE sg.skill_importance
                 WHEN 'Critical' THEN 1 WHEN 'Important' THEN 2
                 WHEN 'Nice-to-have' THEN 3 ELSE 4 END,
               sg.skill_gap_name
            """, (run_id, employee_id),
        )

    def get_courses_for_employee(self, employee_id: str) -> List[Dict[str, Any]]:
        return self._q(
            """
            SELECT ec.course_id, ec.course_name, ec.raw_json
              FROM spectre.employee_courses ec
             WHERE ec.employee_id = %s ORDER BY ec.created_at DESC
            """, (employee_id,),
        )

    def get_employee_details(self, run_id: str, employee_id: str) -> Optional[Dict[str, Any]]:
        return self._q1(
            "SELECT * FROM spectre.employee_details ed WHERE ed.run_id = %s AND ed.employee_id = %s LIMIT 1",
            (run_id, employee_id),
        )

    def save_employee_report(
        self, run_id: str, employee_id: str, report_json: dict,
        report_type: str = "atlas", created_by_agent: str = "atlas_v2",
        model_name: str = "gpt-4o", report_version: str = "1",
        created_by_mutation: str = "original", mutation_summary: str = "original report",
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
            DO UPDATE SET report_json = EXCLUDED.report_json,
                          created_by_agent = EXCLUDED.created_by_agent,
                          model_name = EXCLUDED.model_name,
                          updated_at = EXCLUDED.updated_at,
                          report_version = EXCLUDED.report_version,
                          created_by_mutation = EXCLUDED.created_by_mutation,
                          mutation_summary = EXCLUDED.mutation_summary,
                          version_created_at = EXCLUDED.version_created_at
            """,
            (run_id, employee_id, json.dumps(report_json, ensure_ascii=False),
             report_type, created_by_agent, model_name, now, now,
             report_version, created_by_mutation, mutation_summary, now),
        )
        debug(f"[DB] Saved report type='{report_type}' v{report_version} for employee={employee_id}")


# ════════════════════════════════════════════════════════════════
#  2.  AZURE GPT HELPER
# ════════════════════════════════════════════════════════════════

class AzureGPT:
    """Minimal Azure OpenAI chat wrapper with per-run cost tracking."""

    MODEL_PRICING = {
        "gpt-4o":       {"input_per_1m": 2.50,  "output_per_1m": 10.00},
        "gpt-4o-mini":  {"input_per_1m": 0.15,  "output_per_1m": 0.60},
        "gpt-4":        {"input_per_1m": 30.00, "output_per_1m": 60.00},
        "gpt-35-turbo": {"input_per_1m": 0.50,  "output_per_1m": 1.50},
    }

    def __init__(self, api_key: str = ""):
        self.api_key = (
            api_key or _env("AZURE_OPENAI_API_KEY") or "2be1544b3dc14327b60a870fe8b94f35"
        )
        self.endpoint = _env("AZURE_OPENAI_ENDPOINT") or "https://notedai.openai.azure.com"
        self.deployment = _env("AZURE_OPENAI_DEPLOYMENT_ID") or "gpt-4o"
        self.api_version = _env("AZURE_OPENAI_API_VERSION") or "2024-06-01"
        debug(f"[GPT] Key={'set' if self.api_key else 'missing'} | {self.endpoint}/deployments/{self.deployment}")
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.call_log: List[Dict[str, Any]] = []

    def reset_usage(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.call_log = []

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _record_usage(self, usage: Dict[str, Any], label: str = "") -> None:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self.total_calls += 1
        self.call_log.append({
            "call_number": self.total_calls, "label": label,
            "input_tokens": prompt_tokens, "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        debug(f"[GPT] Call #{self.total_calls} ({label}): in={prompt_tokens} out={completion_tokens}")

    def get_cost_summary(self) -> Dict[str, Any]:
        pricing = self.MODEL_PRICING.get(self.deployment, self.MODEL_PRICING["gpt-4o"])
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input_per_1m"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output_per_1m"]
        return {
            "model": self.deployment, "totalCalls": self.total_calls,
            "inputTokens": self.total_input_tokens,
            "outputTokens": self.total_output_tokens,
            "totalTokens": self.total_input_tokens + self.total_output_tokens,
            "pricing": {"inputPer1MTokens_USD": pricing["input_per_1m"],
                        "outputPer1MTokens_USD": pricing["output_per_1m"]},
            "cost": {"inputCost_USD": round(input_cost, 6),
                     "outputCost_USD": round(output_cost, 6),
                     "totalCost_USD": round(input_cost + output_cost, 6)},
            "callLog": self.call_log,
        }

    def chat(self, system: str, user: str, max_tokens: int = 1500,
             temperature: float = 0.1, label: str = "") -> str:
        if not self.available:
            return ""
        try:
            import requests
            url = (f"{self.endpoint}/openai/deployments/{self.deployment}"
                   f"/chat/completions?api-version={self.api_version}")
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json", "api-key": self.api_key},
                json={"messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=90,
            )
            if resp.status_code >= 400:
                debug(f"[GPT] Error {resp.status_code}: {resp.text[:200]}")
                return ""
            data = resp.json()
            if "usage" in data:
                self._record_usage(data["usage"], label=label)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            debug(f"[GPT] Error: {e}")
            return ""

    def chat_json(self, system: str, user: str, max_tokens: int = 2000,
                  temperature: float = 0.1, label: str = "") -> Optional[Any]:
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
#  3.  DYNAMIC CLUSTER ENGINE (for heatmap only)
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
        "engineering": ["software", "engineer", "developer", "devops", "sre", "backend",
                        "frontend", "full stack", "fullstack", "cloud", "infrastructure",
                        "platform", "architect", "sde", "mobile dev"],
        "data": ["data scientist", "data engineer", "machine learning", "ml ", "ai ",
                 "analytics", "data analyst", "deep learning", "nlp"],
        "product": ["product manager", "product lead", "product owner", "product director",
                    "product head", "pm ", "growth product"],
        "design": ["designer", "ux ", "ui ", "design lead", "creative director"],
        "sales": ["sales", "account executive", "business development", "bdr ", "sdr ",
                  "account manager", "revenue", "partnerships"],
        "marketing": ["marketing", "brand", "content", "growth", "seo", "sem ",
                      "demand gen", "communications"],
        "finance": ["finance", "cfo", "controller", "accounting", "treasury", "fp&a",
                    "investor", "investment"],
        "hr": ["hr ", "human resources", "people", "talent", "recruiter", "recruiting",
               "chro", "culture"],
        "operations": ["operations", "supply chain", "logistics", "procurement",
                       "manufacturing", "coo"],
        "executive": ["ceo", "cto", "cmo", "coo", "cfo", "cpo", "chief", "founder",
                      "co-founder", "president", "managing director", "general manager",
                      "vp ", "vice president", "svp ", "evp ", "director", "head of",
                      "board member"],
    }
    best_domain, best_score = "general", 0
    for domain, kws in domain_keywords.items():
        score = sum(1 for kw in kws if kw in blob)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def generate_dynamic_clusters(
    gpt: AzureGPT,
    target_name: str, target_title: str, target_company: str,
    target_industry: str, target_headline: str,
    all_skills: List[str], career_domain: str,
) -> Dict[str, Any]:
    """Generate skill clusters for the HEATMAP only. Axes are separate."""
    debug(f"\n[CLUSTERS] Generating clusters for domain='{career_domain}', skills={len(all_skills)}")
    skills_unique = sorted(set(s.strip() for s in all_skills if s.strip()))[:400]
    skills_text = "\n".join(f"- {s}" for s in skills_unique)

    system_msg = f"""You are an expert career-development analyst.

Given a TARGET person's role, industry, and a pool of skills, produce 8-12 skill clusters.

RULES:
- Clusters must be RELEVANT to this person's career domain ({career_domain}).
- Every cluster must contain ONLY skills from the provided skill list. Do not invent skills.
- Each skill should appear in exactly one cluster.
- Aim for 8-12 clusters, each with 2-30 skills.

Return ONLY valid JSON (no markdown fences):
{{
  "clusters": {{
    "Cluster Name": {{
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

SKILL POOL ({len(skills_unique)} skills):
{skills_text}"""

    result = gpt.chat_json(system_msg, user_msg, max_tokens=3000, temperature=0.1,
                           label="dynamic_clusters")
    if result and isinstance(result, dict) and "clusters" in result:
        clusters = result["clusters"]
        if len(clusters) >= 3:
            debug(f"[CLUSTERS] GPT returned {len(clusters)} clusters")
            return {"clusters": clusters}

    debug("[CLUSTERS] GPT fallback — using keyword-based clustering")
    return _fallback_dynamic_clusters(skills_unique, career_domain)


def _fallback_dynamic_clusters(all_skills: List[str], career_domain: str) -> Dict[str, Any]:
    keyword_buckets = {
        "Technical Foundations":       {"kw": ["python", "java", "sql", "javascript", "typescript",
            "c++", "go", "rust", "html", "css", "react", "node", "api", "git", "docker",
            "kubernetes", "aws", "azure", "gcp", "linux", "database", "system design",
            "architecture", "microservices", "ci/cd", "testing", "debugging", "algorithms"]},
        "Data & Analytics":            {"kw": ["data", "analytics", "machine learning", "ai",
            "deep learning", "nlp", "statistics", "tableau", "power bi", "excel", "etl",
            "pipeline", "tensorflow", "pytorch", "modeling", "visualization"]},
        "Product & Innovation":        {"kw": ["product", "roadmap", "user research", "a/b testing",
            "mvp", "feature", "backlog", "sprint", "agile", "scrum", "kanban", "jira",
            "discovery", "prototype", "wireframe", "design thinking"]},
        "Sales & Revenue":             {"kw": ["sales", "revenue", "pipeline", "quota", "crm",
            "salesforce", "hubspot", "account management", "client", "customer", "negotiation",
            "closing", "prospecting"]},
        "Marketing & Growth":          {"kw": ["marketing", "brand", "seo", "sem", "content",
            "social media", "campaign", "demand gen", "lead gen", "growth", "conversion",
            "funnel", "acquisition", "retention"]},
        "Strategy & Business":         {"kw": ["strategy", "strategic", "business development",
            "partnership", "market", "competitive", "go-to-market", "gtm", "expansion",
            "p&l", "business model", "stakeholder"]},
        "Leadership & Management":     {"kw": ["leadership", "management", "team", "mentoring",
            "coaching", "hiring", "performance", "culture", "cross-functional", "executive",
            "board", "decision", "influence", "delegation", "vision"]},
        "Finance & Operations":        {"kw": ["finance", "budget", "forecast", "accounting",
            "audit", "compliance", "operations", "logistics", "supply chain", "procurement",
            "inventory", "process", "lean", "six sigma", "cost"]},
        "Communication & Soft Skills": {"kw": ["communication", "presentation", "writing",
            "public speaking", "collaboration", "empathy", "adaptability", "time management",
            "problem solving", "critical thinking", "interpersonal", "conflict resolution"]},
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
            clusters[cluster_name] = {"skills": sorted(matched)}

    remaining = [s for s in all_skills if s.lower() not in assigned_skills]
    if remaining:
        clusters["Other Skills"] = {"skills": sorted(remaining)}
    return {"clusters": clusters}


# ════════════════════════════════════════════════════════════════
#  4.  MULTI-AXIS MARKET PROJECTION ENGINE  ★ NEW in v2.4 ★
# ════════════════════════════════════════════════════════════════

def generate_market_projection_axes(
    gpt: AzureGPT,
    target_title: str, target_company: str,
    target_industry: str, target_headline: str,
    career_domain: str,
) -> Dict[str, Any]:
    """
    Generate 7 X-axes and 7 Y-axes that represent MARKET-PROJECTED dimensions.
    These are INDEPENDENT of skill clusters — they represent where this role
    is heading in today's job market.
    """
    debug(f"\n[AXES] Generating {NUM_AXES}x{NUM_AXES} market-projected axes")

    if gpt.available:
        system_msg = f"""You are an expert career strategist and labor market analyst
specializing in how AI is transforming every job function in 2025-2026.

Given a person's role and industry, generate exactly {NUM_AXES} X-axis dimensions
and exactly {NUM_AXES} Y-axis dimensions for where this role is heading.

MANDATORY AI/EMERGING TECH REQUIREMENTS:
- At LEAST 2 of the {NUM_AXES} X-axes MUST be AI/emerging-technology axes that
  are specific to HOW AI is changing THIS EXACT ROLE. These are NOT generic
  "AI awareness" axes — they must be about concrete AI skills for this role:

  For a Software Engineer:
    → "AI-Assisted Development & Copilot Fluency" (using Copilot, Cursor, AI code review)
    → "ML/AI Engineering & Model Deployment" (building/fine-tuning/deploying models)
    → "Prompt Engineering & LLM Integration" (building AI features into products)

  For a CEO:
    → "AI Strategy & Organizational AI Adoption" (leading AI transformation)
    → "AI-Augmented Decision Making" (using AI for forecasting, ops, strategy)

  For Sales:
    → "AI-Powered Prospecting & Sales Intelligence" (AI lead scoring, outreach tools)
    → "Conversational AI & Sales Automation" (chatbots, AI demos, automated follow-ups)

  For HR:
    → "AI in Talent Acquisition & Screening" (AI-driven recruitment, bias detection)
    → "AI Workforce Planning & Skills Analytics" (predictive workforce analytics)

  For Marketing:
    → "Generative AI Content & Campaigns" (AI copy, images, video production)
    → "AI-Driven Analytics & Personalization" (predictive targeting, dynamic content)

  For Data Scientists:
    → "LLM Fine-Tuning & RAG Systems" (production LLM pipelines)
    → "MLOps & Model Governance" (deployment, monitoring, responsible AI)

  For Finance:
    → "AI-Powered Financial Modeling" (ML forecasting, anomaly detection)
    → "Automated Compliance & RegTech" (AI-driven audit, risk models)

- At LEAST 1 of the {NUM_AXES} Y-axes MUST be about AI-era leadership/delivery:
  → "AI-Augmented Team Productivity" (leveraging AI tools for team output)
  → "AI Ethics & Responsible Innovation" (governance, bias, safety)
  → "AI Change Management" (upskilling teams, managing AI adoption)

AXIS DESIGN RULES:
- X-axes = TECHNICAL / FUNCTIONAL DEPTH (including AI-specific skills for this role)
- Y-axes = BUSINESS IMPACT / LEADERSHIP / DELIVERY (including AI-era leadership)
- Be SPECIFIC to this role and industry. NOT generic.
- Each axis MUST have a "trend" field: "rising", "stable", or "emerging"
  → "rising" = market demand is actively increasing (e.g., AI skills)
  → "stable" = consistently expected, not growing or shrinking
  → "emerging" = early stage but will become important in 1-3 years
- The FIRST axis in each list = MOST IMPORTANT default.
- Each axis: short label (3-7 words) + description (1-2 sentences) + trend.

Return ONLY valid JSON (no markdown fences):
{{
  "xAxes": [
    {{"id": "x0", "label": "...", "description": "...", "trend": "stable|rising|emerging"}},
    ...exactly {NUM_AXES} items
  ],
  "yAxes": [
    {{"id": "y0", "label": "...", "description": "...", "trend": "stable|rising|emerging"}},
    ...exactly {NUM_AXES} items
  ]
}}"""

        user_msg = f"""ROLE: {target_title}
COMPANY: {target_company}
INDUSTRY: {target_industry}
HEADLINE: {target_headline}
CAREER DOMAIN: {career_domain}
DATE: {datetime.now().strftime('%B %Y')}

Generate exactly {NUM_AXES} X-axes and {NUM_AXES} Y-axes.
Remember: at least 2 X-axes and 1 Y-axis MUST be AI/emerging-tech specific
to how AI is transforming the {target_title} role RIGHT NOW."""

        result = gpt.chat_json(system_msg, user_msg, max_tokens=2000,
                               temperature=0.15, label="market_axes")
        if (result and isinstance(result, dict)
                and len(result.get("xAxes", [])) >= NUM_AXES
                and len(result.get("yAxes", [])) >= NUM_AXES):
            for i, ax in enumerate(result["xAxes"][:NUM_AXES]):
                ax["id"] = f"x{i}"
            for i, ax in enumerate(result["yAxes"][:NUM_AXES]):
                ax["id"] = f"y{i}"
            result["xAxes"] = result["xAxes"][:NUM_AXES]
            result["yAxes"] = result["yAxes"][:NUM_AXES]
            debug(f"[AXES] GPT returned {len(result['xAxes'])} X + {len(result['yAxes'])} Y axes")
            return result

    debug("[AXES] GPT unavailable — using fallback axes")
    return _fallback_market_axes(career_domain)


def _fallback_market_axes(career_domain: str) -> Dict[str, Any]:
    """Fallback axes when GPT is unavailable. Includes AI/emerging axes with trends."""
    fallback_configs = {
        "engineering": {
            "xAxes": [
                {"id": "x0", "label": "Core Programming & Algorithms", "description": "Proficiency in modern languages, data structures, and algorithmic problem-solving.", "trend": "stable"},
                {"id": "x1", "label": "AI-Assisted Development & Copilot Fluency", "description": "Using AI coding assistants (Copilot, Cursor, Claude), AI-driven code review, AI-augmented debugging and testing.", "trend": "rising"},
                {"id": "x2", "label": "ML/AI Engineering & Model Deployment", "description": "Building, fine-tuning, and deploying ML models; LLM integration; prompt engineering for production systems.", "trend": "rising"},
                {"id": "x3", "label": "Cloud-Native Architecture (AWS/Azure/GCP)", "description": "Designing scalable cloud infra, containers, serverless, IaC.", "trend": "stable"},
                {"id": "x4", "label": "System Design & Scalability", "description": "Architecting distributed systems, microservices, event-driven architecture at scale.", "trend": "stable"},
                {"id": "x5", "label": "Full-Stack Web Development", "description": "End-to-end web apps — modern React/Next.js frontends, Node/Spring/Django backends.", "trend": "stable"},
                {"id": "x6", "label": "DevSecOps & CI/CD Pipelines", "description": "Security-first DevOps, automated testing, IaC (Terraform/Pulumi), container orchestration.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Technical Leadership & Mentorship", "description": "Leading engineering teams, code reviews, architecture decisions, growing juniors.", "trend": "stable"},
                {"id": "y1", "label": "AI-Augmented Team Productivity", "description": "Leveraging AI tools to multiply team output — AI workflows, automated testing, AI documentation.", "trend": "rising"},
                {"id": "y2", "label": "Cross-Functional Delivery", "description": "Collaborating with product, design, and business to ship on time and within scope.", "trend": "stable"},
                {"id": "y3", "label": "Client Engagement & Requirements", "description": "Managing client expectations, translating business needs into technical solutions.", "trend": "stable"},
                {"id": "y4", "label": "Agile Execution & Sprint Ownership", "description": "Sprint planning, estimation, velocity management, delivery accountability.", "trend": "stable"},
                {"id": "y5", "label": "Business Impact & Revenue Alignment", "description": "Connecting engineering output to business KPIs, cost optimization, revenue impact.", "trend": "stable"},
                {"id": "y6", "label": "Innovation & Open-Source Contribution", "description": "Driving technical innovation, publishing, contributing to OSS, building internal tools.", "trend": "emerging"},
            ],
        },
        "data": {
            "xAxes": [
                {"id": "x0", "label": "Machine Learning & Deep Learning", "description": "Building and tuning ML/DL models for production use cases.", "trend": "stable"},
                {"id": "x1", "label": "LLM Fine-Tuning & RAG Systems", "description": "Building production LLM pipelines, retrieval-augmented generation, prompt engineering.", "trend": "rising"},
                {"id": "x2", "label": "MLOps & Model Governance", "description": "Model deployment, monitoring, versioning, responsible AI practices at scale.", "trend": "rising"},
                {"id": "x3", "label": "Data Pipeline Architecture", "description": "Building scalable ETL/ELT and real-time streaming pipelines.", "trend": "stable"},
                {"id": "x4", "label": "Statistical Analysis & Experimentation", "description": "A/B testing, causal inference, and statistical modeling.", "trend": "stable"},
                {"id": "x5", "label": "Cloud Data Platforms", "description": "Snowflake, Databricks, BigQuery, and cloud-native data tools.", "trend": "stable"},
                {"id": "x6", "label": "Data Visualization & BI", "description": "Creating dashboards, reports, and visual analytics.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Business Insight Translation", "description": "Turning data findings into actionable business recommendations.", "trend": "stable"},
                {"id": "y1", "label": "AI Ethics & Responsible Innovation", "description": "Ensuring fairness, bias mitigation, AI safety, and responsible deployment.", "trend": "rising"},
                {"id": "y2", "label": "Stakeholder Communication", "description": "Presenting findings to non-technical audiences effectively.", "trend": "stable"},
                {"id": "y3", "label": "Data Strategy & Governance", "description": "Defining data strategy, quality standards, and governance.", "trend": "stable"},
                {"id": "y4", "label": "Project Ownership & Delivery", "description": "Owning end-to-end data projects from scoping to deployment.", "trend": "stable"},
                {"id": "y5", "label": "Mentorship & Team Growth", "description": "Growing junior data scientists and building team capability.", "trend": "stable"},
                {"id": "y6", "label": "Cross-Team AI Enablement", "description": "Helping non-data teams adopt AI/ML tools and data-driven practices.", "trend": "emerging"},
            ],
        },
        "product": {
            "xAxes": [
                {"id": "x0", "label": "Product Discovery & User Research", "description": "Customer interviews, user testing, and insight generation.", "trend": "stable"},
                {"id": "x1", "label": "AI Product Design & LLM Integration", "description": "Designing AI-powered features, prompt UX, LLM-based product experiences.", "trend": "rising"},
                {"id": "x2", "label": "AI-Driven Analytics & Experimentation", "description": "Using AI for user insights, predictive analytics, automated A/B testing.", "trend": "rising"},
                {"id": "x3", "label": "Technical Fluency", "description": "Understanding engineering constraints, APIs, and system architecture.", "trend": "stable"},
                {"id": "x4", "label": "Competitive Intelligence", "description": "Market analysis, competitor tracking, and positioning.", "trend": "stable"},
                {"id": "x5", "label": "Roadmap & Prioritization Craft", "description": "Building and managing product roadmaps with clear prioritization.", "trend": "stable"},
                {"id": "x6", "label": "UX & Design Collaboration", "description": "Working closely with design to create intuitive user experiences.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Go-to-Market Execution", "description": "Launch strategy, pricing, and market entry planning.", "trend": "stable"},
                {"id": "y1", "label": "AI Change Management & Adoption", "description": "Leading AI feature adoption, managing user expectations, AI onboarding.", "trend": "rising"},
                {"id": "y2", "label": "Growth & Monetization", "description": "Driving user growth, retention, and revenue from product.", "trend": "stable"},
                {"id": "y3", "label": "Cross-Functional Leadership", "description": "Leading without authority across multiple teams.", "trend": "stable"},
                {"id": "y4", "label": "Strategic Vision & OKRs", "description": "Setting product vision and translating to measurable objectives.", "trend": "stable"},
                {"id": "y5", "label": "Customer Success & Advocacy", "description": "Building customer relationships and driving adoption.", "trend": "stable"},
                {"id": "y6", "label": "Vendor & Partner Management", "description": "Managing third-party integrations and partnerships.", "trend": "stable"},
            ],
        },
        "executive": {
            "xAxes": [
                {"id": "x0", "label": "Strategic Vision & Growth", "description": "Setting long-term direction and driving company growth.", "trend": "stable"},
                {"id": "x1", "label": "AI Strategy & Organizational AI Adoption", "description": "Leading enterprise AI transformation, AI roadmap, AI governance frameworks.", "trend": "rising"},
                {"id": "x2", "label": "AI-Augmented Decision Making", "description": "Using AI for forecasting, scenario planning, operations optimization, real-time analytics.", "trend": "rising"},
                {"id": "x3", "label": "Financial Acumen & Capital Markets", "description": "P&L management, fundraising, and investor relations.", "trend": "stable"},
                {"id": "x4", "label": "Product-Led Growth Strategy", "description": "Building products that drive organic growth and retention.", "trend": "stable"},
                {"id": "x5", "label": "M&A & Corporate Development", "description": "Identifying and executing mergers and acquisitions.", "trend": "stable"},
                {"id": "x6", "label": "Market Expansion & Globalization", "description": "Expanding into new markets and geographies.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Team Building & Culture", "description": "Attracting top talent and building high-performing teams.", "trend": "stable"},
                {"id": "y1", "label": "AI Workforce Transformation", "description": "Upskilling teams for AI, managing AI-driven org changes, AI talent strategy.", "trend": "rising"},
                {"id": "y2", "label": "Board & Investor Communication", "description": "Effective governance, reporting, and stakeholder management.", "trend": "stable"},
                {"id": "y3", "label": "Operational Excellence", "description": "Driving efficiency, scaling operations, process optimization.", "trend": "stable"},
                {"id": "y4", "label": "Crisis Management & Resilience", "description": "Leading through uncertainty, pivots, and disruptions.", "trend": "stable"},
                {"id": "y5", "label": "Customer Centricity & NPS", "description": "Driving customer satisfaction, retention, and advocacy.", "trend": "stable"},
                {"id": "y6", "label": "AI Ethics & Responsible Innovation", "description": "AI safety, bias mitigation, regulatory compliance for AI.", "trend": "emerging"},
            ],
        },
        "sales": {
            "xAxes": [
                {"id": "x0", "label": "Sales Execution & Pipeline Management", "description": "Building pipeline, closing deals, CRM mastery, quota attainment.", "trend": "stable"},
                {"id": "x1", "label": "AI-Powered Prospecting & Sales Intelligence", "description": "Using AI lead scoring, intent data, AI-enriched outreach, predictive analytics.", "trend": "rising"},
                {"id": "x2", "label": "Conversational AI & Sales Automation", "description": "AI chatbots, automated sequences, AI-assisted demos and proposals.", "trend": "rising"},
                {"id": "x3", "label": "Product & Technical Knowledge", "description": "Deep understanding of the product, competitive landscape, use cases.", "trend": "stable"},
                {"id": "x4", "label": "Solution Selling & Consultative Sales", "description": "Discovery-led selling, ROI analysis, complex deal structuring.", "trend": "stable"},
                {"id": "x5", "label": "Account Growth & Expansion", "description": "Upselling, cross-selling, land-and-expand strategies.", "trend": "stable"},
                {"id": "x6", "label": "Social Selling & Digital Presence", "description": "LinkedIn, content-driven outreach, personal brand building.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Revenue Leadership & Forecasting", "description": "Accurate forecasting, territory planning, team quota management.", "trend": "stable"},
                {"id": "y1", "label": "AI-Augmented Sales Coaching", "description": "Using AI call analysis, conversation intelligence, AI-driven coaching.", "trend": "rising"},
                {"id": "y2", "label": "Client Relationship Management", "description": "Building long-term trust, executive relationships, strategic accounts.", "trend": "stable"},
                {"id": "y3", "label": "Cross-Functional GTM Collaboration", "description": "Working with marketing, product, and CS to drive revenue.", "trend": "stable"},
                {"id": "y4", "label": "Negotiation & Deal Strategy", "description": "Complex negotiation, procurement navigation, contract structuring.", "trend": "stable"},
                {"id": "y5", "label": "Team Mentorship & Enablement", "description": "Coaching SDRs/AEs, building playbooks, scaling sales knowledge.", "trend": "stable"},
                {"id": "y6", "label": "Market Intelligence & Feedback Loop", "description": "Bringing market signals back to product and leadership.", "trend": "stable"},
            ],
        },
        "marketing": {
            "xAxes": [
                {"id": "x0", "label": "Content & Brand Strategy", "description": "Brand positioning, messaging, content calendar, editorial strategy.", "trend": "stable"},
                {"id": "x1", "label": "Generative AI Content Production", "description": "AI copywriting, AI image/video generation, AI-powered content at scale.", "trend": "rising"},
                {"id": "x2", "label": "AI-Driven Analytics & Personalization", "description": "Predictive targeting, dynamic content, AI attribution modeling.", "trend": "rising"},
                {"id": "x3", "label": "SEO/SEM & Paid Acquisition", "description": "Organic search, paid campaigns, performance marketing.", "trend": "stable"},
                {"id": "x4", "label": "Demand Generation & Funnel Optimization", "description": "Lead gen, nurture sequences, conversion rate optimization.", "trend": "stable"},
                {"id": "x5", "label": "Social Media & Community", "description": "Platform strategy, community building, influencer partnerships.", "trend": "stable"},
                {"id": "x6", "label": "Marketing Technology Stack", "description": "CRM, MAP, CDP, analytics tools — martech architecture.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Growth Strategy & Revenue Impact", "description": "Connecting marketing to pipeline, revenue attribution, ROI.", "trend": "stable"},
                {"id": "y1", "label": "AI Workflow Automation & Scale", "description": "Automating campaigns, AI-powered A/B testing, scaling with AI tools.", "trend": "rising"},
                {"id": "y2", "label": "Cross-Functional Collaboration", "description": "Working with sales, product, and leadership on GTM.", "trend": "stable"},
                {"id": "y3", "label": "Data-Driven Decision Making", "description": "Using analytics to drive marketing strategy and budget allocation.", "trend": "stable"},
                {"id": "y4", "label": "Brand Leadership & PR", "description": "Managing brand reputation, PR crises, thought leadership.", "trend": "stable"},
                {"id": "y5", "label": "Team Management & Agency Relations", "description": "Managing marketing teams, agencies, and freelancers.", "trend": "stable"},
                {"id": "y6", "label": "Customer Insights & Advocacy", "description": "Voice of customer, NPS programs, customer marketing.", "trend": "stable"},
            ],
        },
        "hr": {
            "xAxes": [
                {"id": "x0", "label": "Talent Acquisition & Employer Brand", "description": "Recruiting strategy, employer branding, candidate experience.", "trend": "stable"},
                {"id": "x1", "label": "AI in Recruitment & Talent Analytics", "description": "AI screening, predictive hiring, AI bias detection, workforce analytics.", "trend": "rising"},
                {"id": "x2", "label": "AI Workforce Planning & Skills Mapping", "description": "Predictive workforce planning, skills gap analysis, AI-driven L&D.", "trend": "rising"},
                {"id": "x3", "label": "Compensation & Benefits Design", "description": "Total rewards, equity, benefits administration, market benchmarking.", "trend": "stable"},
                {"id": "x4", "label": "HRIS & People Operations", "description": "HR systems, payroll, compliance, process automation.", "trend": "stable"},
                {"id": "x5", "label": "Learning & Development", "description": "Training programs, career pathing, leadership development.", "trend": "stable"},
                {"id": "x6", "label": "Labor Law & Compliance", "description": "Employment law, regulatory compliance, policy management.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Culture & Employee Engagement", "description": "Building culture, measuring engagement, driving retention.", "trend": "stable"},
                {"id": "y1", "label": "AI Change Management & Upskilling", "description": "Managing AI-driven org transformation, upskilling programs for AI adoption.", "trend": "rising"},
                {"id": "y2", "label": "DE&I & Belonging", "description": "Diversity, equity, inclusion programs, belonging metrics.", "trend": "stable"},
                {"id": "y3", "label": "Org Design & Transformation", "description": "Restructuring, scaling teams, M&A people integration.", "trend": "stable"},
                {"id": "y4", "label": "Strategic Business Partnering", "description": "Advising leadership on people strategy, workforce planning.", "trend": "stable"},
                {"id": "y5", "label": "Performance Management", "description": "Goal setting, reviews, feedback culture, PIP processes.", "trend": "stable"},
                {"id": "y6", "label": "Employee Relations & Wellbeing", "description": "Conflict resolution, mental health, work-life balance programs.", "trend": "stable"},
            ],
        },
        "finance": {
            "xAxes": [
                {"id": "x0", "label": "Financial Analysis & FP&A", "description": "Budgeting, forecasting, variance analysis, financial modeling.", "trend": "stable"},
                {"id": "x1", "label": "AI-Powered Financial Modeling", "description": "ML-based forecasting, anomaly detection, predictive financial analytics.", "trend": "rising"},
                {"id": "x2", "label": "Automated Compliance & RegTech", "description": "AI-driven audit, real-time compliance monitoring, regulatory reporting automation.", "trend": "rising"},
                {"id": "x3", "label": "Accounting & Reporting Standards", "description": "GAAP/IFRS, audit preparation, consolidation, reporting.", "trend": "stable"},
                {"id": "x4", "label": "Capital Markets & Treasury", "description": "Fundraising, debt management, cash flow optimization, investor relations.", "trend": "stable"},
                {"id": "x5", "label": "Tax Strategy & Planning", "description": "Tax optimization, transfer pricing, international tax.", "trend": "stable"},
                {"id": "x6", "label": "Risk Management & Controls", "description": "Internal controls, risk assessment, SOX compliance.", "trend": "stable"},
            ],
            "yAxes": [
                {"id": "y0", "label": "Strategic Business Partnering", "description": "Advising business units on financial strategy and decisions.", "trend": "stable"},
                {"id": "y1", "label": "AI-Augmented Decision Support", "description": "Using AI dashboards, scenario modeling, real-time financial intelligence for leadership.", "trend": "rising"},
                {"id": "y2", "label": "Cross-Functional Leadership", "description": "Working across departments to drive financial discipline.", "trend": "stable"},
                {"id": "y3", "label": "Board & Stakeholder Communication", "description": "Financial reporting to board, investor updates, transparent communication.", "trend": "stable"},
                {"id": "y4", "label": "Team Management & Development", "description": "Building and leading finance teams, succession planning.", "trend": "stable"},
                {"id": "y5", "label": "Process Optimization & Automation", "description": "Streamlining financial processes, ERP optimization, workflow automation.", "trend": "stable"},
                {"id": "y6", "label": "M&A Financial Due Diligence", "description": "Financial analysis for acquisitions, integration planning.", "trend": "stable"},
            ],
        },
    }
    return fallback_configs.get(career_domain, fallback_configs["engineering"])


def score_persons_on_axes(
    gpt: AzureGPT,
    axes: Dict[str, Any],
    participants: List[Dict[str, Any]],
    skills_by_person: Dict[str, List[str]],
    career_domain: str,
) -> Dict[str, Dict[str, float]]:
    """
    GPT evaluates each person on ALL 14 axes (7X + 7Y) with scores 0-10.
    INDEPENDENT of skill clusters — this is a market-projection evaluation.
    Returns: { person_id: { "x0": 5.2, "x1": 3.1, ..., "y0": 7.8, ... } }
    """
    all_axes = axes["xAxes"] + axes["yAxes"]
    axis_descriptions = "\n".join(
        f"  {ax['id']}: {ax['label']} — {ax['description']} [trend: {ax.get('trend', 'stable')}]"
        for ax in all_axes
    )

    person_profiles = []
    for p in participants:
        eid = p["employee_id"]
        skills = skills_by_person.get(eid, [])
        person_profiles.append({
            "id": p["id"],
            "name": p.get("name", ""),
            "title": p.get("title", ""),
            "company": p.get("company", ""),
            "isTarget": p.get("isTarget", False),
            "skills": skills[:40],
        })

    if gpt.available:
        system_msg = f"""You are an expert career assessor evaluating professionals against
market-relevant competency dimensions for the {career_domain} domain in 2025-2026.

For each person, score them on EACH of the {len(all_axes)} axes on a 0-10 scale.

SCORING GUIDE:
  0-1: No evidence at all — nothing in their profile relates to this dimension
  2-3: Minimal / tangential — very weak, only loosely related skills
  4-5: Basic / emerging — some relevant foundation but clearly below market standard
  6-7: Competent — solid mid-level capability, meets market expectations
  8-9: Strong — above average, clear competitive advantage
  10:  Exceptional — top-tier, market-leading, rare expertise

CRITICAL SCORING RULES:
- Be HONEST and DIFFERENTIATED. Use the FULL 0-10 range.
- Look at the person's ACTUAL skills list. If they have 0 skills related to
  an axis, score them 0-2. If they have deep, relevant skills, score 7-10.
- DIFFERENTIATE between people — if one has 6 relevant skills and another
  has 0, their scores should differ by 4-7 points.

AI / EMERGING TECHNOLOGY AXES — SPECIAL SCORING GUIDANCE:
  These axes have [trend: rising] or [trend: emerging] markers. For these:
  - Score ONLY based on ACTUAL evidence. Do NOT give benefit of the doubt.
  - AI-related skill signals to look for:
    • Direct AI skills: "machine learning", "deep learning", "tensorflow",
      "pytorch", "LLM", "GPT", "NLP", "computer vision", "AI", "ML",
      "model deployment", "MLOps", "prompt engineering", "copilot",
      "generative AI", "RAG", "fine-tuning", "transformers"
    • AI-adjacent skills that suggest readiness: "python", "data science",
      "statistics", "neural networks", "feature engineering", "data pipeline"
    • AI leadership signals: "AI strategy", "AI adoption", "AI governance",
      "AI transformation", "automation", "AI tools"
  - If a person has NONE of these → score 0-1 on AI axes
  - If a person has "python" + generic programming but no AI skills → score 1-3
  - If a person has explicit ML/AI skills → score 5-8 depending on depth
  - If a person has production AI deployment experience → score 8-10

  Example: A Senior Software Engineer whose skills are "L2 Support on IBM Products",
  "French Server Monitoring", "Credit Note Processing", "Invoice Processing"
  → AI-Assisted Development: 0-1 (no AI evidence at all)
  → Core Programming: 1-2 (support work, not development)

  Example: A person with "Java Full Stack", "Python", "AWS", "Spring Boot",
  "SQL", "JavaScript/React"
  → AI-Assisted Development: 2-3 (has Python which is AI-adjacent, but no actual AI)
  → Core Programming: 7-8 (strong multi-language development)
  → Cloud-Native: 5-6 (AWS experience)

  Example: A person with "TensorFlow", "PyTorch", "ML model deployment",
  "Python", "AWS SageMaker", "NLP"
  → AI/ML Engineering: 8-9 (deep, production-grade AI)
  → AI-Assisted Development: 6-7 (would naturally use AI tools)

AXES TO EVALUATE:
{axis_descriptions}

Return ONLY valid JSON (no markdown fences):
{{
  "scores": {{
    "person_id_here": {{"x0": 5.2, "x1": 3.1, "y0": 7.8, ...all {len(all_axes)} axes}},
    ...one entry per person
  }}
}}"""

        user_msg = f"""PERSONS TO EVALUATE ({len(person_profiles)} people):
{json.dumps(person_profiles, indent=2)}

Score each person on ALL {len(all_axes)} axes. Return complete scores."""

        result = gpt.chat_json(system_msg, user_msg, max_tokens=3000,
                               temperature=0.1, label="axis_scoring")
        if result and isinstance(result, dict) and "scores" in result:
            scores = result["scores"]
            validated: Dict[str, Dict[str, float]] = {}
            for pid, axis_scores in scores.items():
                validated[pid] = {}
                for ax in all_axes:
                    val = axis_scores.get(ax["id"], 0)
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = 0.0
                    validated[pid][ax["id"]] = round(max(0.0, min(10.0, val)), 1)
            # Ensure every participant has scores
            for p in participants:
                if p["id"] not in validated:
                    validated[p["id"]] = {ax["id"]: 0.0 for ax in all_axes}
            debug(f"[AXES] GPT scored {len(validated)} persons on {len(all_axes)} axes")
            return validated

    debug("[AXES] GPT unavailable — using heuristic scoring")
    return _fallback_axis_scoring(axes, participants, skills_by_person)


def _fallback_axis_scoring(
    axes: Dict[str, Any],
    participants: List[Dict[str, Any]],
    skills_by_person: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Heuristic scoring when GPT unavailable. Includes AI-specific keyword sets."""
    all_axes = axes["xAxes"] + axes["yAxes"]

    # Extended keyword sets for AI axes
    ai_keywords = {
        "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
        "natural language", "computer vision", "llm", "gpt", "bert", "transformer",
        "neural network", "model deployment", "mlops", "prompt engineering",
        "copilot", "generative ai", "rag", "fine-tuning", "ai", "ml ",
        "sagemaker", "vertex ai", "openai", "langchain", "vector database",
        "embeddings", "chatbot", "ai strategy", "ai adoption", "ai governance",
        "ai tools", "automation", "intelligent automation", "data science",
    }
    ai_adjacent = {"python", "statistics", "data pipeline", "data engineering",
                   "analytics", "feature engineering", "jupyter", "pandas", "numpy"}

    scores: Dict[str, Dict[str, float]] = {}
    for p in participants:
        eid = p["employee_id"]
        skills = skills_by_person.get(eid, [])
        skill_text = " ".join(skills).lower()
        scores[p["id"]] = {}

        for ax in all_axes:
            trend = ax.get("trend", "stable")
            is_ai_axis = trend == "rising" or any(
                kw in ax.get("label", "").lower() for kw in ["ai", "ml", "copilot", "llm"]
            )

            if is_ai_axis:
                # AI-specific scoring: check for actual AI evidence
                direct_hits = sum(1 for kw in ai_keywords if kw in skill_text)
                adjacent_hits = sum(1 for kw in ai_adjacent if kw in skill_text)
                if direct_hits >= 3:
                    score = min(10.0, 5.0 + direct_hits * 0.8)
                elif direct_hits >= 1:
                    score = min(7.0, 2.5 + direct_hits * 1.5 + adjacent_hits * 0.3)
                elif adjacent_hits >= 2:
                    score = min(3.0, adjacent_hits * 0.8)
                else:
                    score = 0.5  # No AI evidence at all
            else:
                # Standard keyword matching for non-AI axes
                label_words = [w.lower() for w in ax.get("label", "").split() if len(w) > 2]
                desc_words = [w.lower() for w in ax.get("description", "").split() if len(w) > 3]
                all_keywords = set(label_words + desc_words)
                hits = sum(1 for kw in all_keywords if kw in skill_text)
                score = min(10.0, hits * 1.5)

            scores[p["id"]][ax["id"]] = round(score, 1)
    return scores


def generate_market_benchmarks(
    gpt: AzureGPT,
    axes: Dict[str, Any],
    target_title: str,
    target_industry: str,
    career_domain: str,
) -> Dict[str, float]:
    """
    GPT generates the IDEAL market benchmark score (0-10) for each axis.
    This represents: "Where should a competitive person with this job title
    score on each axis to meet current market expectations?"

    Returns: { "x0": 7.5, "x1": 6.0, ..., "y0": 7.0, ... }
    """
    all_axes = axes["xAxes"] + axes["yAxes"]
    axis_list = "\n".join(
        f"  {ax['id']}: {ax['label']} — {ax['description']} [trend: {ax.get('trend', 'stable')}]"
        for ax in all_axes
    )

    if gpt.available:
        system_msg = f"""You are a labor market analyst and hiring expert specializing in
how AI is reshaping every job function in 2025-2026.

For the given job title and industry, determine the IDEAL MARKET BENCHMARK score
(0-10) on each competency axis. This score represents:
  "Where should a COMPETITIVE, WELL-POSITIONED professional with this exact
   title score to meet current and NEAR-FUTURE (2025-2027) market expectations?"

This is NOT the minimum. This is the TARGET — what top companies look for when
hiring or promoting for this role TODAY.

SCORING GUIDE:
  10 = absolute must-have, market won't consider you without it
  8-9 = strongly expected, most competitive candidates have this
  6-7 = expected for this level, a gap here is noticeable
  4-5 = nice to have at this level, becoming more important
  2-3 = emerging expectation, not yet critical but trending
  0-1 = not relevant to this role currently

TREND-AWARE BENCHMARKING — CRITICAL:
Each axis has a [trend] marker. Use it:
  - [trend: rising] = Market demand is ACTIVELY increasing. These axes should
    be benchmarked HIGHER than you might think, because the market in 12 months
    will expect MORE than today. For AI axes especially:
    • "AI-Assisted Development" for engineers → benchmark 6-7 (it's becoming
      standard to use Copilot/Cursor — engineers who can't are falling behind)
    • "AI Strategy" for executives → benchmark 7-8 (boards expect AI plans)
    • "Generative AI Content" for marketing → benchmark 6-7 (already table stakes)
  - [trend: stable] = Consistently expected. Benchmark at the traditional level.
  - [trend: emerging] = Early stage. Benchmark at 3-5 (not yet mandatory but
    will matter in 2-3 years — the forward-looking professional should start).

COMPANY TYPE MATTERS:
  - IT services (Accenture, TCS, HCL, Wipro, Infosys) → clients are asking for
    AI capabilities NOW, so AI benchmarks should be HIGHER for services companies
  - Product companies → AI is a feature differentiator
  - Startups → AI is often core to the product

Be REALISTIC but FORWARD-LOOKING. The benchmark should push people toward
where the market IS GOING, not where it was 2 years ago.

AXES:
{axis_list}

Return ONLY valid JSON (no markdown fences):
{{
  "benchmarks": {{"x0": 8.0, "x1": 6.5, ..., "y0": 7.0, ...}},
  "rationale": "Brief explanation of the benchmark philosophy for this role"
}}"""

        user_msg = f"""JOB TITLE: {target_title}
INDUSTRY: {target_industry}
CAREER DOMAIN: {career_domain}
DATE: {datetime.now().strftime('%B %Y')}

What scores should a competitive {target_title} achieve on each axis?"""

        result = gpt.chat_json(system_msg, user_msg, max_tokens=800,
                               temperature=0.15, label="market_benchmarks")
        if result and isinstance(result, dict) and "benchmarks" in result:
            benchmarks = result["benchmarks"]
            validated = {}
            for ax in all_axes:
                val = benchmarks.get(ax["id"], 6.0)
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 6.0
                validated[ax["id"]] = round(max(0.0, min(10.0, val)), 1)
            debug(f"[BENCHMARK] GPT generated market benchmarks for {len(validated)} axes")
            rationale = result.get("rationale", "")
            return {"scores": validated, "rationale": rationale}

    debug("[BENCHMARK] GPT unavailable — using fallback benchmarks")
    return _fallback_market_benchmarks(axes, career_domain)


def _fallback_market_benchmarks(
    axes: Dict[str, Any], career_domain: str,
) -> Dict[str, Any]:
    """Fallback benchmarks that respect trend indicators."""
    benchmarks = {}
    trend_boost = {"rising": 1.0, "emerging": 0.0, "stable": 0.0}
    for ax in axes["xAxes"]:
        base = 7.0
        boost = trend_boost.get(ax.get("trend", "stable"), 0)
        benchmarks[ax["id"]] = min(10.0, base + boost)
    for ax in axes["yAxes"]:
        base = 6.5
        boost = trend_boost.get(ax.get("trend", "stable"), 0)
        benchmarks[ax["id"]] = min(10.0, base + boost)
    # First X and first Y are "most important"
    if axes["xAxes"]:
        benchmarks[axes["xAxes"][0]["id"]] = max(benchmarks.get(axes["xAxes"][0]["id"], 8.0), 8.0)
    if axes["yAxes"]:
        benchmarks[axes["yAxes"][0]["id"]] = max(benchmarks.get(axes["yAxes"][0]["id"], 7.5), 7.5)
    return {"scores": benchmarks, "rationale": "Fallback benchmarks — rising AI axes get +1.0 boost."}


def compute_all_axis_combinations(
    axes: Dict[str, Any],
    person_scores: Dict[str, Dict[str, float]],
    participants: List[Dict[str, Any]],
    market_benchmarks: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-compute quadrant positions for ALL 49 (7x7) axis combinations.
    Each combo includes: person positions + market benchmark target zone.
    Pure math — no GPT calls.
    Returns: { "x0_y0": { "positions": [...], "marketTarget": {x, y} }, ... }
    """
    debug(f"\n[COMBOS] Computing {NUM_AXES}x{NUM_AXES} = {NUM_AXES**2} axis combinations")
    combinations: Dict[str, Dict[str, Any]] = {}

    for x_ax in axes["xAxes"]:
        for y_ax in axes["yAxes"]:
            combo_key = f"{x_ax['id']}_{y_ax['id']}"
            positions = []
            for p in participants:
                pid = p["id"]
                x_val = person_scores.get(pid, {}).get(x_ax["id"], 0.0)
                y_val = person_scores.get(pid, {}).get(y_ax["id"], 0.0)

                if x_val >= 5 and y_val >= 5:
                    qlabel = "Upper-Right"
                elif x_val < 5 and y_val >= 5:
                    qlabel = "Upper-Left"
                elif x_val >= 5 and y_val < 5:
                    qlabel = "Lower-Right"
                else:
                    qlabel = "Lower-Left"

                positions.append({
                    "personId": pid,
                    "personName": p["name"],
                    "x": x_val,
                    "y": y_val,
                    "quadrantLabel": qlabel,
                    "isTarget": p.get("isTarget", False),
                })

            # Market benchmark target point for this axis combo
            bench_x = market_benchmarks.get(x_ax["id"], 7.0)
            bench_y = market_benchmarks.get(y_ax["id"], 6.5)

            combinations[combo_key] = {
                "positions": positions,
                "marketTarget": {
                    "x": bench_x,
                    "y": bench_y,
                    "xAxisLabel": x_ax["label"],
                    "yAxisLabel": y_ax["label"],
                },
            }

    debug(f"[COMBOS] Generated {len(combinations)} combinations with market targets")
    return combinations


def pick_best_default_combo(
    quadrant_combinations: Dict[str, Dict[str, Any]],
    axes: Dict[str, Any],
) -> Dict[str, str]:
    """
    Pick the axis combo that produces the BEST-LOOKING scatter plot.
    
    "Best aesthetic" means:
      1. Points spread across multiple quadrants (not all in Lower-Left)
      2. Good variance — people are differentiated, not clumped together
      3. Target person is NOT at 0,0 (boring) — ideally somewhere interesting
      4. Market benchmark creates a visible gap (not sitting on top of someone)

    Returns: {"xAxisId": "x2", "yAxisId": "y3", "comboKey": "x2_y3", "aestheticScore": 8.5}
    """
    debug("\n[AESTHETIC] Picking best default combo from 49 options...")
    best_key = f"{axes['xAxes'][0]['id']}_{axes['yAxes'][0]['id']}"
    best_score = -1.0

    for combo_key, combo_data in quadrant_combinations.items():
        positions = combo_data.get("positions", [])
        market_target = combo_data.get("marketTarget", {})
        if not positions:
            continue

        xs = [p["x"] for p in positions]
        ys = [p["y"] for p in positions]
        n = len(positions)

        # --- Score component 1: Quadrant diversity (0-4 pts) ---
        quadrants_used = set(p["quadrantLabel"] for p in positions)
        quadrant_score = len(quadrants_used)  # 1-4

        # --- Score component 2: Point spread / variance (0-3 pts) ---
        if n > 1:
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            spread_score = min(3.0, (x_range + y_range) / 6.0)  # normalize
        else:
            spread_score = 0.0

        # --- Score component 3: Not all clumped in one spot (0-2 pts) ---
        # Penalize if all points are within a 2x2 box
        if n > 1:
            avg_x = sum(xs) / n
            avg_y = sum(ys) / n
            avg_dist = sum(((x - avg_x)**2 + (y - avg_y)**2)**0.5 for x, y in zip(xs, ys)) / n
            dispersion_score = min(2.0, avg_dist / 2.0)
        else:
            dispersion_score = 0.0

        # --- Score component 4: Target not at origin (0-2 pts) ---
        target_pos = next((p for p in positions if p.get("isTarget")), None)
        if target_pos:
            target_dist = (target_pos["x"]**2 + target_pos["y"]**2)**0.5
            # Best if target is somewhere interesting (2-7 range), not at 0 or 10
            target_interest = min(2.0, target_dist / 4.0)
        else:
            target_interest = 0.0

        # --- Score component 5: Visible gap to market benchmark (0-2 pts) ---
        if target_pos and market_target:
            bench_dist = ((market_target.get("x", 7) - target_pos["x"])**2 +
                          (market_target.get("y", 7) - target_pos["y"])**2)**0.5
            # Good if benchmark is visibly away but not absurdly far
            gap_visibility = min(2.0, bench_dist / 4.0)
        else:
            gap_visibility = 0.0

        # --- Score component 6: At least one person in Upper-Right (0-1 pt) ---
        has_upper_right = 1.0 if "Upper-Right" in quadrants_used else 0.0

        total = (quadrant_score + spread_score + dispersion_score +
                 target_interest + gap_visibility + has_upper_right)

        if total > best_score:
            best_score = total
            best_key = combo_key

    parts = best_key.split("_")
    x_id = parts[0] if len(parts) >= 1 else axes["xAxes"][0]["id"]
    y_id = parts[1] if len(parts) >= 2 else axes["yAxes"][0]["id"]

    x_label = next((a["label"] for a in axes["xAxes"] if a["id"] == x_id), x_id)
    y_label = next((a["label"] for a in axes["yAxes"] if a["id"] == y_id), y_id)
    debug(f"[AESTHETIC] ✅ Best combo: {best_key} (score={best_score:.1f})")
    debug(f"  X: {x_label}")
    debug(f"  Y: {y_label}")

    return {
        "xAxisId": x_id,
        "yAxisId": y_id,
        "comboKey": best_key,
        "aestheticScore": round(best_score, 1),
        "xAxisLabel": x_label,
        "yAxisLabel": y_label,
    }


def compute_axis_gaps(
    target_pid: str,
    person_scores: Dict[str, Dict[str, float]],
    axes: Dict[str, Any],
    participants: List[Dict[str, Any]],
    market_benchmarks: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Compute gaps on EACH axis:
      1. Gap vs best peer
      2. Gap vs market benchmark (where the market says you SHOULD be)
    The market gap is the primary sort — it shows how far from market expectations.
    """
    all_axes = axes["xAxes"] + axes["yAxes"]
    target_scores = person_scores.get(target_pid, {})
    peer_pids = [p["id"] for p in participants if p["id"] != target_pid]

    gaps = []
    for ax in all_axes:
        target_val = target_scores.get(ax["id"], 0.0)
        peer_vals = [person_scores.get(pid, {}).get(ax["id"], 0.0) for pid in peer_pids]
        max_peer = max(peer_vals) if peer_vals else 0.0
        avg_peer = sum(peer_vals) / len(peer_vals) if peer_vals else 0.0
        benchmark = market_benchmarks.get(ax["id"], 7.0)

        gap_vs_peer = round(max_peer - target_val, 1)
        gap_vs_market = round(benchmark - target_val, 1)

        # Include if there's ANY gap (peer or market)
        if gap_vs_peer > 0 or gap_vs_market > 0:
            gaps.append({
                "axisId": ax["id"],
                "axisLabel": ax["label"],
                "axisType": "x" if ax["id"].startswith("x") else "y",
                "targetScore": target_val,
                "maxPeerScore": round(max_peer, 1),
                "avgPeerScore": round(avg_peer, 1),
                "marketBenchmark": benchmark,
                "gapVsPeer": max(0, gap_vs_peer),
                "gapVsMarket": max(0, gap_vs_market),
                "gapPercent": round((max(0, gap_vs_market) / 10) * 100, 1),
                "description": ax.get("description", ""),
                "severity": (
                    "critical" if gap_vs_market >= 5 else
                    "significant" if gap_vs_market >= 3 else
                    "moderate" if gap_vs_market >= 1.5 else
                    "minor"
                ),
            })

    # Sort by market gap (primary) — this is the projection gap
    gaps.sort(key=lambda g: g["gapVsMarket"], reverse=True)
    return gaps


# ════════════════════════════════════════════════════════════════
#  5.  HEATMAP ENGINE (cluster-based, unchanged)
# ════════════════════════════════════════════════════════════════

class AtlasEngine:
    @staticmethod
    def build_heatmap(participants, clusters, skills_by_person):
        matrix = []
        for cluster_name, cluster_data in clusters.items():
            cluster_skills_lower = {s.lower() for s in cluster_data.get("skills", [])}
            values = {}
            for person in participants:
                eid = person["employee_id"]
                pid = person["id"]
                person_skills_lower = {s.lower() for s in skills_by_person.get(eid, [])}
                match_count = len(cluster_skills_lower & person_skills_lower)
                if match_count == 0: score = 0
                elif match_count <= 2: score = 1
                elif match_count <= 4: score = 2
                else: score = 3
                values[pid] = score
            matrix.append({"cluster": cluster_name, "values": values})
        return matrix

    @staticmethod
    def compute_cluster_gaps(target_pid, heatmap_matrix):
        gaps = []
        for row in heatmap_matrix:
            values = row["values"]
            target_score = values.get(target_pid, 0)
            comp_scores = [v for k, v in values.items() if k != target_pid]
            max_comp = max(comp_scores) if comp_scores else 0
            gap = max_comp - target_score
            if gap > 0:
                gaps.append({
                    "cluster": row["cluster"], "targetScore": target_score,
                    "maxScore": max_comp, "gap": gap,
                    "gapPercent": round((gap / 3) * 100, 1),
                })
        gaps.sort(key=lambda g: g["gap"], reverse=True)
        return gaps


# ════════════════════════════════════════════════════════════════
#  6.  GPT-ENHANCED SECTIONS
# ════════════════════════════════════════════════════════════════

def generate_executive_summary(gpt, target_label, career_domain, default_position, axis_gaps, skill_gap_rows):
    if not gpt.available:
        return f"{target_label} has development areas across multiple market dimensions."
    payload = {
        "target": target_label, "career_domain": career_domain,
        "defaultPosition": default_position, "topAxisGaps": axis_gaps[:5],
        "criticalSkillGaps": [{"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
             "reason": sg["gap_reasoning"]} for sg in skill_gap_rows[:8]],
    }
    text = gpt.chat(
        system="Write a crisp 2-paragraph executive summary for a skill-gap report. "
               "Reference market positioning and key axis gaps. Plain English, no buzzwords.",
        user=json.dumps(payload, ensure_ascii=False, indent=2),
        max_tokens=400, label="executive_summary",
    )
    return text.strip() if text else ""


def generate_gap_actions(gpt, target_label, career_domain, axis_gaps, skill_gap_rows):
    if gpt.available and axis_gaps:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "axisGaps": axis_gaps[:6],
            "skillGapsDetail": [{"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
                 "reason": sg["gap_reasoning"]} for sg in skill_gap_rows[:10]],
        }
        system = (
            "You are an executive coach. For each axis gap, return 2-4 concrete "
            "actions and a timeline (0-3 months, 3-9 months, 9-18 months).\n\n"
            "Return ONLY JSON array:\n"
            '[{"axisId":"...","axisLabel":"...","gap":3.2,"actions":["..."],"timeline":"0-3 months"}, ...]'
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=1500, label="gap_actions")
        if isinstance(result, list) and result:
            return result
    return [
        {"axisId": g["axisId"], "axisLabel": g["axisLabel"], "gap": g["gap"],
         "actions": [f"Complete a focused learning path on '{g['axisLabel']}'.",
                     f"Apply {g['axisLabel']} skills in real work.",
                     f"Find a mentor strong in {g['axisLabel']}."],
         "timeline": "0-3 months" if g["gap"] >= 3 else "3-9 months"}
        for g in axis_gaps[:6]
    ]


def generate_course_plans(gpt, target_label, career_domain, skill_gap_rows, existing_courses):
    top_gaps = skill_gap_rows[:6]
    if gpt.available and top_gaps:
        payload = {
            "target": target_label, "career_domain": career_domain,
            "skill_gaps": [{"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
                 "reason": sg["gap_reasoning"]} for sg in top_gaps],
            "existing_courses": [{"name": c.get("course_name", ""), "id": str(c.get("course_id", ""))}
                for c in existing_courses[:5]],
        }
        system = (
            "Curriculum designer. For each skill gap, design 3-level course ladder "
            "(beginner/intermediate/advanced). Each: modules (2-4), outcome, successMetrics (2-3), estimatedHours.\n"
            "Return ONLY JSON array: [{skill, importance, reason, levels: {beginner, intermediate, advanced}}]"
        )
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=2500, label="course_plans")
        if isinstance(result, list) and result:
            return result
    return [{"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
             "reason": sg["gap_reasoning"],
             "levels": {
                 "beginner": {"modules": [f"{sg['skill_gap_name']} Foundations"],
                              "outcome": f"Understand {sg['skill_gap_name']}.", "successMetrics": ["Complete assessment"], "estimatedHours": 4},
                 "intermediate": {"modules": [f"Applying {sg['skill_gap_name']}"],
                              "outcome": f"Apply {sg['skill_gap_name']} in work.", "successMetrics": ["Complete 2 exercises"], "estimatedHours": 8},
                 "advanced": {"modules": [f"Advanced {sg['skill_gap_name']}"],
                              "outcome": f"Lead with {sg['skill_gap_name']}.", "successMetrics": ["Drive a project"], "estimatedHours": 12},
             }} for sg in top_gaps]


# ════════════════════════════════════════════════════════════════
#  6b. HELPER BUILDERS
# ════════════════════════════════════════════════════════════════

def build_person_skills_map(participants, skills_by_person):
    return {p["name"]: sorted(skills_by_person.get(p["employee_id"], [])) for p in participants}

def build_cluster_skill_map(clusters):
    return {cname: sorted(cdata.get("skills", [])) for cname, cdata in clusters.items()}

def build_cluster_evidence_by_person(participants, clusters, skills_by_person):
    evidence = {}
    for p in participants:
        eid = p["employee_id"]
        person_skills_lower = {s.lower(): s for s in skills_by_person.get(eid, [])}
        person_evidence = {}
        for cname, cdata in clusters.items():
            cluster_skills_lower = {s.lower() for s in cdata.get("skills", [])}
            matched_lower = cluster_skills_lower & set(person_skills_lower.keys())
            if matched_lower:
                matched_original = sorted(person_skills_lower[sl] for sl in matched_lower)
                person_evidence[cname] = {"matchCount": len(matched_original), "matchedSkills": matched_original}
        evidence[p["name"]] = person_evidence
    return evidence

def build_heatmap_markdown(heatmap_matrix, participants):
    if not heatmap_matrix or not participants: return ""
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

HEATMAP_LEGEND = "0 = no/low evidence · 1 = basic · 2 = solid · 3 = strong"


def generate_peer_descriptions(gpt, participants, skills_by_person, career_domain, person_scores, axes):
    default_x = axes["xAxes"][0]["id"]
    default_y = axes["yAxes"][0]["id"]
    peer_data = []
    for p in participants:
        scores = person_scores.get(p["id"], {})
        peer_data.append({
            "id": p["id"], "name": p.get("name", ""), "title": p.get("title", ""),
            "company": p.get("company", ""), "isTarget": p.get("isTarget", False),
            "defaultXScore": scores.get(default_x, 0), "defaultYScore": scores.get(default_y, 0),
            "topSkills": skills_by_person.get(p["employee_id"], [])[:8],
        })
    if gpt.available and peer_data:
        system = ("Generate one-line professional descriptions (max 15 words) per person.\n"
                  f"Career domain: {career_domain}.\nReturn ONLY JSON: {{\"p_id\": \"description\", ...}}")
        result = gpt.chat_json(system, json.dumps(peer_data), max_tokens=1000, label="peer_descriptions")
        if isinstance(result, dict) and result:
            return result
    return {p["id"]: f"{p.get('title', 'Professional')} with emerging profile." for p in participants}


def build_swot_structured(gpt, target_label, career_domain, default_position, axis_gaps,
                          heatmap_matrix, target_pid, skill_gap_rows):
    strong_clusters, weak_clusters = [], []
    for row in heatmap_matrix:
        ts = row["values"].get(target_pid, 0)
        cs = [v for k, v in row["values"].items() if k != target_pid]
        avg = sum(cs) / len(cs) if cs else 0
        if ts >= 2 and ts >= avg:
            strong_clusters.append({"cluster": row["cluster"], "score": ts})
        elif ts < avg:
            weak_clusters.append({"cluster": row["cluster"], "score": ts, "peerAvg": round(avg, 1)})

    if gpt.available:
        payload = {"target": target_label, "career_domain": career_domain,
                   "defaultPosition": default_position,
                   "strong_clusters": strong_clusters, "weak_clusters": weak_clusters,
                   "axisGaps": axis_gaps[:5],
                   "skillGaps": [{"skill": sg["skill_gap_name"], "importance": sg["skill_importance"]}
                                 for sg in skill_gap_rows[:6]]}
        system = ("Generate 3-5 pros and 3-5 cons. Each: 'title' (3-6 words) + 'description' (1-2 sentences).\n"
                  f"Career domain: {career_domain}.\n"
                  "Return ONLY JSON: {\"pros\":[{\"title\":\"...\",\"description\":\"...\"}],\"cons\":[...]}")
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=1000, label="swot_pros_cons")
        if isinstance(result, dict) and "pros" in result and result["pros"]:
            if isinstance(result["pros"][0], dict) and "title" in result["pros"][0]:
                return result

    pros = [{"title": f"Strong in {c['cluster']}", "description": f"Scores {c['score']}/3."}
            for c in strong_clusters[:4]] or [{"title": "Foundational Skills", "description": "Core skills detected."}]
    cons = [{"title": f"Gap in {c['cluster']}", "description": f"Scores {c['score']}/3 vs peer avg {c['peerAvg']}."}
            for c in weak_clusters[:4]] or [{"title": "Development Needed", "description": "Gaps across clusters."}]
    return {"pros": pros, "cons": cons}


def generate_presentation_pack(gpt, target_label, career_domain, executive_summary,
                                axis_gaps, pros_cons, gap_actions, default_position, heatmap_matrix):
    if gpt.available:
        payload = {"target": target_label, "career_domain": career_domain,
                   "executive_summary": executive_summary, "defaultPosition": default_position,
                   "topAxisGaps": axis_gaps[:5], "pros": pros_cons.get("pros", []),
                   "cons": pros_cons.get("cons", []), "gap_actions": gap_actions[:5],
                   "cluster_count": len(heatmap_matrix)}
        system = ("Create 10-14 slide exec presentation. Each: title, 3-5 bullets, 1-2 speaker notes.\n"
                  "Cover: context, heatmap, multi-axis quadrant, gaps, pros/cons, upskilling, roadmap, next steps.\n"
                  "Return ONLY JSON: {\"slides\":[{\"title\":\"...\",\"bullets\":[...],\"speakerNotes\":[...]}]}")
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=2500, label="presentation_pack")
        if isinstance(result, dict) and "slides" in result:
            return result
    return {"slides": [
        {"title": "Overview", "bullets": [f"Target: {target_label}", f"Domain: {career_domain}"], "speakerNotes": ["Set context."]},
        {"title": "Summary", "bullets": [executive_summary[:200]], "speakerNotes": ["Key takeaway."]},
        {"title": "Next Steps", "bullets": ["Assign mentor", "Schedule check-ins"], "speakerNotes": ["Close."]},
    ]}


def generate_roadmap(gpt, target_label, career_domain, gap_actions, course_plans, axis_gaps):
    if gpt.available:
        payload = {"target": target_label, "career_domain": career_domain,
                   "gap_actions": gap_actions[:6],
                   "course_plans": [{"skill": cp.get("skill", ""), "importance": cp.get("importance", "")}
                                    for cp in course_plans[:6]], "axisGaps": axis_gaps[:6]}
        system = ("Create 90/180-day roadmap. 0-90: quick wins. 90-180: deeper application. 4-8 items each.\n"
                  "Return ONLY JSON: {\"0_to_90_days\":[...],\"90_to_180_days\":[...]}")
        result = gpt.chat_json(system, json.dumps(payload), max_tokens=800, label="roadmap")
        if isinstance(result, dict) and "0_to_90_days" in result:
            return result
    p1 = [f"[{ga.get('axisLabel', '')}] {ga['actions'][0]}" for ga in gap_actions[:3] if ga.get("actions")]
    p2 = [f"[{ga.get('axisLabel', '')}] {ga['actions'][-1]}" for ga in gap_actions[3:6] if ga.get("actions")]
    return {"0_to_90_days": p1 or ["Identify top gaps"], "90_to_180_days": p2 or ["Apply learnings"]}


def extract_course_chapters(course_rows, skill_gap_rows):
    gap_skills = {sg["skill_gap_name"].lower(): sg for sg in skill_gap_rows if sg.get("skill_gap_name")}
    courses_out = []
    for cr in course_rows:
        course_id = str(cr.get("course_id", ""))
        course_name = cr.get("course_name", "Untitled")
        raw = cr.get("raw_json") or {}
        if isinstance(raw, str):
            try: raw = json.loads(raw)
            except: raw = {}
        chapters = []
        raw_chapters = raw.get("chapters") or raw.get("topics") or raw.get("modules") or raw.get("sections") or []
        if isinstance(raw_chapters, list):
            for ch in raw_chapters:
                if isinstance(ch, dict):
                    ch_title = ch.get("title") or ch.get("chapter_name") or ch.get("name") or "Untitled"
                    lessons_raw = ch.get("lessons") or ch.get("topics") or ch.get("subtopics") or []
                    lessons = []
                    for lsn in (lessons_raw if isinstance(lessons_raw, list) else []):
                        if isinstance(lsn, str): lessons.append(lsn)
                        elif isinstance(lsn, dict): lessons.append(lsn.get("title") or lsn.get("name") or str(lsn))
                    chapters.append({"title": ch_title, "lessons": lessons})
                elif isinstance(ch, str):
                    chapters.append({"title": ch, "lessons": []})

        duration = raw.get("duration") or raw.get("total_duration") or ""
        if isinstance(duration, (int, float)): duration = f"{int(duration)} hours"
        elif not duration:
            total_lessons = sum(len(ch.get("lessons", [])) for ch in chapters)
            duration = f"~{total_lessons * 30} min" if total_lessons else "Self-paced"

        linked_gap, gap_score = "", 0
        for gn, gd in gap_skills.items():
            if gn in course_name.lower() or any(w in course_name.lower() for w in gn.split() if len(w) > 3):
                linked_gap = gd["skill_gap_name"]
                imp = (gd.get("skill_importance") or "").lower()
                gap_score = 100 if imp == "critical" else 66 if imp == "important" else 33
                break

        course_url = raw.get("url") or raw.get("course_url") or f"https://mynoted.com/course/{course_id}"
        video_resources = []
        for vid in (raw.get("videoResources") or raw.get("videos") or []):
            if isinstance(vid, dict):
                video_resources.append({"title": vid.get("title") or "Video", "url": vid.get("url") or "", "duration": vid.get("duration") or ""})
            elif isinstance(vid, str):
                video_resources.append({"title": "Video", "url": vid, "duration": ""})

        courses_out.append({"courseId": course_id, "courseName": course_name,
            "duration": str(duration), "gapScore": gap_score, "skillGapLinked": linked_gap,
            "url": course_url, "chapters": chapters, "videoResources": video_resources})
    return courses_out


# ════════════════════════════════════════════════════════════════
#  7.  REPORT BUILDER
# ════════════════════════════════════════════════════════════════

class AtlasReportBuilder:
    def __init__(self, db, gpt, run_id, employee_id, max_peers=10):
        self.db = db
        self.gpt = gpt
        self.run_id = run_id
        self.employee_id = employee_id
        self.max_peers = max_peers

    def build(self) -> Dict[str, Any]:
        self.gpt.reset_usage()

        # ─── 1. TARGET ───────────────────────────────────────
        debug("\n[ATLAS] === STEP 1: Loading target ===")
        target_row = self.db.get_employee(self.employee_id)
        if not target_row:
            raise RuntimeError(f"Employee {self.employee_id} not found")
        target_name = target_row["full_name"] or "Unknown"
        target_title = target_row.get("current_title") or ""
        target_company = target_row.get("company_name") or ""
        target_industry = target_row.get("company_industry") or ""
        target_headline = target_row.get("headline") or ""
        target_label = _format_person_label(target_name, target_title, target_company)
        career_domain = _infer_career_domain(target_title, target_headline, target_industry)
        debug(f"[ATLAS] Target: {target_label} | domain: {career_domain}")

        # ─── 2. PEERS ────────────────────────────────────────
        debug("\n[ATLAS] === STEP 2: Loading peers ===")
        match_rows = self.db.get_matches_for_employee(self.run_id, self.employee_id, limit=self.max_peers)
        target_pid = f"p_{target_name.lower().replace(' ', '_')[:20]}"
        participants = [{"id": target_pid, "employee_id": self.employee_id,
            "name": target_label, "isTarget": True, "company": target_company, "title": target_title}]
        seen_ids = {self.employee_id}
        for m in match_rows:
            m_eid = str(m.get("matched_employee_id") or "")
            if m_eid in seen_ids or not m_eid: continue
            seen_ids.add(m_eid)
            m_name = m.get("matched_name") or "Unknown"
            m_title = m.get("matched_title") or ""
            m_company = m.get("matched_company_name") or ""
            m_label = _format_person_label(m_name, m_title, m_company)
            m_pid = f"p_{m_name.lower().replace(' ', '_')[:20]}"
            counter = 2
            base_pid = m_pid
            while any(p["id"] == m_pid for p in participants):
                m_pid = f"{base_pid}_{counter}"; counter += 1
            participants.append({"id": m_pid, "employee_id": m_eid, "name": m_label,
                "isTarget": False, "company": m_company, "title": m_title, "matchScore": m.get("match_score")})
        debug(f"[ATLAS] Participants: {len(participants)}")

        # ─── 3. SKILLS ───────────────────────────────────────
        debug("\n[ATLAS] === STEP 3: Loading skills ===")
        skills_by_person = {}
        all_skills_pool = []
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

        # ─── 4. SKILL CLUSTERS (heatmap) ─────────────────────
        debug("\n[ATLAS] === STEP 4: Skill clusters ===")
        cluster_result = generate_dynamic_clusters(
            gpt=self.gpt, target_name=target_name, target_title=target_title,
            target_company=target_company, target_industry=target_industry,
            target_headline=target_headline, all_skills=all_skills_pool, career_domain=career_domain)
        clusters = cluster_result.get("clusters", {})

        # ─── 5. HEATMAP ──────────────────────────────────────
        debug("\n[ATLAS] === STEP 5: Heatmap ===")
        heatmap_matrix = AtlasEngine.build_heatmap(participants, clusters, skills_by_person)

        # ─── 6. MARKET-PROJECTED AXES (INDEPENDENT) ──────────
        debug("\n[ATLAS] === STEP 6: Market-projected multi-axes ===")
        axes = generate_market_projection_axes(
            gpt=self.gpt, target_title=target_title, target_company=target_company,
            target_industry=target_industry, target_headline=target_headline, career_domain=career_domain)
        debug(f"[ATLAS] X-axes: {[a['label'] for a in axes['xAxes']]}")
        debug(f"[ATLAS] Y-axes: {[a['label'] for a in axes['yAxes']]}")

        # ─── 7. GPT SCORES ALL PERSONS ON ALL AXES ───────────
        debug("\n[ATLAS] === STEP 7: Scoring on all axes ===")
        person_scores = score_persons_on_axes(
            gpt=self.gpt, axes=axes, participants=participants,
            skills_by_person=skills_by_person, career_domain=career_domain)

        # ─── 7b. MARKET BENCHMARKS (where should this role BE?) ──
        debug("\n[ATLAS] === STEP 7b: Market benchmarks ===")
        benchmark_result = generate_market_benchmarks(
            gpt=self.gpt, axes=axes, target_title=target_title,
            target_industry=target_industry, career_domain=career_domain)
        market_benchmarks = benchmark_result["scores"]
        benchmark_rationale = benchmark_result.get("rationale", "")
        debug(f"[ATLAS] Benchmarks: {market_benchmarks}")

        # ─── 8. ALL 49 COMBINATIONS (with market target) ─────
        debug("\n[ATLAS] === STEP 8: 49 axis combinations ===")
        quadrant_combinations = compute_all_axis_combinations(
            axes, person_scores, participants, market_benchmarks)

        # ─── 9. AXIS GAPS (vs peers AND vs market) ───────────
        debug("\n[ATLAS] === STEP 9: Axis gaps ===")
        axis_gaps = compute_axis_gaps(
            target_pid, person_scores, axes, participants, market_benchmarks)

        # ─── 10. CLUSTER GAPS (heatmap) ──────────────────────
        cluster_gaps = AtlasEngine.compute_cluster_gaps(target_pid, heatmap_matrix)

        # ─── 11. SKILL GAPS FROM FRACTAL ──────────────────────
        debug("\n[ATLAS] === STEP 11: Skill gaps ===")
        skill_gap_rows = self.db.get_skill_gaps_for_employee(self.run_id, self.employee_id)

        # ─── 12. COURSES ─────────────────────────────────────
        course_rows = self.db.get_courses_for_employee(self.employee_id)
        primary_course_link = None
        if course_rows:
            first = course_rows[0]
            raw = first.get("raw_json") or {}
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            course_url = raw.get("url") or raw.get("course_url") or f"https://mynoted.com/course/{first.get('course_id', '')}"
            primary_course_link = {"label": first.get("course_name") or "Open Course", "url": course_url}

        # ─── 13. PICK BEST DEFAULT COMBO (aesthetics) ────────
        debug("\n[ATLAS] === STEP 13: Pick best default combo ===")
        best_default = pick_best_default_combo(quadrant_combinations, axes)
        default_x_id = best_default["xAxisId"]
        default_y_id = best_default["yAxisId"]
        default_combo_key = best_default["comboKey"]

        default_combo_data = quadrant_combinations.get(
            default_combo_key, {"positions": [], "marketTarget": {}})
        default_positions = default_combo_data.get("positions", [])
        default_market_target = default_combo_data.get("marketTarget", {})
        target_default_pos = next(
            (p for p in default_positions if p.get("personId") == target_pid),
            {"personId": target_pid, "x": 0, "y": 0, "quadrantLabel": "Lower-Left"})

        # ─── 14. GPT ENHANCEMENTS ────────────────────────────
        debug("\n[ATLAS] === STEP 14: GPT enhancements ===")
        executive_summary = generate_executive_summary(self.gpt, target_label, career_domain, target_default_pos, axis_gaps, skill_gap_rows)
        gap_actions = generate_gap_actions(self.gpt, target_label, career_domain, axis_gaps, skill_gap_rows)
        course_plans = generate_course_plans(self.gpt, target_label, career_domain, skill_gap_rows, course_rows)
        pros_cons = build_swot_structured(self.gpt, target_label, career_domain, target_default_pos, axis_gaps, heatmap_matrix, target_pid, skill_gap_rows)
        peer_descriptions = generate_peer_descriptions(self.gpt, participants, skills_by_person, career_domain, person_scores, axes)
        recommended_courses = extract_course_chapters(course_rows, skill_gap_rows)
        presentation_pack = generate_presentation_pack(self.gpt, target_label, career_domain, executive_summary, axis_gaps, pros_cons, gap_actions, target_default_pos, heatmap_matrix)
        roadmap = generate_roadmap(self.gpt, target_label, career_domain, gap_actions, course_plans, axis_gaps)

        # ─── 15. ASSEMBLE JSON ───────────────────────────────
        debug("\n[ATLAS] === STEP 15: Assembling JSON ===")

        # Resolve default axis labels
        default_x_label = next((a["label"] for a in axes["xAxes"] if a["id"] == default_x_id), "")
        default_y_label = next((a["label"] for a in axes["yAxes"] if a["id"] == default_y_id), "")

        # Peers with default axis scores
        peers_json = []
        for p in participants:
            pid = p["id"]
            scores = person_scores.get(pid, {})
            peer_entry = {
                "id": pid, "name": p["name"], "displayName": p["name"],
                "isTarget": p.get("isTarget", False), "isUser": p.get("isTarget", False),
                "company": p.get("company", ""), "title": p.get("title", ""),
                "xScore": scores.get(default_x_id, 0.0),
                "yScore": scores.get(default_y_id, 0.0),
                "description": peer_descriptions.get(pid, ""),
            }
            if p.get("matchScore"):
                peer_entry["matchScore"] = p["matchScore"]
            peers_json.append(peer_entry)

        # Quadrant for default combo
        quadrant_json = []
        for pos in default_positions:
            rationale_parts = []
            if pos["x"] >= 6 and pos["y"] >= 6:
                rationale_parts.append("Strong across both dimensions.")
            elif pos["x"] >= 6:
                rationale_parts.append(f"Strong on {default_x_label}.")
            elif pos["y"] >= 6:
                rationale_parts.append(f"Strong on {default_y_label}.")
            if pos["x"] < 4:
                rationale_parts.append(f"Limited {default_x_label} — development opportunity.")
            if pos["y"] < 4:
                rationale_parts.append(f"{default_y_label} lighter — room to grow.")
            if not rationale_parts:
                rationale_parts.append("Emerging capabilities.")
            quadrant_json.append({
                "personId": pos["personId"], "personName": pos["personName"],
                "x": pos["x"], "y": pos["y"], "quadrantLabel": pos["quadrantLabel"],
                "rationale": " ".join(rationale_parts),
            })

        skill_gaps_json = [
            {"skill": sg["skill_gap_name"], "importance": sg["skill_importance"],
             "reasoning": sg["gap_reasoning"],
             "description": sg["gap_reasoning"] or f"Gap in {sg['skill_gap_name']}.",
             "category": ("critical" if (sg.get("skill_importance") or "").lower() == "critical"
                          else "important" if (sg.get("skill_importance") or "").lower() == "important"
                          else "nice-to-have"),
             "competitorCompanies": sg["competitor_companies"] if isinstance(sg.get("competitor_companies"), list) else []}
            for sg in skill_gap_rows
        ]

        report = {
            "meta": {
                "targetPerson": target_label,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "version": "2.4", "runId": self.run_id, "employeeId": self.employee_id,
                "careerDomain": career_domain,
            },
            "targetPerson": target_label,
            "executiveSummary": executive_summary,

            # ═══ DEFAULT VIEW — what the frontend loads first ═══
            "defaultView": {
                "xAxisId": default_x_id,
                "yAxisId": default_y_id,
                "xAxisLabel": default_x_label,
                "yAxisLabel": default_y_label,
                "comboKey": default_combo_key,
                "positions": default_positions,
                "marketTarget": default_market_target,
                "quadrant": quadrant_json,
            },

            # ═══ ALL AVAILABLE AXES — dropdown options ═══
            "axisOptions": {
                "xAxes": axes["xAxes"],
                "yAxes": axes["yAxes"],
                "totalCombinations": NUM_AXES * NUM_AXES,
            },

            # ═══ MARKET BENCHMARK — where this role SHOULD be ═══
            "marketBenchmark": {
                "scores": market_benchmarks,
                "rationale": benchmark_rationale,
            },

            # ═══ PERSON SCORES — all persons on all 14 axes ═══
            "personAxisScores": person_scores,

            # ═══ ALL 49 COMBOS — frontend looks up by key ═══
            "quadrantCombinations": quadrant_combinations,

            # ═══ AXIS GAPS — vs market benchmark + vs peers ═══
            "axisGaps": axis_gaps,

            # ═══ PEERS — with default axis scores ═══
            "peers": peers_json,

            # Heatmap (cluster-based, unchanged)
            "heatmapMatrix": heatmap_matrix,
            "heatmapMarkdownTable": build_heatmap_markdown(heatmap_matrix, participants),
            "heatmapLegend": HEATMAP_LEGEND,

            # Cluster data
            "personSkills": build_person_skills_map(participants, skills_by_person),
            "clusterDefinition": {"clusters": [{"name": cn, "skills": cd.get("skills", [])} for cn, cd in clusters.items()]},
            "clusterSkillMap": build_cluster_skill_map(clusters),
            "clusterEvidenceByPerson": build_cluster_evidence_by_person(participants, clusters, skills_by_person),

            # Gaps
            "gaps": cluster_gaps,
            "skillGaps": skill_gaps_json,

            # Actions & courses
            "gapActions": gap_actions,
            "recommendedCourses": recommended_courses,
            "coursePlan": course_plans,

            # SWOT, presentation, roadmap
            "prosAndCons": pros_cons,
            "presentationPack": presentation_pack,
            "roadmap": roadmap,

            # Misc
            "experienceSignals": [], "goalChips": [],
            "primaryCourseLink": primary_course_link or {"label": "Open Course", "url": ""},
            "costSummary": self.gpt.get_cost_summary(),
        }

        # ─── 15. SAVE ────────────────────────────────────────
        debug("\n[ATLAS] === STEP 15: Saving ===")
        self.db.save_employee_report(
            run_id=self.run_id, employee_id=self.employee_id,
            report_json=report, report_type="atlas",
            created_by_agent="atlas_v2.4",
            model_name="gpt-4o" if self.gpt.available else "fallback")
        debug("[ATLAS] Atlas v2.4 report complete")
        return report


# ════════════════════════════════════════════════════════════════
#  8.  ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run_atlas(run_id, employee_id, azure_api_key="", db_host="", db_port=5432,
              db_name="", db_user="", db_password="", max_peers=10):
    print("\n" + "=" * 70)
    print("ATLAS v2.4 — Multi-Axis Dynamic Skill Intelligence Engine")
    print(f"  {NUM_AXES} X-axes x {NUM_AXES} Y-axes = {NUM_AXES**2} combinations")
    print("=" * 70)
    db = SpectreDB(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password)
    gpt = AzureGPT(api_key=azure_api_key)
    try:
        builder = AtlasReportBuilder(db=db, gpt=gpt, run_id=run_id, employee_id=employee_id, max_peers=max_peers)
        report = builder.build()
        print(f"\n Report: {report['targetPerson']}")
        print(f"   Peers: {len(report['peers']) - 1}")
        print(f"   Clusters: {len(report['clusterDefinition']['clusters'])}")
        print(f"   X-Axes: {len(report.get('axisOptions', {}).get('xAxes', []))}")
        print(f"   Y-Axes: {len(report.get('axisOptions', {}).get('yAxes', []))}")
        print(f"   Combinations: {report.get('axisOptions', {}).get('totalCombinations', 0)}")
        print(f"   Axis Gaps: {len(report['axisGaps'])}")
        cost = report.get("costSummary", {})
        print(f"   GPT Calls: {cost.get('totalCalls', 0)}")
        print(f"   Tokens: {cost.get('totalTokens', 0)}")
        print(f"   Cost: ${cost.get('cost', {}).get('totalCost_USD', 0):.4f}")
        return report
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════
#  9.  FASTAPI
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="Atlas v2.4 — Multi-Axis Skill Intelligence API", version="2.4")


class AtlasRequest(BaseModel):
    run_id: str = Field(...)
    employee_id: Optional[str] = Field(None)
    max_peers: int = Field(10, ge=1, le=50)
    azure_api_key: Optional[str] = Field(None)
    db_host: Optional[str] = Field(None)
    db_port: int = Field(5432)
    db_name: Optional[str] = Field(None)
    db_user: Optional[str] = Field(None)
    db_password: Optional[str] = Field(None)


class AtlasResponse(BaseModel):
    success: bool
    target_person: str
    career_domain: str
    run_id: str
    employee_id: str
    axis_count: int = Field(description="Axes per dimension")
    total_combinations: int = Field(description="Pre-computed combos")
    cost_summary: Dict[str, Any] = Field(default_factory=dict)
    report: Dict[str, Any]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "atlas_v2.4", "version": "2.4",
            "axesPerDimension": NUM_AXES, "totalCombinations": NUM_AXES ** 2}


@app.post("/atlas/report", response_model=AtlasResponse)
def generate_atlas_report(req: AtlasRequest):
    try:
        employee_id = req.employee_id
        if not employee_id:
            db = SpectreDB(host=req.db_host or "", port=req.db_port,
                           dbname=req.db_name or "", user=req.db_user or "", password=req.db_password or "")
            try:
                target_row = db.get_target_employee_for_run(req.run_id)
                if not target_row:
                    raise HTTPException(status_code=404, detail=f"No target for run '{req.run_id}'.")
                employee_id = target_row["employee_id"]
            finally:
                db.close()
        report = run_atlas(run_id=req.run_id, employee_id=employee_id,
            azure_api_key=req.azure_api_key or "", db_host=req.db_host or "",
            db_port=req.db_port, db_name=req.db_name or "", db_user=req.db_user or "",
            db_password=req.db_password or "", max_peers=req.max_peers)
        return AtlasResponse(success=True, target_person=report.get("targetPerson", ""),
            career_domain=report.get("meta", {}).get("careerDomain", ""),
            run_id=req.run_id, employee_id=employee_id, axis_count=NUM_AXES,
            total_combinations=NUM_AXES ** 2, cost_summary=report.get("costSummary", {}), report=report)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/atlas/report")
def generate_atlas_report_get(run_id: str = Query(...), employee_id: Optional[str] = Query(None),
                              max_peers: int = Query(10, ge=1, le=50)):
    try:
        eid = employee_id
        if not eid:
            db = SpectreDB()
            try:
                tr = db.get_target_employee_for_run(run_id)
                if not tr: raise HTTPException(status_code=404, detail=f"No target for run '{run_id}'.")
                eid = tr["employee_id"]
            finally:
                db.close()
        report = run_atlas(run_id=run_id, employee_id=eid, max_peers=max_peers)
        return AtlasResponse(success=True, target_person=report.get("targetPerson", ""),
            career_domain=report.get("meta", {}).get("careerDomain", ""),
            run_id=run_id, employee_id=eid, axis_count=NUM_AXES,
            total_combinations=NUM_AXES ** 2, cost_summary=report.get("costSummary", {}), report=report)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# ════════════════════════════════════════════════════════════════
#  10.  CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Atlas v2.4")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--employee-id", required=True)
    parser.add_argument("--max-peers", type=int, default=10)
    parser.add_argument("--output", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.quiet: DEBUG = False
    report = run_atlas(run_id=args.run_id, employee_id=args.employee_id, max_peers=args.max_peers)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.output}")