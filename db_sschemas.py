from datetime import datetime, timezone
from typing import Optional, Any, Dict
from uuid import UUID
from pydantic import BaseModel, Field

class Company(BaseModel):
    company_id: UUID
    name: str
    domain: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_type: str = "unknown"
    metadata_json: Optional[Dict[str, Any]] = None
    raw_json: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    business_model: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None

class Employee(BaseModel):
    employee_id: UUID
    full_name: str
    linkedin_url: str
    canonical_linkedin_id: Optional[str] = None
    current_company_id: Optional[UUID] = None
    current_title: Optional[str] = None
    location: Optional[str] = None
    headline: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    raw_json: Optional[Dict[str, Any]] = None
    last_processed_run_id: Optional[UUID] = None
    last_processed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    profile_cache_text: Optional[str] = None

class Run(BaseModel):
    run_id: UUID
    scope: Optional[str] = None
    status: str = "queued"
    target_company_id: Optional[UUID] = None
    requested_employee_count: Optional[int] = None
    requested_linkedin_url: Optional[str] = None
    allow_cache: bool = True
    freshness_days: int = 30
    force_refresh: bool = False
    config_json: Optional[Dict[str, Any]] = None
    raw_json: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    candidate_pool: Optional[Dict[str, Any]] = None
    interaction_status: Optional[str] = None


class Skill(BaseModel):
    skill_id: UUID
    name: str
    category: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    raw_json: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))


class EmployeeSkill(BaseModel):
    run_id: UUID
    employee_id: UUID
    skill_id: UUID
    skill_confidence: Optional[float] = None
    level: Optional[str] = None
    rationale_json: Optional[Dict[str, Any]] = None
    raw_json: Dict[str, Any]
    created_by_agent: str = "CIPHER"
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))


class EmployeeSkillGap(BaseModel):
    run_id: UUID
    employee_id: UUID
    skill_id: UUID
    skill_gap_name: Optional[str] = None
    skill_importance: Optional[str] = None
    gap_reasoning: Optional[str] = None
    competitor_companies: Optional[Dict[str, Any]] = None
    raw_json: Dict[str, Any]
    created_by_agent: str = "AGENT4_SKILL_GAPS"
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))


class EmployeeReport(BaseModel):
    run_id: UUID
    employee_id: UUID
    report_json: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    report_type: str = "atlas"
    created_by_agent: Optional[str] = None
    model_name: Optional[str] = None
    updated_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    raw_json: Optional[Dict[str, Any]] = None
    report_version: Optional[str] = None
    created_by_mutation: Optional[str] = None
    mutation_summary: Optional[str] = None
    version_created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

class EmployeeMatch(BaseModel):
    run_id: UUID
    employee_id: UUID
    matched_employee_id: Optional[UUID] = None
    matched_name: str
    matched_title: Optional[str] = None
    matched_company_id: UUID
    matched_company_name: Optional[str] = None
    match_score: Optional[float] = None
    match_type: str = "peer"
    match_source: Optional[str] = None
    rationale_json: Optional[Dict[str, Any]] = None
    raw_json: Dict[str, Any]
    created_by_agent: str = "MIRAGE"
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))


class EmployeeCourse(BaseModel):
    employee_id: UUID
    course_id: UUID
    course_name: str
    raw_json: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    course_version: int = 0
    created_by: str = "original"
    run_id: Optional[UUID] = None

class EmployeeDetail(BaseModel):
    run_id: UUID
    employee_id: UUID
    data_origin: str
    details_json: Optional[Dict[str, Any]] = None
    raw_json: Optional[Dict[str, Any]] = None
    created_by_agent: str
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    rp_current_company_company_id: Optional[str] = None
    v_location: Optional[str] = None
    v_company: Optional[str] = None
    v_title: Optional[str] = None
    v_about: Optional[str] = None
    v_name: Optional[str] = None
    v_city: Optional[str] = None
    v_country_code: Optional[str] = None
    v_employee_id: Optional[str] = None
    v_connections: Optional[str] = None
    v_followers: Optional[str] = None
    v_position: Optional[str] = None
    v_experience: Optional[str] = None
    v_languages: Optional[str] = None
    v_education: Optional[str] = None
    v_activity: Optional[str] = None
    v_courses: Optional[str] = None

    rp_last_name: Optional[str] = None
    rp_first_name: Optional[str] = None
    rp_banner_image: Optional[str] = None
    rp_avatar: Optional[str] = None
    rp_url: Optional[str] = None
    rp_linkedin_profile_id: Optional[str] = None

class RunCompany(BaseModel):
    run_id: UUID
    company_id: UUID
    role_in_run: str = "competitor"
    raw_json: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    role: Optional[str] = "competitor"

class RunEmployee(BaseModel):
    run_id: UUID
    employee_id: UUID
    role_in_run: str
    source_company_id: Optional[UUID] = None
    raw_json: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))