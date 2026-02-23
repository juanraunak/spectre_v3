import os
import uuid
from typing import List, Any, Dict
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

DATABASE_URL =     "postgresql://monsteradmin:M0nsteradmin@monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require"

#app = FastAPI(title="Spectre Data API")
router = APIRouter(prefix="/fetch", tags=["Fetch Data"])
def get_connection():
    print("Connecting to database...")
    return psycopg2.connect(DATABASE_URL)


def fetch_all(query: str, params: tuple):
    print("Executing query:", query)
    print("With params:", params)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return results
    finally:
        conn.close()


def fetch_one(query: str, params: tuple):
    print("Executing query:", query)
    print("With params:", params)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result
    finally:
        conn.close()

class RunRequest(BaseModel):
    run_id: uuid.UUID


class CompanyRequest(BaseModel):
    company_id: uuid.UUID


class EmployeeRequest(BaseModel):
    employee_id: uuid.UUID


class SkillRequest(BaseModel):
    skill_id: uuid.UUID

@router.post("/run")
def get_run(req: RunRequest):
    print("POST /run")
    result = fetch_one(
        "SELECT * FROM spectre.runs WHERE run_id = %s",
        (req.run_id,)
    )
    if not result:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


@router.post("/run/companies")
def get_run_companies(req: RunRequest):
    print("POST /run/companies")
    return fetch_all(
        "SELECT * FROM spectre.run_companies WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employees")
def get_run_employees(req: RunRequest):
    print("POST /run/employees")
    return fetch_all(
        "SELECT * FROM spectre.run_employees WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-details")
def get_employee_details(req: RunRequest):
    print("POST /run/employee-details")
    return fetch_all(
        "SELECT * FROM spectre.employee_details WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-reports")
def get_employee_reports(req: RunRequest):
    print("POST /run/employee-reports")
    return fetch_all(
        "SELECT * FROM spectre.employee_reports WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-matches")
def get_employee_matches(req: RunRequest):
    print("POST /run/employee-matches")
    return fetch_all(
        "SELECT * FROM spectre.employee_matches WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-skills")
def get_employee_skills(req: RunRequest):
    print("POST /run/employee-skills")
    return fetch_all(
        "SELECT * FROM spectre.employee_skills WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-skill-gaps")
def get_employee_skill_gaps(req: RunRequest):
    print("POST /run/employee-skill-gaps")
    return fetch_all(
        "SELECT * FROM spectre.employee_skill_gaps WHERE run_id = %s",
        (req.run_id,)
    )


@router.post("/run/employee-courses")
def get_employee_courses(req: RunRequest):
    print("POST /run/employee-courses")
    return fetch_all(
        "SELECT * FROM spectre.employee_courses WHERE run_id = %s",
        (req.run_id,)
    )

@router.post("/company")
def get_company(req: CompanyRequest):
    print("POST /company")
    result = fetch_one(
        "SELECT * FROM spectre.companies WHERE company_id = %s",
        (req.company_id,)
    )
    if not result:
        raise HTTPException(status_code=404, detail="Company not found")
    return result


@router.post("/employee")
def get_employee(req: EmployeeRequest):
    print("POST /employee")
    result = fetch_one(
        "SELECT * FROM spectre.employees WHERE employee_id = %s",
        (req.employee_id,)
    )
    if not result:
        raise HTTPException(status_code=404, detail="Employee not found")
    return result


@router.post("/skill")
def get_skill(req: SkillRequest):
    print("POST /skill")
    result = fetch_one(
        "SELECT * FROM spectre.skills WHERE skill_id = %s",
        (req.skill_id,)
    )
    if not result:
        raise HTTPException(status_code=404, detail="Skill not found")
    return result