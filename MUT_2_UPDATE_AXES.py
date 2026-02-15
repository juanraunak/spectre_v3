# mutation_2_update_axes.py
from __future__ import annotations

import os
import json
import logging
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - MUTATION2 - %(levelname)s - %(message)s")
log = logging.getLogger("mutation2")

# -------------------------
# DB config / helper
# -------------------------
DEFAULT_DSN = os.getenv(
    "SPECTRE_DB_URL",
    "postgresql://monsteradmin:M0nsteradmin@monsterdb.postgres.database.azure.com:5432/postgres?sslmode=require",
)

def _get_conn():
    dsn = os.getenv("SPECTRE_DB_URL", DEFAULT_DSN)
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn

class Mutation2DB:
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
        try:
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def rollback(self):
        try:
            self.conn.rollback()
        except Exception:
            pass

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT run_id, scope, status FROM spectre.runs WHERE run_id = %s", (run_id,))
            return cur.fetchone()

    def get_target_employee_for_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT re.employee_id, e.full_name, e.current_title
                FROM spectre.run_employees re
                JOIN spectre.employees e ON e.employee_id = re.employee_id
                WHERE re.run_id = %s
                ORDER BY re.created_at ASC
                LIMIT 1
            """, (run_id,))
            return cur.fetchone()

    def get_current_atlas_report(self, run_id: str, employee_id: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT report_version, report_json, created_at, model_name,
                       created_by_agent, created_by_mutation, mutation_summary, version_created_at
                FROM spectre.employee_reports
                WHERE run_id = %s AND employee_id = %s AND report_type = 'atlas'
                LIMIT 1
            """, (run_id, employee_id))
            return cur.fetchone()

    def archive_current_report(self, run_id: str, employee_id: str) -> Optional[int]:
        """
        Copy current 'atlas' row to 'atlas_v{N}' (same pattern as Mutation 1).
        Returns previous version number (int) if present, else None.
        """
        row = self.get_current_atlas_report(run_id, employee_id)
        if not row:
            return None

        try:
            current_v = int(row["report_version"] or 0)
        except Exception:
            current_v = 0

        archive_type = f"atlas_v{current_v}"
        report_json_val = row["report_json"]
        if isinstance(report_json_val, (dict, list)):
            report_json_val = json.dumps(report_json_val)

        with self.conn.cursor() as cur:
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
                row.get("created_at"), row.get("model_name"), row.get("created_by_agent"),
                str(current_v), row.get("created_by_mutation"), row.get("mutation_summary"),
                row.get("version_created_at") or row.get("created_at"),
            ))
        self.commit()
        return current_v

    def upsert_atlas_report(
        self,
        run_id: str,
        employee_id: str,
        report_json: Dict[str, Any],
        new_version: int,
        mutation_summary: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO spectre.employee_reports (
                    run_id, employee_id, report_type, report_json,
                    created_at, model_name, created_by_agent,
                    report_version, created_by_mutation, mutation_summary,
                    version_created_at
                ) VALUES (%s, %s, 'atlas', %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, employee_id, report_type) DO UPDATE SET
                    report_json = EXCLUDED.report_json,
                    report_version = EXCLUDED.report_version,
                    created_by_mutation = EXCLUDED.created_by_mutation,
                    mutation_summary = EXCLUDED.mutation_summary,
                    version_created_at = EXCLUDED.version_created_at,
                    updated_at = NOW()
            """, (
                run_id, employee_id,
                json.dumps(report_json, ensure_ascii=False),
                now,
                "mutation-operator",
                "mutation_2",
                str(new_version),
                "mutation_2_update_axes",
                mutation_summary,
                now
            ))
        self.commit()

# -------------------------
# FastAPI app + models
# -------------------------
app = FastAPI(
    title="Mutation 2 — Update X/Y Axis",
    description="Creates a new version of the atlas report by replacing defaultView.xAxisId / defaultView.yAxisId.",
    version="1.0.0",
)

class UpdateAxesRequest(BaseModel):
    run_id: str = Field(..., description="Run UUID")
    new_x_axis: str = Field(..., description="New X axis id (matches axisOptions.xAxes[].id)")
    new_y_axis: str = Field(..., description="New Y axis id (matches axisOptions.yAxes[].id)")
    employee_id: Optional[str] = Field(None, description="Optional employee_id; if omitted, resolve target for run")
    reason: Optional[str] = Field(None, description="Optional reason text to store in mutation_summary")

class UpdateAxesResponse(BaseModel):
    success: bool
    run_id: str
    employee_id: str
    previous_version: Optional[int]
    archived_as: Optional[str]
    new_version: int
    mutation_summary: str

@app.get("/mutation/health")
def health():
    return {"status": "ok", "agent": "mutation_2_update_axes", "version": "1.0.0"}

@app.post("/mutation/update-axes", response_model=UpdateAxesResponse)
def update_axes(req: UpdateAxesRequest, dry_run: bool = Query(False, description="If true, return the mutated JSON diff but do not write to DB.")):
    db = Mutation2DB()
    try:
        run_id = (req.run_id or "").strip()
        if not run_id:
            raise HTTPException(status_code=400, detail="run_id is required")

        run_info = db.get_run(run_id)
        if not run_info:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        employee_id = (req.employee_id or "").strip()
        if not employee_id:
            target = db.get_target_employee_for_run(run_id)
            if not target:
                raise HTTPException(status_code=404, detail=f"No employees found for run {run_id}")
            employee_id = str(target["employee_id"])

        current = db.get_current_atlas_report(run_id, employee_id)
        if not current:
            raise HTTPException(status_code=404, detail="No existing atlas report to mutate")

        prev_version = int(current.get("report_version") or 0)
        prev_json = current.get("report_json") or {}
        if not isinstance(prev_json, dict):
            prev_json = {}

        # --- Archive current atlas report as atlas_v{prev_version}
        archived_v = db.archive_current_report(run_id, employee_id)
        archived_as = f"atlas_v{archived_v}" if archived_v is not None else None

        # --- Mutate JSON using exact keys from your sample JSON:
        new_json = json.loads(json.dumps(prev_json))  # deep copy

        # Ensure defaultView exists
        default_view = new_json.get("defaultView")
        if not isinstance(default_view, dict):
            default_view = {}

        # Prev axis ids for history
        prev_x = default_view.get("xAxisId")
        prev_y = default_view.get("yAxisId")

        # Set new ids
        default_view["xAxisId"] = req.new_x_axis
        default_view["yAxisId"] = req.new_y_axis

        # Update comboKey (front-end sometimes uses this)
        default_view["comboKey"] = f"{req.new_x_axis}_{req.new_y_axis}"

        # Try to pull human-friendly labels from axisOptions if available
        axis_opts = new_json.get("axisOptions") or {}
        x_label = None
        y_label = None
        try:
            x_axes_list = axis_opts.get("xAxes") or []
            y_axes_list = axis_opts.get("yAxes") or []
            for item in x_axes_list:
                if isinstance(item, dict) and item.get("id") == req.new_x_axis:
                    x_label = item.get("label") or item.get("name") or None
                    break
            for item in y_axes_list:
                if isinstance(item, dict) and item.get("id") == req.new_y_axis:
                    y_label = item.get("label") or item.get("name") or None
                    break
        except Exception:
            # If axisOptions isn't as expected, silently ignore
            pass

        # Set axis labels (fallback to ids if no friendly label)
        default_view["xAxisLabel"] = x_label or default_view.get("xAxisLabel") or req.new_x_axis
        default_view["yAxisLabel"] = y_label or default_view.get("yAxisLabel") or req.new_y_axis

        # Place defaultView back
        new_json["defaultView"] = default_view

        # Append mutation history
        new_json.setdefault("_mutation_history", [])
        new_json["_mutation_history"].append({
            "mutation": "mutation_2_update_axes",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "by": "mutation_2",
            "prev_x_axis": prev_x,
            "prev_y_axis": prev_y,
            "new_x_axis": req.new_x_axis,
            "new_y_axis": req.new_y_axis,
            "reason": req.reason or ""
        })

        # Idempotency: if no change and not dry_run, return success without creating new version
        if not dry_run and prev_x == req.new_x_axis and prev_y == req.new_y_axis:
            mutation_summary = f"mutation_2_update_axes: no-op (axes unchanged) v{prev_version}"
            return UpdateAxesResponse(
                success=True,
                run_id=run_id,
                employee_id=employee_id,
                previous_version=prev_version,
                archived_as=None,
                new_version=prev_version,
                mutation_summary=mutation_summary
            )

        # If dry_run, return the mutated JSON details without writing to DB
        if dry_run:
            # Compute a small diff for convenience (previous vs new)
            diff = {
                "prev_defaultView": {
                    "xAxisId": prev_x,
                    "yAxisId": prev_y,
                    "xAxisLabel": (prev_json.get("defaultView") or {}).get("xAxisLabel"),
                    "yAxisLabel": (prev_json.get("defaultView") or {}).get("yAxisLabel"),
                    "comboKey": (prev_json.get("defaultView") or {}).get("comboKey"),
                },
                "new_defaultView": {
                    "xAxisId": new_json["defaultView"].get("xAxisId"),
                    "yAxisId": new_json["defaultView"].get("yAxisId"),
                    "xAxisLabel": new_json["defaultView"].get("xAxisLabel"),
                    "yAxisLabel": new_json["defaultView"].get("yAxisLabel"),
                    "comboKey": new_json["defaultView"].get("comboKey"),
                },
            }
            # Return a 200 with the would-be version (prev+1)
            return UpdateAxesResponse(
                success=True,
                run_id=run_id,
                employee_id=employee_id,
                previous_version=prev_version,
                archived_as=None,
                new_version=prev_version + 1,
                mutation_summary=f"DRY_RUN: would apply mutation_2_update_axes v{prev_version+1}; diff={json.dumps(diff)}"
            )

        # Build mutation_summary text
        mutation_summary = (
            f"mutation_2_update_axes: v{prev_version} -> v{prev_version+1}; "
            f"x={req.new_x_axis},y={req.new_y_axis}"
        )
        if req.reason:
            mutation_summary += f"; reason={req.reason}"

        # Upsert mutated atlas as version+1
        new_version = prev_version + 1
        db.upsert_atlas_report(run_id, employee_id, new_json, new_version, mutation_summary)

        return UpdateAxesResponse(
            success=True,
            run_id=run_id,
            employee_id=employee_id,
            previous_version=prev_version,
            archived_as=archived_as,
            new_version=new_version,
            mutation_summary=mutation_summary
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.exception("Mutation 2 failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Mutation 2 — Update report defaultView X/Y axis")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        # run the app object directly so filename / module name doesn't matter
        uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
