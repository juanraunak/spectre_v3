-- =============================================================================
-- MIRAGE Interactive Selection: DB Migration
-- =============================================================================
-- Adds two columns to spectre.runs to persist state between the discovery
-- phase and the user-confirmation phase, so the same run_id is reused
-- throughout and the downstream pipeline sees no changes.
-- =============================================================================

-- 1. Persist all discovered candidates after Phase 3+4.
--    Stored as a JSONB array; each element is a "candidate card" the UI
--    can display to the user.
--    Shape per element:
--    {
--      "rank":               1,
--      "linkedin_url":       "https://www.linkedin.com/in/...",
--      "name":               "Jane Doe",
--      "title":              "Senior Product Manager",
--      "company":            "Acme Corp",
--      "similarity_score":   87.4,
--      "confidence":         "high",
--      "employee_id":        "<uuid>",          -- already written to spectre.employees
--      "match_rationale":    "...",
--      "matching_factors":   {...},
--      "source":             "discovered" | "user_added"
--    }
ALTER TABLE spectre.runs
    ADD COLUMN IF NOT EXISTS candidate_pool     JSONB    DEFAULT NULL;

-- 2. Track which phase the run is currently in so the confirm endpoint can
--    validate that discovery has already completed before running enrichment.
--    Values: 'discovery_complete' | 'confirmed' | (existing: 'in_progress' | 'completed')
--    We reuse the existing `status` column for coarse state and add this for
--    the intermediate sub-state, to avoid touching status semantics the
--    downstream pipeline depends on.
ALTER TABLE spectre.runs
    ADD COLUMN IF NOT EXISTS interaction_status VARCHAR(64) DEFAULT NULL;

-- Optional index so the confirm endpoint can quickly look up a run by its
-- interaction_status without a full scan (useful if the table is large).
CREATE INDEX IF NOT EXISTS idx_runs_interaction_status
    ON spectre.runs (interaction_status)
    WHERE interaction_status IS NOT NULL;

-- Verify
SELECT column_name, data_type
FROM   information_schema.columns
WHERE  table_schema = 'spectre'
  AND  table_name   = 'runs'
  AND  column_name  IN ('candidate_pool', 'interaction_status')
ORDER BY column_name;