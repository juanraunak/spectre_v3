#!/usr/bin/env python3
"""
Global cost tracker for the Spectre pipeline.

- Used by all agents (SHADE, MIRAGE, CIPHER, FRACTAL, SPECTRE SPIDER)
- Tracks GPT token usage, Google CSE calls, and Bright Data rows
- Exposes get_cost_tracker() singleton used across the project
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# -----------------------------
# Pricing (adjust to your real rates)
# -----------------------------

# Base FX
INR_PER_USD = float(os.getenv("INR_PER_USD", "88"))

# Azure GPT pricing (per 1K tokens)
AZURE_GPT_IN_RATE_USD_PER_1K = float(os.getenv("AZURE_GPT_IN_RATE_USD_PER_1K", "0.0025"))
AZURE_GPT_OUT_RATE_USD_PER_1K = float(os.getenv("AZURE_GPT_OUT_RATE_USD_PER_1K", "0.01"))

# Google CSE pricing (per 1K queries)
GOOGLE_CSE_RATE_USD_PER_1K = float(os.getenv("GOOGLE_CSE_RATE_USD_PER_1K", "5.0"))

# Bright Data pricing (per 1K rows / records)
BRIGHT_DATA_RATE_USD_PER_1K = float(os.getenv("BRIGHT_DATA_RATE_USD_PER_1K", "10.0"))


@dataclass
class CostTracker:
    # GPT
    gpt_prompt_tokens: int = 0
    gpt_completion_tokens: int = 0
    gpt_calls: int = 0
    gpt_cost_inr: float = 0.0

    # Google CSE
    google_calls: int = 0
    google_cost_inr: float = 0.0

    # Bright Data
    bright_rows: int = 0
    bright_cost_inr: float = 0.0

    # Extra bucket if needed later
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------ GPT helpers ------------

    def add_gpt_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Primary method to record a GPT call in tokens."""
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)

        self.gpt_calls += 1
        self.gpt_prompt_tokens += prompt_tokens
        self.gpt_completion_tokens += completion_tokens

        usd_cost = (
            (prompt_tokens / 1000.0) * AZURE_GPT_IN_RATE_USD_PER_1K
            + (completion_tokens / 1000.0) * AZURE_GPT_OUT_RATE_USD_PER_1K
        )
        self.gpt_cost_inr += usd_cost * INR_PER_USD

    # Backwards-compatible alias used by agents: track_gpt_tokens(...)
    def track_gpt_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.add_gpt_usage(prompt_tokens, completion_tokens)

    # ------------ Google helpers ------------

    def add_google_call(self, num_calls: int = 1) -> None:
        """Record one or more Google CSE calls."""
        num_calls = int(num_calls or 0)
        if num_calls <= 0:
            return

        self.google_calls += num_calls
        usd_cost = (self.google_calls / 1000.0) * GOOGLE_CSE_RATE_USD_PER_1K
        self.google_cost_inr = usd_cost * INR_PER_USD

    # Alias for agents: track_google_query(...)
    def track_google_query(self, num_items: int = 0) -> None:
        # We don’t care about num_items for cost; each call is roughly same price
        self.add_google_call(1)

    # ------------ Bright Data helpers ------------

    def add_bright_call(self, num_rows: int = 0) -> None:
        """Record Bright Data usage based on number of rows/records fetched."""
        num_rows = int(num_rows or 0)
        if num_rows <= 0:
            return

        self.bright_rows += num_rows
        usd_cost = (self.bright_rows / 1000.0) * BRIGHT_DATA_RATE_USD_PER_1K
        self.bright_cost_inr = usd_cost * INR_PER_USD

    # Alias for agents: track_bright_data_rows(...)
    def track_bright_data_rows(self, num_rows: int) -> None:
        self.add_bright_call(num_rows)

    # ------------ Management ------------

    def reset(self) -> None:
        self.gpt_prompt_tokens = 0
        self.gpt_completion_tokens = 0
        self.gpt_calls = 0
        self.gpt_cost_inr = 0.0
        self.google_calls = 0
        self.google_cost_inr = 0.0
        self.bright_rows = 0
        self.bright_cost_inr = 0.0
        self.extra.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of current costs."""
        total_cost_inr = self.gpt_cost_inr + self.google_cost_inr + self.bright_cost_inr
        return {
            "gpt": {
                "calls": self.gpt_calls,
                "prompt_tokens": self.gpt_prompt_tokens,
                "completion_tokens": self.gpt_completion_tokens,
                "cost_inr": round(self.gpt_cost_inr, 2),
            },
            "google": {
                "calls": self.google_calls,
                "cost_inr": round(self.google_cost_inr, 2),
            },
            "bright_data": {
                "rows": self.bright_rows,
                "cost_inr": round(self.bright_cost_inr, 2),
            },
            "total_cost_inr": round(total_cost_inr, 2),
        }

    def pretty_print(self) -> None:
        """Print a nice human readable summary to stdout."""
        data = self.to_dict()
        print("\n========== COST SUMMARY (ESTIMATED) ==========")
        print("GPT:")
        print(f"  Calls:             {data['gpt']['calls']}")
        print(f"  Prompt tokens:     {data['gpt']['prompt_tokens']}")
        print(f"  Completion tokens: {data['gpt']['completion_tokens']}")
        print(f"  Cost (₹):          {data['gpt']['cost_inr']}")

        print("\nGoogle CSE:")
        print(f"  Calls:             {data['google']['calls']}")
        print(f"  Cost (₹):          {data['google']['cost_inr']}")

        print("\nBright Data:")
        print(f"  Rows:              {data['bright_data']['rows']}")
        print(f"  Cost (₹):          {data['bright_data']['cost_inr']}")

        print("\n---------------------------------------------")
        print(f"TOTAL ESTIMATED COST (₹): {data['total_cost_inr']}")
        print("=============================================\n")


# -----------------------------
# Global singleton
# -----------------------------

_global_cost_tracker: Optional[CostTracker] = None
_cost_lock = threading.Lock()


def get_cost_tracker() -> CostTracker:
    """
    Returns the global CostTracker instance, creating it if needed.
    This is what all agents import and use.
    """
    global _global_cost_tracker
    with _cost_lock:
        if _global_cost_tracker is None:
            _global_cost_tracker = CostTracker()
        return _global_cost_tracker


def reset_cost_tracker() -> None:
    """
    Convenience function: reset the global tracker.
    Useful at the start of each pipeline run.
    """
    tracker = get_cost_tracker()
    tracker.reset()
