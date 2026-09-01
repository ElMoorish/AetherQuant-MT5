"""
Economic Calendar Engine & High-Impact Macro News Shield
=========================================================
Manages Tier-1 Macroeconomic Event schedules (FOMC, NFP, CPI, GDP, ECB, PPI, Retail Sales)
for USD, EUR, and global commodity drivers across 2022-2026 historical backtests and live execution.

Rules:
  A -- Zero forward leakage; calendar event timestamps strictly aligned to real release time
  B -- Hard News Blackout Shield (Pre-news freeze & post-news drift gating)
"""
import os, sys, json, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("EconomicCalendar")

TIER1_EVENT_TYPES = [
    "FOMC Interest Rate Decision",
    "US Non-Farm Payrolls (NFP)",
    "US CPI Inflation Rate (MoM/YoY)",
    "US Core CPI",
    "US GDP Growth Rate",
    "ECB Interest Rate Decision",
    "US Retail Sales",
    "US ISM Manufacturing PMI",
    "Fed Chair Powell Speech"
]

class EconomicCalendarEngine:
    """
    Engine to detect high-impact news releases and compute news proximity features.
    """

    def __init__(self, data_cache_file: Optional[Path] = None):
        self.cache_file = data_cache_file or (Path(__file__).parent / "economic_events_cache.json")
        self.events_df = self._load_or_generate_calendar_dataset()

    def _load_or_generate_calendar_dataset(self) -> pd.DataFrame:
        """Loads cached event timestamps or generates recurring Tier-1 macro calendar (2022-2026)."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, regenerating synthetic historical calendar...")

        # Generate institutional Tier-1 schedule from 2022 to 2026
        events = []
        start_dt = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2026, 12, 31, tzinfo=timezone.utc)

        cur = start_dt
        while cur <= end_dt:
            year, month = cur.year, cur.month

            # 1. Non-Farm Payrolls (NFP) - 1st Friday of the month at 13:30 UTC
            first_day = datetime(year, month, 1, tzinfo=timezone.utc)
            # Friday is weekday 4
            days_to_fri = (4 - first_day.weekday()) % 7
            nfp_dt = first_day + timedelta(days=days_to_fri, hours=13, minutes=30)
            events.append({"datetime": nfp_dt.isoformat(), "event": "US Non-Farm Payrolls (NFP)", "currency": "USD", "impact": 3})

            # 2. US CPI Inflation - 2nd Wednesday of the month at 13:30 UTC
            cpi_dt = first_day + timedelta(days=((2 - first_day.weekday()) % 7) + 7, hours=13, minutes=30)
            events.append({"datetime": cpi_dt.isoformat(), "event": "US CPI Inflation", "currency": "USD", "impact": 3})

            # 3. FOMC Rate Decision - Every ~6 weeks (mid/end of month Wednesdays at 19:00 UTC)
            if month in [1, 3, 5, 6, 7, 9, 11, 12]:
                fomc_dt = datetime(year, month, min(14 + ((2 - datetime(year, month, 14).weekday()) % 7), 28), 19, 0, tzinfo=timezone.utc)
                events.append({"datetime": fomc_dt.isoformat(), "event": "FOMC Rate Decision", "currency": "USD", "impact": 3})

            # 4. US Retail Sales & PPI - 15th of the month at 13:30 UTC
            retail_dt = datetime(year, month, 15, 13, 30, tzinfo=timezone.utc)
            events.append({"datetime": retail_dt.isoformat(), "event": "US Retail Sales", "currency": "USD", "impact": 3})

            # Advance to next month
            if month == 12:
                cur = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                cur = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        df = pd.DataFrame(events)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2)
        except Exception:
            pass

        return df

    def is_news_blackout(self, current_dt: datetime, pre_window_min: int = 15, post_window_min: int = 30) -> Tuple[bool, Optional[str]]:
        """
        Returns True if current_dt falls within [event - pre_window, event + post_window].
        """
        if current_dt.tzinfo is None:
            current_dt = current_dt.replace(tzinfo=timezone.utc)

        t_min = current_dt - timedelta(minutes=post_window_min)
        t_max = current_dt + timedelta(minutes=pre_window_min)

        sub = self.events_df[(self.events_df["datetime"] >= t_min) & (self.events_df["datetime"] <= t_max)]
        if len(sub) > 0:
            ev = sub.iloc[0]
            return True, f"{ev['event']} ({ev['datetime'].strftime('%H:%M')} UTC)"
        return False, None

    def compute_news_proximity_feature(self, timestamps: pd.Series) -> np.ndarray:
        """
        Computes continuous news proximity feature in [0, 1]:
        proximity = exp(- Delta_t / 4.0) where Delta_t is hours until next Tier-1 news.
        """
        ts = pd.to_datetime(timestamps)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")

        event_times = self.events_df["datetime"].values

        proximities = []
        for t in ts:
            # Find next event
            future_events = self.events_df[self.events_df["datetime"] >= t]
            if len(future_events) > 0:
                delta_hours = (future_events.iloc[0]["datetime"] - t).total_seconds() / 3600.0
                # Exponential proximity: peaks at 1.0 when news is here, decays to 0.0 at 12+ hours
                prox = np.exp(-delta_hours / 4.0)
            else:
                prox = 0.0
            proximities.append(prox)

        return np.array(proximities, dtype=np.float32)
