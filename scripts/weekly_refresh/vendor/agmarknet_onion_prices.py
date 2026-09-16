#!/usr/bin/env python3
"""
Extract daily Onion price/arrival data from AGMARKNET 2.0 for Indian APMCs.

Default run:
    python agmarknet_onion_prices.py --out data/onion_apmc_daily_2000_2025.csv

Selected states:
    python agmarknet_onion_prices.py ^
      --states "Maharashtra,Karnataka,Madhya Pradesh,Gujarat,Rajasthan,Uttar Pradesh,Delhi" ^
      --out data/onion_major_states_2000_2025.csv

The output is one row per state/market/date/variety observation.
Prices are Rs./quintal and arrivals are metric tonnes as returned by AGMARKNET.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import requests


BASE_URL = "https://api.agmarknet.gov.in/v1"
DEFAULT_COMMODITY = "Onion"
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025

CSV_FIELDS = [
    "source",
    "commodity",
    "commodity_id",
    "state",
    "state_id",
    "state_code",
    "district",
    "district_id",
    "market",
    "market_id",
    "arrival_date",
    "arrival_date_raw",
    "year",
    "month",
    "variety",
    "arrivals_tonnes",
    "total_arrivals_tonnes",
    "min_price_rs_per_quintal",
    "max_price_rs_per_quintal",
    "modal_price_rs_per_quintal",
]


@dataclass(frozen=True)
class StateMeta:
    id: int
    name: str
    code: str | None = None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def norm(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("&amp;", "&")
    text = re.sub(r"\bapmc\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 4,
    timeout: int = 60,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"{response.status_code} from {response.url}", response=response
                )
            response.raise_for_status()
            if sleep_seconds:
                time.sleep(sleep_seconds)
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait = min(60.0, (2 ** (attempt - 1)) + sleep_seconds)
            logging.warning("Retrying after error on %s: %s", url, exc)
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def fetch_states(session: requests.Session, args: argparse.Namespace) -> list[StateMeta]:
    payload = request_json(
        session,
        "/location/state",
        params={"page_size": 100},
        retries=args.retries,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )
    states = [
        StateMeta(
            id=int(item["id"]),
            name=clean_text(item.get("state_name")),
            code=clean_text(item.get("state_code")) or None,
        )
        for item in payload.get("states", [])
        if item.get("id") is not None and item.get("state_name")
    ]
    if not states:
        raise RuntimeError("No states returned by AGMARKNET.")
    return states


def select_states(states: list[StateMeta], names_csv: str | None) -> list[StateMeta]:
    if not names_csv:
        return states

    wanted = [norm(part) for part in names_csv.split(",") if part.strip()]
    by_name = {norm(state.name): state for state in states}
    by_code = {norm(state.code): state for state in states if state.code}
    selected: list[StateMeta] = []
    missing: list[str] = []

    for item in wanted:
        state = by_name.get(item) or by_code.get(item)
        if state:
            selected.append(state)
        else:
            missing.append(item)

    if missing:
        available = ", ".join(state.name for state in states)
        raise ValueError(
            f"Unknown state(s): {', '.join(missing)}. Available states: {available}"
        )

    return selected


def fetch_commodity_id(
    session: requests.Session, commodity_name: str, args: argparse.Namespace
) -> int:
    payload = request_json(
        session,
        "/commodities",
        params={"page_size": 1000},
        retries=args.retries,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )
    matches = [
        item
        for item in payload.get("data", [])
        if norm(item.get("cmdt_name")) == norm(commodity_name)
    ]
    if not matches:
        raise RuntimeError(f"Commodity not found in AGMARKNET list: {commodity_name}")
    return int(matches[0]["id"])


def fetch_market_directory(
    session: requests.Session, args: argparse.Namespace
) -> dict[tuple[int, str], dict[str, Any]]:
    """Return lookup keyed by (state_id, normalized_market_name)."""
    try:
        payload = request_json(
            session,
            "/market-district-state",
            retries=args.retries,
            timeout=args.timeout,
            sleep_seconds=args.sleep,
        )
    except Exception as exc:
        logging.warning("Market directory unavailable, continuing without districts: %s", exc)
        return {}

    rows = payload if isinstance(payload, list) else payload.get("value", [])
    lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        state_id = row.get("state_id")
        market_name = row.get("market_name")
        if state_id is None or not market_name:
            continue
        lookup[(int(state_id), norm(market_name))] = row
    return lookup


def month_payload(
    session: requests.Session,
    state: StateMeta,
    commodity_id: int,
    year: int,
    month: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return request_json(
        session,
        "/prices-and-arrivals/date-wise/specific-commodity",
        params={
            "year": year,
            "month": month,
            "includeExcel": "false",
            "stateId": state.id,
            "commodityId": commodity_id,
        },
        retries=args.retries,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )


def parse_arrival_date(raw: str) -> str:
    raw = clean_text(raw)
    if not raw:
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return raw


def flatten_month(
    payload: dict[str, Any],
    *,
    commodity_name: str,
    commodity_id: int,
    state: StateMeta,
    market_lookup: dict[tuple[int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market_block in payload.get("markets", []) or []:
        market_name = clean_text(market_block.get("marketName"))
        market_meta = market_lookup.get((state.id, norm(market_name)), {})

        for date_block in market_block.get("dates", []) or []:
            arrival_date_raw = clean_text(date_block.get("arrivalDate"))
            arrival_date = parse_arrival_date(arrival_date_raw)
            total_arrivals = to_float(date_block.get("total_arrivals"))
            year = arrival_date[:4] if re.match(r"\d{4}-\d{2}-\d{2}", arrival_date) else ""
            month = arrival_date[5:7] if re.match(r"\d{4}-\d{2}-\d{2}", arrival_date) else ""

            for item in date_block.get("data", []) or []:
                rows.append(
                    {
                        "source": "AGMARKNET 2.0",
                        "commodity": commodity_name,
                        "commodity_id": commodity_id,
                        "state": state.name,
                        "state_id": state.id,
                        "state_code": state.code or "",
                        "district": clean_text(market_meta.get("district_name")),
                        "district_id": market_meta.get("district_id", ""),
                        "market": market_name,
                        "market_id": market_meta.get("market_id", ""),
                        "arrival_date": arrival_date,
                        "arrival_date_raw": arrival_date_raw,
                        "year": year,
                        "month": month,
                        "variety": clean_text(item.get("variety")),
                        "arrivals_tonnes": to_float(item.get("arrivals")),
                        "total_arrivals_tonnes": total_arrivals,
                        "min_price_rs_per_quintal": to_float(item.get("minimumPrice")),
                        "max_price_rs_per_quintal": to_float(item.get("maximumPrice")),
                        "modal_price_rs_per_quintal": to_float(item.get("modalPrice")),
                    }
                )
    return rows


def job_key(state_id: int, year: int, month: int) -> str:
    return f"{state_id}:{year}:{month:02d}"


def read_completed(progress_path: Path) -> set[str]:
    completed: set[str] = set()
    if not progress_path.exists():
        return completed

    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok" and item.get("key"):
                completed.add(str(item["key"]))
    return completed


def write_progress(progress_path: Path, record: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_year_months(start_year: int, end_year: int) -> Iterable[tuple[int, int]]:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download daily Onion prices from AGMARKNET 2.0 by state/month."
    )
    parser.add_argument("--commodity", default=DEFAULT_COMMODITY)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--states",
        default=None,
        help='Comma-separated state names/codes. Omit for all states, e.g. "MH,KA" or "Maharashtra,Karnataka".',
    )
    parser.add_argument(
        "--out",
        default="data/onion_apmc_daily_2000_2025.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the CSV and skip state/month jobs recorded in the .progress.jsonl file.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay after each request.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=4, help="HTTP retry attempts.")
    parser.add_argument(
        "--no-market-directory",
        action="store_true",
        help="Skip market directory lookup. District and market_id columns will be blank.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.start_year > args.end_year:
        raise ValueError("--start-year cannot be greater than --end-year")

    output_path = Path(args.out)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://agmarknet.gov.in",
            "Referer": "https://agmarknet.gov.in/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
        }
    )

    commodity_id = fetch_commodity_id(session, args.commodity, args)
    states = select_states(fetch_states(session, args), args.states)
    market_lookup = {} if args.no_market_directory else fetch_market_directory(session, args)

    completed = read_completed(progress_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"
    write_header = mode == "w"

    logging.info(
        "Starting extraction: commodity=%s id=%s states=%s years=%s-%s output=%s",
        args.commodity,
        commodity_id,
        len(states),
        args.start_year,
        args.end_year,
        output_path,
    )

    total_rows = 0
    total_jobs = len(states) * 12 * (args.end_year - args.start_year + 1)
    finished_jobs = 0

    with output_path.open(mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for state in states:
            for year, month in iter_year_months(args.start_year, args.end_year):
                key = job_key(state.id, year, month)
                if key in completed:
                    finished_jobs += 1
                    continue

                try:
                    payload = month_payload(session, state, commodity_id, year, month, args)
                    rows = flatten_month(
                        payload,
                        commodity_name=args.commodity,
                        commodity_id=commodity_id,
                        state=state,
                        market_lookup=market_lookup,
                    )
                    writer.writerows(rows)
                    csvfile.flush()
                    total_rows += len(rows)
                    finished_jobs += 1

                    write_progress(
                        progress_path,
                        {
                            "key": key,
                            "status": "ok",
                            "state": state.name,
                            "state_id": state.id,
                            "year": year,
                            "month": month,
                            "rows": len(rows),
                            "ts": utc_now_iso(),
                        },
                    )
                    logging.info(
                        "[%s/%s] %s %04d-%02d rows=%s total_rows=%s",
                        finished_jobs,
                        total_jobs,
                        state.name,
                        year,
                        month,
                        len(rows),
                        total_rows,
                    )
                except Exception as exc:
                    write_progress(
                        progress_path,
                        {
                            "key": key,
                            "status": "error",
                            "state": state.name,
                            "state_id": state.id,
                            "year": year,
                            "month": month,
                            "error": str(exc),
                            "ts": utc_now_iso(),
                        },
                    )
                    logging.error("%s %04d-%02d failed: %s", state.name, year, month, exc)

    logging.info("Done. Wrote %s rows to %s", total_rows, output_path)
    logging.info("Progress log: %s", progress_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.", file=sys.stderr)
        raise SystemExit(130)
