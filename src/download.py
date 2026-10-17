"""Download a stratified sample of rated CodeChef problems.

The public list endpoint returns metadata only. Problem text, topic tags, and
difficulty live on the per-problem endpoint. Tags are kept as labels and are
never used as retrieval features.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CATALOG_PATH = RAW / "catalog.json"
PROBLEMS_PATH = RAW / "problems.jsonl"

API = "https://www.codechef.com/api"
UA = {"User-Agent": "Mozilla/5.0 (research; deja-code educational project)"}
TARGET = 1400
SEED = 17


def fetch_json(url: str, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last}")


def download_catalog() -> list[dict]:
    first = fetch_json(f"{API}/list/problems?start_rating=1&end_rating=5000&limit=100&page=1")
    total = int(first["count"])
    rows = list(first["data"])
    pages = (total + 99) // 100
    print(f"catalog: {total} rated problems, {pages} pages", flush=True)
    for page in range(2, pages + 1):
        payload = fetch_json(
            f"{API}/list/problems?start_rating=1&end_rating=5000&limit=100&page={page}"
        )
        rows.extend(payload.get("data") or [])
        if page % 10 == 0:
            print(f"  page {page}/{pages} ({len(rows)} rows)", flush=True)
        time.sleep(0.05)
    # de-duplicate by code, keep first
    seen = set()
    unique = []
    for row in rows:
        code = row.get("code")
        if not code or code in seen:
            continue
        seen.add(code)
        unique.append(
            {
                "code": code,
                "name": row.get("name") or "",
                "difficulty_rating": int(float(row.get("difficulty_rating") or 0)),
                "successful_submissions": int(float(row.get("successful_submissions") or 0)),
                "total_submissions": int(float(row.get("total_submissions") or 0)),
                "contest_code": row.get("contest_code") or "PRACTICE",
            }
        )
    return unique


def stratified_sample(catalog: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = {}
    for row in catalog:
        bucket = min(row["difficulty_rating"] // 200, 12)
        buckets.setdefault(bucket, []).append(row)
    # proportional, but at least a few from every populated bucket
    selected: list[dict] = []
    for bucket, items in sorted(buckets.items()):
        share = max(8, round(target * len(items) / len(catalog)))
        share = min(share, len(items))
        selected.extend(rng.sample(items, share))
    if len(selected) > target:
        selected = rng.sample(selected, target)
    elif len(selected) < target:
        chosen = {row["code"] for row in selected}
        rest = [row for row in catalog if row["code"] not in chosen]
        rng.shuffle(rest)
        selected.extend(rest[: target - len(selected)])
    return selected


def fetch_problem(meta: dict) -> dict | None:
    contest = meta["contest_code"] or "PRACTICE"
    url = f"{API}/contests/{contest}/problems/{meta['code']}"
    try:
        payload = fetch_json(url, retries=3)
    except RuntimeError:
        if contest != "PRACTICE":
            try:
                payload = fetch_json(
                    f"{API}/contests/PRACTICE/problems/{meta['code']}", retries=2
                )
            except RuntimeError:
                return None
        else:
            return None
    if payload.get("status") != "success":
        return None
    components = payload.get("problemComponents") or {}
    samples = []
    for case in components.get("sampleTestCases") or []:
        if isinstance(case, dict):
            samples.append(
                {
                    "input": case.get("input") or "",
                    "output": case.get("output") or "",
                }
            )
    return {
        "code": meta["code"],
        "title": payload.get("problem_name") or meta["name"],
        "difficulty_rating": int(payload.get("difficulty_rating") or meta["difficulty_rating"] or 0),
        "category": payload.get("category_name") or "",
        "contest_code": payload.get("contest_code") or contest,
        "successful_submissions": meta["successful_submissions"],
        "total_submissions": meta["total_submissions"],
        "user_tags": payload.get("user_tags") or [],
        "computed_tags": payload.get("computed_tags") or [],
        "best_tag": payload.get("best_tag") or "",
        "body": payload.get("body") or "",
        "statement": components.get("statement") or "",
        "input_format": components.get("inputFormat") or "",
        "output_format": components.get("outputFormat") or "",
        "constraints": components.get("constraints") or "",
        "samples": samples[:2],
        "link": f"https://www.codechef.com/problems/{meta['code']}",
    }


def load_done() -> dict[str, dict]:
    done = {}
    if not PROBLEMS_PATH.exists():
        return done
    with PROBLEMS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["code"]] = row
    return done


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        print(f"loaded catalog ({len(catalog)})", flush=True)
    else:
        catalog = download_catalog()
        CATALOG_PATH.write_text(json.dumps(catalog), encoding="utf-8")
        print(f"wrote catalog ({len(catalog)})", flush=True)

    sample = stratified_sample(catalog, TARGET, SEED)
    done = load_done()
    pending = [row for row in sample if row["code"] not in done]
    print(f"sample {len(sample)}, already saved {len(done)}, pending {len(pending)}", flush=True)

    if not pending:
        return

    saved = 0
    with PROBLEMS_PATH.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_problem, row): row["code"] for row in pending}
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    problem = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  skip {code}: {exc}", flush=True)
                    continue
                if problem is None:
                    continue
                handle.write(json.dumps(problem, ensure_ascii=False) + "\n")
                handle.flush()
                saved += 1
                if saved % 50 == 0:
                    print(f"  saved {saved}/{len(pending)}", flush=True)
    print(f"done, appended {saved}", flush=True)


if __name__ == "__main__":
    main()
