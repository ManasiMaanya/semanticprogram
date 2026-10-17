"""Clean CodeChef statements and separate topic tags from author handles."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "problems.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "problems.jsonl"

BOILERPLATE = [
    r"all submissions for this problem are available\.?",
    r"read problems statements in [^.]+(?:and [^.]+)?\.?",
    r"this is an example problem statement in markdown[\s\S]{0,400}?below is an example problem statement[^\n]*",
    r"please make sure to remove everything here before publishing your problem\.",
    r"codechef uses markdown for its problem statements\.",
    r"the input and output formats given below are only if you want to test using custom inputs\.?",
    r"function declaration",
]

TEMPLATE_MARKERS = (
    "this is an example problem statement",
    "mini guide on writing statements",
)


def html_to_text(value: str) -> str:
    if not value:
        return ""
    text = unescape(value)
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "aside"]):
            tag.decompose()
        text = soup.get_text("\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\(leq|geq|times|cdot|le|ge)", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_concept_tag(tag: str) -> bool:
    tag = (tag or "").strip()
    if len(tag) < 2 or len(tag) > 60:
        return False
    # Author handles on this API are lowercase tokens. Concept tags are phrases
    # or title-case words ("Dynamic Programming", "Factorials", "Graphs").
    if re.fullmatch(r"[a-z0-9_.]+", tag):
        return False
    if tag.lower() in {
        "practice",
        "easy",
        "medium",
        "hard",
        "school",
        "challenge",
        "codechef",
        "author",
        "editorial",
        "admin",
    }:
        return False
    return True


def collect_tags(row: dict) -> list[str]:
    raw = []
    for key in ("user_tags", "computed_tags"):
        value = row.get(key) or []
        if isinstance(value, str):
            value = [value]
        raw.extend(value)
    if row.get("best_tag"):
        raw.append(row["best_tag"])
    cleaned = []
    seen = set()
    for tag in raw:
        if not is_concept_tag(str(tag)):
            continue
        norm = re.sub(r"\s+", " ", str(tag)).strip()
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(norm)
    return cleaned


def narrative_text(text: str) -> str:
    """Drop input/output templates. They are shared by unrelated problems and drown out the task."""
    parts = re.split(
        r"\n\s*(?:input(?: format)?|output(?: format)?|constraints|sample input|sample output|subtasks?|explanation)\s*:?\s*\n",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return parts[0].strip()


def build_document(row: dict) -> str:
    statement = html_to_text(row.get("statement") or "")
    body = html_to_text(row.get("body") or "")
    # The structured statement is the problem. Fall back to the HTML body when
    # the contest still stores the statement only in `body`.
    narrative = statement if len(statement) >= 40 else body
    narrative = narrative_text(narrative)
    title = (row.get("title") or "").strip()
    document = f"{title}\n{narrative}".strip() if narrative else title
    document = re.sub(r"\n{3,}", "\n\n", document).strip()
    return document[:4000]


def is_usable(document: str) -> bool:
    lowered = document.lower()
    if any(marker in lowered for marker in TEMPLATE_MARKERS):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", document)
    return len(words) >= 40


def load_raw(path: Path = RAW_PATH) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preprocess(rows: list[dict] | None = None) -> pd.DataFrame:
    rows = rows if rows is not None else load_raw()
    records = []
    for row in rows:
        document = build_document(row)
        if not is_usable(document):
            continue
        tags = collect_tags(row)
        records.append(
            {
                "code": row["code"],
                "title": row.get("title") or row["code"],
                "difficulty": int(row.get("difficulty_rating") or 0),
                "category": row.get("category") or "",
                "tags": tags,
                "tag_text": " | ".join(tags),
                "successful_submissions": int(row.get("successful_submissions") or 0),
                "link": row.get("link") or f"https://www.codechef.com/problems/{row['code']}",
                "text": document,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("preprocessing removed every problem; check the raw download")
    frame = frame.drop_duplicates("code").reset_index(drop=True)
    return frame


def save(frame: pd.DataFrame, path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    frame = preprocess()
    path = save(frame)
    tagged = int((frame["tags"].map(len) > 0).sum())
    print(f"kept {len(frame)} problems, {tagged} with concept tags -> {path}")


if __name__ == "__main__":
    main()
