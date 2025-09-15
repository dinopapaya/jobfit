from typing import List
import io
import re

# ---------- File readers ----------

def _from_pdf(raw: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    txt_parts = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            txt_parts.append(page.extract_text() or "")
    return "\n".join(txt_parts)

def _from_docx(raw: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")
    bio = io.BytesIO(raw)
    d = docx.Document(bio)
    return "\n".join(p.text for p in d.paragraphs)

def _from_txt(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        text = _from_pdf(raw)
    elif name.endswith(".docx"):
        text = _from_docx(raw)
    else:
        text = _from_txt(raw)
    return _normalize(text)

# ---------- Normalization & bulletizing ----------

_BULLET_PREFIXES = r"^[\s>*•▪◦●\-–—\u2022\u25CF\u25E6\u2219]+\s*"

def _normalize(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)   # join hyphenated line breaks
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

HEADING_WORDS = {
    "education","experience","skills","projects","leadership","involvement",
    "honors","awards","summary","objective","responsibilities","requirements",
    "qualifications","activities","coursework","publications","certifications"
}

def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith(":"):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.8 and len(s) <= 40:
            return True
    low = re.sub(r"[^a-z ]", "", s.lower()).strip()
    if low in HEADING_WORDS:
        return True
    parts = [p for p in low.split() if p]
    if 1 <= len(parts) <= 3 and any(p in HEADING_WORDS for p in parts):
        return True
    return False

def to_bullets(text: str) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(_BULLET_PREFIXES, "", raw_line).strip(" •\t-–—")
        if not line:
            continue
        if _looks_like_heading(line):
            continue
        lines.append(line)

    split_more: List[str] = []
    for line in lines:
        if len(line) > 180:
            split_more.extend(re.split(r"(?<=[.;])\s+(?=[A-Z0-9])", line))
        else:
            split_more.append(line)

    bullets: List[str] = []
    seen = set()
    for b in split_more:
        bb = b.strip()
        if len(bb) < 3:
            continue
        key = bb.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(bb)

    return bullets[:200]
