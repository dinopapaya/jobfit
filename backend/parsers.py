from typing import List
import io, re

# ---------- Prefer PyMuPDF (fitz) for PDFs; fallback to pdfplumber ----------

def _from_pdf(raw: bytes) -> str:
    # Try PyMuPDF blocks first (handles multi-column better)
    try:
        import fitz  # PyMuPDF
        text_parts = []
        with fitz.open(stream=raw, filetype="pdf") as doc:
            for page in doc:
                # get_text("blocks") -> tuples (..., text, ...)
                blocks = page.get_text("blocks")
                # sort by (y, x) to preserve reading order
                blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
                for b in blocks:
                    t = (b[4] or "").strip()
                    if t:
                        text_parts.append(t)
        if text_parts:
            return "\n".join(text_parts)
    except Exception:
        pass

    # Fallback: pdfplumber
    try:
        import pdfplumber
        txt_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                txt_parts.append(page.extract_text() or "")
        return "\n".join(txt_parts)
    except Exception:
        # last resort: return empty; caller handles
        return ""

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
    # join hyphenated line breaks: "develop-\nment" -> "development"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

HEADING_WORDS = {
    "education","experience","skills","projects","leadership","involvement",
    "honors","awards","summary","objective","responsibilities","requirements",
    "qualifications","activities","coursework","publications","certifications",
    "title","about","profile","contact","languages"
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

def _stitch_short_lines(lines: List[str]) -> List[str]:
    """
    Merge short fragments into a preceding line until they form a reasonable sentence.
    Good for PDFs that split bullets like: "Monitor performance" / "performance."
    """
    out: List[str] = []
    buf = ""
    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for line in lines:
        l = line.strip()
        if not l:
            continue
        if not buf:
            buf = l
            continue
        # If previous is short (<5 tokens) OR doesn't end with sentence punctuation, keep appending.
        if len(buf.split()) < 5 or not re.search(r"[.!?)]$", buf):
            buf = f"{buf} {l}"
        else:
            flush()
            buf = l
    flush()
    return out

def _spacy_split_paragraphs(lines: List[str]) -> List[str]:
    """Optional: split longer paragraphs into sentences with spaCy if available."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return lines  # spaCy not available; skip
    out: List[str] = []
    for ln in lines:
        if len(ln) > 180:
            for s in nlp(ln).sents:
                ss = s.text.strip()
                if ss:
                    out.append(ss)
        else:
            out.append(ln)
    return out

def to_bullets(text: str) -> List[str]:
    if not text:
        return []

    # 1) strip bullet prefixes & headings
    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(_BULLET_PREFIXES, "", raw_line).strip(" •\t-–—")
        if not line:
            continue
        if _looks_like_heading(line):
            continue
        lines.append(line)

    # 2) Stitch short fragments from bad PDF splits
    lines = _stitch_short_lines(lines)

    # 3) Split long paragraphs via spaCy (if available)
    lines = _spacy_split_paragraphs(lines)

    # 4) cleanup, dedupe, drop very short/noisy bits
    bullets: List[str] = []
    seen = set()
    for b in lines:
        bb = b.strip()
        # keep if has >=4 tokens OR contains digits/% OR ends with punctuation
        tokens = re.findall(r"[A-Za-z0-9\+\.\-]+", bb)
        if len(tokens) < 4 and not re.search(r"(\d|%)", bb) and not re.search(r"[.!?)]$", bb):
            continue
        key = bb.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(bb)

    return bullets[:250]  # safety cap
