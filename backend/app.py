from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from parsers import extract_text, to_bullets
from scorer import jobfit, parse_required_skills_from_jd, tailor_bullets
import os

app = FastAPI(title="JobFit API")

# ---- CORS: localhost + any *.vercel.app preview/prod ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CT = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}
MAX_BYTES = 8_000_000  # 8 MB

class AnalyzeOut(BaseModel):
    resume_bullets: list[str]
    jd_lines: list[str]
    counts: dict

class ScoreOut(BaseModel):
    score: float
    skills_found: list[str]
    skills_missing: list[str]
    matches: list[dict]

class TailorOut(BaseModel):
    bullets: list[dict]

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"msg": "JobFit backend running!"}

def _check_content_type(resume: UploadFile):
    if resume.content_type not in ALLOWED_CT:
        raise HTTPException(status_code=422, detail="Upload PDF, DOCX, or TXT.")

def _check_size(raw: bytes):
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (>8 MB).")

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(resume: UploadFile, jd: str = Form(...)):
    _check_content_type(resume)
    raw = await resume.read()
    _check_size(raw)

    rtxt = extract_text(resume.filename, raw)
    rbul = to_bullets(rtxt)
    jd_lines = to_bullets(jd)

    return AnalyzeOut(
        resume_bullets=rbul[:50],
        jd_lines=jd_lines[:50],
        counts={"resume_bullets": len(rbul), "jd_lines": len(jd_lines)},
    )

@app.post("/score", response_model=ScoreOut)
async def score(resume: UploadFile, jd: str = Form(...)):
    _check_content_type(resume)
    raw = await resume.read()
    _check_size(raw)

    rtxt = extract_text(resume.filename, raw)
    rbul = to_bullets(rtxt)
    jd_lines = to_bullets(jd)
    req_skills = parse_required_skills_from_jd(jd)
    return ScoreOut(**jobfit(rbul, jd_lines, req_skills))

@app.post("/tailor", response_model=TailorOut)
async def tailor(resume: UploadFile, jd: str = Form(...), k: int = 6):
    _check_content_type(resume)
    raw = await resume.read()
    _check_size(raw)

    rtxt = extract_text(resume.filename, raw)
    rbul = to_bullets(rtxt)
    jd_lines = to_bullets(jd)
    req_skills = parse_required_skills_from_jd(jd)
    bullets = tailor_bullets(rbul, jd_lines, req_skills, k=k)
    return TailorOut(bullets=bullets)
