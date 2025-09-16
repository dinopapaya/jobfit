"use client";
import { useState } from "react";

type Match = { jd_line: string; resume_bullet: string; similarity: number };
type ScoreOut = {
  score: number;
  skills_found: string[];
  skills_missing: string[];
  matches: Match[];
};
type Tailored = { original: string; suggested: string; match_score: number };

const API = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export default function Home() {
  const [resume, setResume] = useState<File | null>(null);
  const [jd, setJd] = useState("");
  const [result, setResult] = useState<ScoreOut | null>(null);
  const [tailored, setTailored] = useState<Tailored[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!resume || !jd.trim()) {
      setError("Upload a resume and paste a job description.");
      return;
    }
    const form = new FormData();
    form.append("resume", resume);
    form.append("jd", jd);

    setLoading(true);
    try {
      const res = await fetch(`${API}/score`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data: ScoreOut = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err?.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function generateTailored() {
    setTailored(null);
    setError(null);
    if (!resume || !jd.trim()) {
      setError("Upload a resume and paste a job description.");
      return;
    }
    const form = new FormData();
    form.append("resume", resume);
    form.append("jd", jd);
    form.append("k", "6");
    try {
      const res = await fetch(`${API}/tailor`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json(); // { bullets: Tailored[] }
      setTailored(data.bullets);
    } catch (err: any) {
      setError(err?.message ?? "Request failed");
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-5xl p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold">JobFit</h1>
            <p className="text-gray-600 mt-1">Resume ↔ Job Description matcher</p>
          </div>
          <span className="rounded-full border px-3 py-1 text-xs text-gray-600">
            API: {API}
          </span>
        </div>

        <form onSubmit={handleSubmit} className="mt-6 grid gap-4 rounded-2xl bg-white p-5 shadow">
          <div className="grid gap-2">
            <label className="text-sm font-medium">Resume (PDF / DOCX / TXT)</label>
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setResume(e.target.files?.[0] ?? null)}
              className="block w-full rounded border border-gray-300 p-2"
            />
          </div>

          <div className="grid gap-2">
            <label className="text-sm font-medium">Job Description (paste text)</label>
            <textarea
              value={jd}
              onChange={(e) => setJd(e.target.value)}
              placeholder={`Responsibilities:
- Build and ship C++ and Python services.
- Work with NLP embeddings and vector search.
Requirements:
- Experience with Python, C++, and Git.`}
              className="min-h-[180px] w-full rounded border border-gray-300 p-3 placeholder:text-gray-500"
            />
          </div>

          <div className="flex gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 rounded-2xl bg-black px-4 py-3 text-white hover:bg-gray-800 disabled:opacity-60"
            >
              {loading ? "Scoring..." : "Get JobFit score"}
            </button>

            <button
              type="button"
              onClick={generateTailored}
              className="rounded-2xl border border-gray-300 px-4 py-3 hover:bg-gray-100"
            >
              Tailor bullets
            </button>
          </div>

          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </form>

        {result && (
          <section className="mt-6 grid gap-6">
            <div className="rounded-2xl bg-white p-5 shadow">
              <div className="text-sm text-gray-600">Overall JobFit score</div>
              <div className="mt-1 text-5xl font-bold">{Math.round(result.score)}</div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl bg-white p-5 shadow">
                <h2 className="text-lg font-semibold">Skills found</h2>
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.skills_found.length === 0 && (
                    <span className="text-sm text-gray-500">None detected</span>
                  )}
                  {result.skills_found.map((s) => (
                    <span key={s} className="rounded-full bg-green-100 px-3 py-1 text-sm text-green-700">
                      {s}
                    </span>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl bg-white p-5 shadow">
                <h2 className="text-lg font-semibold">Missing skills</h2>
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.skills_missing.length === 0 && (
                    <span className="text-sm text-gray-500">None</span>
                  )}
                  {result.skills_missing.map((s) => (
                    <span key={s} className="rounded-full bg-amber-100 px-3 py-1 text-sm text-amber-800">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded-2xl bg-white p-5 shadow">
              <h2 className="text-lg font-semibold">Line-by-line matches (top 10)</h2>
              <ol className="mt-4 grid gap-3">
                {result.matches.slice(0, 10).map((m, i) => (
                  <li key={i} className="rounded-xl border border-gray-200 p-3">
                    <div className="text-sm text-gray-600">JD:</div>
                    <div className="text-gray-900">{m.jd_line}</div>
                    <div className="mt-2 text-sm text-gray-600">Best resume bullet:</div>
                    <div className="text-gray-900">{m.resume_bullet}</div>
                    <div className="mt-1 text-xs text-gray-500">similarity: {m.similarity.toFixed(2)}</div>
                  </li>
                ))}
              </ol>
            </div>
          </section>
        )}

        {tailored && (
          <section className="mt-6 rounded-2xl bg-white p-5 shadow">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Tailored resume bullets</h2>
              <button
                onClick={() => navigator.clipboard.writeText(tailored.map(t => "• " + t.suggested).join("\n"))}
                className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-100"
              >
                Copy all
              </button>
            </div>
            <ol className="mt-4 grid gap-3">
              {tailored.map((t, i) => (
                <li key={i} className="rounded-xl border border-gray-200 p-3">
                  <div className="text-sm text-gray-600">Original:</div>
                  <div className="text-gray-900">{t.original}</div>
                  <div className="mt-2 text-sm text-gray-600">Suggestion:</div>
                  <div className="text-gray-900">{t.suggested}</div>
                  <div className="mt-1 text-xs text-gray-500">match score: {t.match_score.toFixed(2)}</div>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </main>
  );
}
