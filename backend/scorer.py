from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import re
import os

# -----------------------------
# Model choices (switchable)
# -----------------------------
EMB_MODEL = os.getenv("JOBFIT_EMB_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
USE_RERANKER = os.getenv("JOBFIT_USE_RERANKER", "1") == "1"
RERANKER_MODEL = os.getenv("JOBFIT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOPK_FOR_RERANK = int(os.getenv("JOBFIT_TOPK_FOR_RERANK", "5"))  # re-rank top-k resume bullets per JD line

# small, high-quality embedding model
_model = SentenceTransformer(EMB_MODEL)
_reranker = CrossEncoder(RERANKER_MODEL) if USE_RERANKER else None

# ===== BIG SKILL BANK (lowercase) =====
SKILL_BANK = {
    # Languages
    "python","c","c++","c#","java","kotlin","scala","go","golang","rust","ruby","php",
    "javascript","typescript","bash","shell","powershell","matlab","r","swift","objective-c",
    "sql","nosql",
    # Backend / APIs
    "fastapi","flask","django","starlette","tornado","aiohttp","express","express.js",
    "node","nodejs","node.js","spring","spring boot","quarkus","ktor","grpc","rest","rest api",
    "graphql","apollo","hasura","websocket","socket.io","openapi","swagger","oauth","oauth2","jwt",
    # Frontend
    "react","react.js","reactjs","next","next.js","nextjs","redux","tanstack query","react query",
    "vue","vue.js","nuxt","nuxt.js","angular","svelte","sveltekit","vite","webpack","rollup",
    "tailwind","tailwind css","bootstrap","material ui","mui","chakra ui","d3","chart.js","three.js",
    "leaflet","mapbox","plotly",
    # Mobile
    "android","ios","swiftui","jetpack compose","react native","flutter","dart","xamarin",
    # Databases
    "postgres","postgresql","mysql","mariadb","sqlite","oracle","sql server","mssql",
    "snowflake","redshift","bigquery","synapse","clickhouse","duckdb","cassandra","dynamodb",
    "cosmos db","couchdb","couchbase","mongodb","neo4j","arangodb","redis","memcached",
    "elasticsearch","opensearch","solr","lucene","pgvector",
    # Data eng / streaming
    "airflow","luigi","prefect","dagster","dbt","kafka","redpanda","pulsar","rabbitmq",
    "sqs","sns","pubsub","pub/sub","kinesis","flink","spark","pyspark","hadoop","hive",
    "presto","trino","beam","apache beam","emr","glue","athena","delta lake","iceberg","hudi",
    "parquet","avro","orc","csv","json","xml","yaml","protobuf",
    # ML / DS
    "numpy","pandas","scipy","scikit-learn","xgboost","lightgbm","catboost","statsmodels",
    "tensorflow","keras","pytorch","torch","jax","prophet","opencv","matplotlib","seaborn",
    "plotly","altair","networkx",
    # NLP / IR / LLM
    "nltk","spacy","gensim","transformers","sentence-transformers","nlp","natural language processing",
    "text classification","ner","named entity recognition","topic modeling","sentiment analysis",
    "question answering","summarization","retrieval","rag","retrieval augmented generation",
    "vector search","faiss","chroma","chromadb","weaviate","pinecone","qdrant","milvus","pgvector",
    "bm25","tf-idf","tfidf","word2vec","glove","bert","gpt","llm","llms","bleu","rouge",
    "perplexity","embeddings","cosine similarity",
    # MLOps
    "mlops","mlflow","dvc","kubeflow","seldon","bentoml","clearml","wandb","weights & biases",
    "onnx","onnxruntime","tensorrt","numba","cuda",
    # Cloud
    "aws","ec2","s3","iam","lambda","rds","aurora","ecs","eks","ecr","elb","alb","cloudwatch",
    "cloudtrail","athena","glue","redshift","step functions","api gateway","eventbridge",
    "gcp","google cloud","gcs","bigquery","dataproc","dataflow","pubsub","cloud run","cloud functions",
    "composer","vertex ai","spanner","firestore","bigtable",
    "azure","adls","synapse","databricks","cosmos db","aks","azure functions","event hub","blob storage",
    # DevOps / IaC / CI-CD
    "docker","docker compose","kubernetes","k8s","helm","terraform","pulumi","ansible","packer",
    "jenkins","github actions","gitlab ci","circleci","argo cd","argo workflows",
    # Observability
    "prometheus","grafana","loki","tempo","opentelemetry","otel","datadog","new relic","splunk",
    "elk","logstash","kibana","graylog",
    # Build / env / tools
    "maven","gradle","sbt","cmake","make","pip","pipenv","poetry","conda","virtualenv",
    # Testing
    "pytest","unittest","jest","mocha","vitest","cypress","playwright","selenium","locust","k6","postman",
    # Security
    "tls","ssl","mfa","sso","secrets management","hashicorp vault","kms","oidc","saml",
    # Workflow / misc
    "git","github","gitlab","bitbucket","jira","confluence","notion","linux","unix","wsl","vscode",
    # Concepts
    "microservices","event-driven","domain driven design","ddd","clean architecture","cqrs",
    "message queue","data modeling","etl","elt","data pipelines","data warehousing",
    "feature engineering","ab testing","a/b testing","time series","forecasting",
    "asyncio","multiprocessing","multithreading","concurrency","parallelism",
}

# Canonicalization map (normalize aliases to a single key)
CANON = {
    "postgresql": "postgres",
    "rest api": "rest",
    "nodejs": "node", "node.js": "node", "express.js": "express",
    "react.js": "react", "reactjs": "react",
    "nextjs": "next", "next.js": "next",
    "tailwind css": "tailwind",
    "google cloud": "gcp",
    "pgvector": "pgvector",  # keep
}

def canon(s: str) -> str:
    s = s.lower().strip()
    return CANON.get(s, s)

# --- BM25 tokenization & stopwords
STOPWORDS = {
    "the","a","an","and","or","to","of","in","for","with","on","by","at","from",
    "as","is","are","be","this","that","these","those","you","we","they",
    "title","responsibilities","requirements","qualifications","must","preferred","nice","nice to have",
    "ability","will","etc"
}
TOKEN_RE = re.compile(r"[A-Za-z0-9\+\.\-]+")

def tok(s: str):
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOPWORDS]

# ---------- Boundary-aware skill matching ----------
def has_skill(text: str, skill: str) -> bool:
    sk = re.escape(skill.lower())
    # Words like "git" should not match "digital"
    return re.search(rf"(?<![A-Za-z0-9]){sk}(?![A-Za-z0-9])", text) is not None

# ---------- JD skill parsing (no substring mistakes) ----------
def parse_required_skills_from_jd(jd_text: str) -> List[str]:
    t = jd_text.lower()
    found = set()
    for s in SKILL_BANK:
        s_can = canon(s)
        if has_skill(t, s) or (s != s_can and has_skill(t, s_can)):
            found.add(s_can)
    return sorted(found)

# ---------- JD line weighting ----------
def line_weight(line: str) -> float:
    l = line.lower()
    # Requirements / Must-have get more weight
    if any(k in l for k in ["must have", "required", "requirements", "need to", "minimum"]):
        return 1.6
    # Responsibilities / nice-to-have get default or slightly less
    if any(k in l for k in ["responsibilities", "you will", "nice to have", "preferred"]):
        return 1.0
    # Title / headers get less
    if any(k in l for k in ["title:", "role:", "about the role"]):
        return 0.6
    return 1.0

# ---------- Embedding helpers ----------
def embed(texts: List[str]):
    return _model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)

# Re-rank top-k resume bullets for each JD line with a cross-encoder (if enabled)
def rerank_best(resume_bullets: List[str], jd_line: str, sims: np.ndarray, topk: int) -> Tuple[int, float]:
    # sims is the vector of sim scores for this jd_line across all resume bullets
    top_idx = np.argsort(-sims)[:topk]
    pairs = [(resume_bullets[i], jd_line) for i in top_idx]
    scores = _reranker.predict(pairs)  # higher is better
    best_local = int(top_idx[int(np.argmax(scores))])
    return best_local, float(max(scores))

# ---------- Scoring ----------
def jobfit(resume_bullets: List[str], jd_lines: List[str], required_skills: List[str]) -> Dict:
    if not resume_bullets or not jd_lines:
        return {"score": 0.0, "skills_found": [], "skills_missing": required_skills, "matches": []}

    # Embedding sims
    R = embed(resume_bullets)
    J = embed(jd_lines)
    sim = util.cos_sim(R, J).cpu().numpy()  # [len(R), len(J)]

    # For each JD line, get best resume bullet (optionally reranked)
    matches = []
    weighted_scores = []
    weights = []
    for j_idx, jd_line in enumerate(jd_lines):
        w = line_weight(jd_line)
        weights.append(w)

        sims_for_j = sim[:, j_idx]
        if USE_RERANKER and _reranker is not None:
            r_idx, best_score = rerank_best(resume_bullets, jd_line, sims_for_j, TOPK_FOR_RERANK)
            # normalize cross-encoder score roughly into [0,1] via sigmoid-like mapping
            # (cross-encoder outputs are unbounded; this keeps coverage sane)
            norm = 1.0 / (1.0 + np.exp(-best_score))
            best_sim = float(norm)
        else:
            r_idx = int(np.argmax(sims_for_j))
            best_sim = float(sims_for_j[r_idx])

        matches.append({
            "jd_line": jd_line,
            "resume_bullet": resume_bullets[r_idx],
            "similarity": best_sim
        })
        weighted_scores.append(best_sim * w)

    # Weighted semantic coverage
    semantic_cov = float(np.sum(weighted_scores) / max(1e-9, np.sum(weights)))

    # BM25 coverage with cleaned tokens
    bm25 = BM25Okapi([tok(b) for b in resume_bullets])
    bm_scores = np.array([np.mean(bm25.get_scores(tok(l))) for l in jd_lines])
    bm_cov = float((bm_scores / (bm_scores.max() + 1e-9)).mean())

    # Boundary-aware skill detection on resume text
    resume_text = " ".join(resume_bullets).lower()
    required_can = sorted({canon(s) for s in required_skills})
    skills_found = [s for s in required_can if has_skill(resume_text, s)]
    skills_missing = [s for s in required_can if s not in skills_found]
    skill_cov = (len(skills_found) / max(1, len(required_can))) if required_can else 0.0

    # Weighted score → 0..100 (tune weights if you like)
    score = 100.0 * (0.6 * semantic_cov + 0.25 * bm_cov + 0.15 * skill_cov)

    return {
        "score": round(score, 2),
        "skills_found": skills_found,
        "skills_missing": skills_missing,
        "matches": matches[:20],  # preview
    }

# ---------- Tailoring ----------
def tailor_bullets(resume_bullets: list[str], jd_lines: list[str], required_skills: list[str], k: int = 6):
    if not resume_bullets or not jd_lines:
        return []

    R = embed(resume_bullets)
    J = embed(jd_lines)
    sim = util.cos_sim(R, J).cpu().numpy()

    bullet_best = sim.max(axis=1)
    top_idx = np.argsort(-bullet_best)[:k]

    jd_text = " ".join(jd_lines).lower()
    required_can = sorted({canon(s) for s in required_skills})

    suggestions = []
    for i in map(int, top_idx):
        b = resume_bullets[i]
        b_low = b.lower()
        # Add only skills that truly appear in the JD (word boundary) and are missing in this bullet
        adds = [s for s in required_can if not has_skill(b_low, s) and has_skill(jd_text, s)][:3]
        suggested = f"{b} (add: {', '.join(adds)})" if adds else b
        suggestions.append({
            "original": b,
            "suggested": suggested,
            "match_score": float(bullet_best[i])
        })
    return suggestions
