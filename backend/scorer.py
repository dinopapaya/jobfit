from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
import re

# small, fast model
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== BIG SKILL BANK =====
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

STOPWORDS = {
    "the","a","an","and","or","to","of","in","for","with","on","by","at","from",
    "as","is","are","be","this","that","these","those","you","we","they",
    "responsibilities","requirements","qualifications","must","ability","will","etc"
}
TOKEN_RE = re.compile(r"[A-Za-z0-9\+\.\-]+")

def tok(s: str):
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOPWORDS]

def parse_required_skills_from_jd(jd_text: str) -> List[str]:
    t = jd_text.lower()
    found = []
    for s in SKILL_BANK:
        if s in t:
            found.append(s)
    caps = set(re.findall(r"\b([A-Za-z][A-Za-z0-9+\.\-]{2,})\b", jd_text))
    for c in caps:
        low = c.lower()
        if low in SKILL_BANK and low not in found:
            found.append(low)
    return sorted(found)

def embed(texts: List[str]):
    return _model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)

def jobfit(resume_bullets: List[str], jd_lines: List[str], required_skills: List[str]) -> Dict:
    if not resume_bullets or not jd_lines:
        return {"score": 0.0, "skills_found": [], "skills_missing": required_skills, "matches": []}

    R = embed(resume_bullets)
    J = embed(jd_lines)
    sim = util.cos_sim(R, J).cpu().numpy()

    jd_best = sim.max(axis=0)
    semantic_cov = float(jd_best.mean())

    bm25 = BM25Okapi([tok(b) for b in resume_bullets])
    bm_scores = np.array([np.mean(bm25.get_scores(tok(l))) for l in jd_lines])
    bm_cov = float((bm_scores / (bm_scores.max() + 1e-9)).mean())

    def has_skill(text: str, skill: str) -> bool:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\.\+\-]*", skill):
            return re.search(rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9])", text) is not None
        return skill in text  # e.g., "c++"

    resume_text = " ".join(resume_bullets).lower()
    skills_found = [s for s in required_skills if has_skill(resume_text, s)]
    skills_missing = [s for s in required_skills if s not in skills_found]
    skill_cov = (len(skills_found) / max(1, len(required_skills))) if required_skills else 0.0

    score = 100.0 * (0.5 * semantic_cov + 0.3 * bm_cov + 0.2 * skill_cov)

    matches = []
    for j_idx, jd_line in enumerate(jd_lines):
        r_idx = int(sim[:, j_idx].argmax())
        matches.append({
            "jd_line": jd_line,
            "resume_bullet": resume_bullets[r_idx],
            "similarity": float(sim[r_idx, j_idx])
        })

    return {
        "score": round(score, 2),
        "skills_found": skills_found,
        "skills_missing": skills_missing,
        "matches": matches[:20],
    }

def tailor_bullets(resume_bullets: list[str], jd_lines: list[str], required_skills: list[str], k: int = 6):
    if not resume_bullets or not jd_lines:
        return []
    R = embed(resume_bullets)
    J = embed(jd_lines)
    sim = util.cos_sim(R, J).cpu().numpy()
    bullet_best = sim.max(axis=1)
    top_idx = np.argsort(-bullet_best)[:k]
    jd_text = " ".join(jd_lines).lower()
    suggestions = []
    for i in map(int, top_idx):
        b = resume_bullets[i]
        b_low = b.lower()
        adds = [s for s in required_skills if s not in b_low and s in jd_text][:3]
        suggested = f"{b} (add: {', '.join(adds)})" if adds else b
        suggestions.append({"original": b, "suggested": suggested, "match_score": float(bullet_best[i])})
    return suggestions
