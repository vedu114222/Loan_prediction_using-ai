# Loan Prediction API

> ML-powered loan approval · PostgreSQL + pgvector · RAG over underwriting policy · MCP server for LLM agents

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML Model | Scikit-learn Logistic Regression |
| Database | PostgreSQL 15 + pgvector |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| RAG | pgvector cosine similarity search |
| Agent interface | Model Context Protocol (`mcp[cli]`) |

---

## Project layout

```
loan-prediction-api/
├── app/
│   ├── main.py                    # FastAPI entrypoint, startup DB init
│   ├── routers/prediction.py      # POST /api/predict-loan, GET /api/predictions
│   ├── services/prediction_service.py
│   ├── models/request_models.py   # Pydantic LoanRequest
│   ├── ml/
│   │   ├── train_model.py         # trains + saves loan_model.pkl
│   │   └── loan_model.pkl         # generated — not committed to git
│   ├── rag/
│   │   ├── ingest.py              # chunk + embed policy docs → pgvector
│   │   └── retriever.py           # semantic search helper
│   ├── mcp/server.py              # MCP server (4 tools)
│   └── utils/db.py                # SQLAlchemy engine, pgvector setup
├── policy_docs/                   # underwriting policy markdown files
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── .env
```

---

## Quick start

### 1. Prerequisites

- Python 3.11+
- Docker Desktop (running)

### 2. Clone & install

```bash
git clone <repo-url> loan-prediction-api
cd loan-prediction-api

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Postgres with pgvector

```bash
docker compose up -d db
```

Verify it's up:

```bash
docker compose ps
```

### 4. Train the ML model

```bash
python app/ml/train_model.py
```

This saves `app/ml/loan_model.pkl`.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** — you'll see the interactive Swagger UI with
`POST /api/predict-loan`, `GET /api/predictions`, and `GET /api/feature-importance`.

### 6. Ingest policy documents

With the API running (so the `policy_chunks` table exists):

```bash
python -m app.rag.ingest
```

To wipe and re-ingest:

```bash
python -m app.rag.ingest --reset
```

### 7. Test the retriever

```bash
python -c "
from app.rag.retriever import search_policy_chunks
import json
results = search_policy_chunks('minimum credit score for approval')
print(json.dumps(results, indent=2))
"
```

### 8. Run the MCP server

```bash
python -m app.mcp.server
```

---

## API reference

### `POST /api/predict-loan`

```json
{
  "age": 34,
  "salary": 95000,
  "credit_score": 720,
  "loan_amount": 250000
}
```

Response:

```json
{
  "loan_status": "Approved",
  "probability": 0.8731,
  "approved": true
}
```

### `GET /api/predictions?limit=10`

Returns the last N stored predictions.

### `GET /api/feature-importance`

```json
{
  "age": 0.012345,
  "salary": 0.000087,
  "credit_score": 0.023456,
  "loan_amount": -0.000043
}
```

---

## Claude Desktop integration

Edit your Claude Desktop config (`Settings → Developer → Edit Config`):

```json
{
  "mcpServers": {
    "loan-prediction": {
      "command": "C:/path/to/loan-prediction-api/venv/Scripts/python.exe",
      "args": ["-m", "app.mcp.server"],
      "cwd": "C:/path/to/loan-prediction-api"
    }
  }
}
```

> **macOS/Linux**: use `venv/bin/python` instead of `venv/Scripts/python.exe`.

Restart Claude Desktop. You'll see four tools available:

| Tool | Description |
|---|---|
| `predict_loan` | Run ML inference on applicant data |
| `search_policy` | Semantic search over policy docs |
| `get_history` | Fetch recent predictions from PostgreSQL |
| `feature_importance` | Per-feature model coefficients |

### Example prompts

- *"Predict approval for age 28, salary $85k, credit score 760, loan $300k — and explain using our policy."*
- *"Why might someone with a 580 credit score be rejected? Check the policy docs."*
- *"Show the last 5 predictions and summarize any pattern."*
- *"Which factor does the model weight most heavily?"*

---

## Docker (full stack)

To run both the DB and the API in containers:

```bash
# First build & train the model inside the container image
docker compose up --build
```

The API will be available at `http://localhost:8000`.  
Then ingest docs into the containerized DB:

```bash
docker compose exec api python -m app.rag.ingest
```

---

## Customising

| Change | Where |
|---|---|
| Swap to real loan data | `app/ml/train_model.py` |
| Use OpenAI embeddings | `app/rag/ingest.py` + `app/rag/retriever.py` |
| Add new policy docs | Drop `.md` / `.txt` into `policy_docs/`, re-run ingest |
| Add SHAP explanations | `app/services/prediction_service.py` |
| Secure the MCP server | See `mcp` SDK OAuth docs |

---

## License

MIT
