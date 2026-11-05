# Equilibra: Synthetic Data Pipeline for Imbalanced Data

Agentic resampling, RAG-guided hyperparameters and leakage-free evaluation for severely imbalanced datasets (tabular and image).

**Live site:** https://nikhil-creat.github.io/synthetic-data-pipeline/

## Leakage rules (enforced in code, covered by tests)
1. Stratified train/validation/test split happens first.
2. Imputer and scaler are fitted on training rows only.
3. SMOTE, ADASYN, Borderline-SMOTE and SMOTE-Tomek touch the training split only.
4. The ResNet backbone is frozen and pretrained, so image features can be extracted before the split.
5. The threshold is tuned on validation data by default. `threshold_source=test` is allowed but flagged `optimistic`.
6. Accuracy is never computed. Metrics: precision, recall, F1, PR-AUC (plus the random-model baseline).

## Quick start
```bash
cp .env.example .env
docker compose up --build        # API on :8000, Qdrant on :6333
open http://localhost:8000/docs
```
Without Docker: `make install && make run`. Run tests with `make test`.

## API
| Endpoint | Purpose |
|---|---|
| `POST /ingest` | CSV (tabular) or ZIP of `class_name/image.jpg` folders |
| `POST /resample` | Agent profiles data, queries Qdrant, routes, resamples (self-healing) |
| `POST /compare` | Benchmark every resampler on validation data |
| `POST /train` | Fit `logreg` or `rf` on the resampled training rows |
| `POST /evaluate` | PR-AUC, precision, recall, F1 on the untouched test set |
| `GET /report/{id}` | Markdown model card |
| `GET /demo-data` | Download a synthetic imbalanced CSV |
| `POST /assistant/chat` | AI assistant grounded in the Qdrant knowledge base and optional dataset results |
| `POST /assistant/explain` | Plain-English explanation of an evaluated dataset |
| `GET /assistant/status` | Detected providers and the auto-selected model (no keys returned) |
| `GET /metrics` | Prometheus counters and latency sums |
| `GET /strategies`, `/datasets`, `/health` | Introspection |

Set `API_KEY` to require an `X-API-Key` header and `CORS_ORIGINS` to restrict browsers.

## AI assistant (free Groq and Gemini keys)
Put a key in `.env` as `GROQ_API_KEY` or `GEMINI_API_KEY`. Either variable accepts either kind of key.
- **Provider detection:** Groq keys start with `gsk_`, Gemini keys with `AIza`.
- **Model detection:** the router lists the models your key can use, drops speech, embedding, image and guard models, ranks the rest, and caches the choice for an hour.
- **Self-healing:** if a call hits a rate limit or a retired model, it tries the next model, then the other provider.
- **No key?** The assistant still answers from the Qdrant knowledge base (`provider: offline-rag`).
- **On the website:** the chat widget accepts a key in the browser (stored in localStorage, never committed). Keys never go in the repo.

## Routing rules
Minority under 2%: Borderline-SMOTE. High class overlap: SMOTE-Tomek. 2 to 10% with low noise and at most 50 features: ADASYN. Otherwise SMOTE. Failures retry with a smaller k, then SMOTE, then random oversampling.

## Layout
```
pipeline.py  assistant.py  main.py  Dockerfile  docker-compose.yml  requirements*.txt
tests/       sample_data/   docs/ (GitHub Pages site)   .github/workflows/
```

Note: the Qdrant knowledge base ships with seed guidance and sample policies. Replace them with your own papers and compliance documents in `KNOWLEDGE_BASE`.

MIT licensed. Built by Nikhil Chary Sriramoju.

## GitHub description and topics
**Description:** Equilibra: agentic synthetic-data pipeline for severely imbalanced datasets. SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, ResNet features, Qdrant RAG, a Groq + Gemini AI assistant, FastAPI + Docker, and honest PR-AUC evaluation.

**Topics:** machine-learning, imbalanced-data, smote, imbalanced-learn, synthetic-data, pytorch, resnet, fastapi, qdrant, rag, agentic-ai, llm, ai-assistant, groq, gemini, docker, mlops, data-science, precision-recall, github-pages

Set them from Termux with `bash scripts/set_github_meta.sh` (needs `gh auth login`).
