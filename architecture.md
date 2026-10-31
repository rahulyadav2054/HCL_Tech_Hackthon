# Architecture

IT Ticket Resolution Assistant: analyzes one ticket and recommends a fix. It never closes tickets or changes systems.

```
frontend/ (Streamlit) --HTTP + JWT--> backend/ (FastAPI) --> ai/ (analysis pipeline)
                                          |                     |
                                          v                     v
                                   data/app.sqlite       index/, models/, LLM API
                                   (users, audit,        (LLM gets masked text only)
                                    feedback)
```

## Pipeline

1. Preprocess: normalize, mask PII, extract entities.
2. Classify: calibrated scikit-learn models for category, subcategory and priority. The LLM gives a second opinion on close calls only.
3. Retrieve: role filter, then BM25 + FAISS, Reciprocal Rank Fusion, boosts and cross-encoder rerank.
4. Generate: 1-3 resolution paths from the evidence pack only, using the LLM or an offline extractive fallback.
5. Validate: every citation must exist in the pack and its excerpt must match the source.
6. Score: High / Medium / Low / Escalate. Low and Escalate route to a human.
7. Audit: the analysis is written to SQLite before the response is returned.
