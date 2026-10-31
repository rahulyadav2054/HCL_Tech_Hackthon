FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai ai
COPY backend backend
COPY frontend frontend
COPY data data
# Build the index (downloads the embedding + reranker models) and train the classifiers at image build time
RUN python -m ai.build_index && python -m ai.classify --train
EXPOSE 8000 8501
# backend (FastAPI) and frontend (Streamlit) in one container for the demo; pass GROK_API_KEY with --env-file .env
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
