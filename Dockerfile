# Trading app image: builds the React cockpit, then serves it + the API from FastAPI.
# Run:  docker build -t equity-agent .  &&  docker run -p 8000:8000 -v %cd%/data:/app/data equity-agent
# Then open http://localhost:8000  (ingest data into the mounted ./data volume first).

# --- Stage 1: build the frontend ---
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: python app ---
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"
COPY config.yaml ./
COPY --from=frontend /app/frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["uvicorn", "equity_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
