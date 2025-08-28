# --- Stage 1: build frontend ---
FROM node:20-bullseye AS web
WORKDIR /app
COPY frontend ./frontend
RUN cd frontend && npm init -y && npm install react react-dom react-webcam vite && npx vite build --outDir dist \

# --- Stage 2: backend runtime ---
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend

COPY --from=web /app/frontend/dist ./frontend/dist
EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/healthz || exit 1
USER 1000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]