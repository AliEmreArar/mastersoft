FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/ekap-anom /usr/local/bin/ekap-anom
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY configs/ configs/
COPY src/ src/

EXPOSE 8000

ENV EKAP_CONFIG=configs/prod.yaml

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"

ENTRYPOINT ["ekap-anom"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
