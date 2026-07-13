FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

ARG INSTALL_OCR=false
COPY requirements-app.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt \
    && if [ "$INSTALL_OCR" = "true" ]; then pip install --no-cache-dir -r requirements-ocr.txt; fi

COPY src ./src
COPY evaluation ./evaluation
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY .env.example ./.env.example 
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "electrical_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
