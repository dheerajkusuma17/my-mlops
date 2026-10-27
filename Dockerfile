FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py .
COPY config.yml .
COPY models/model.pkl models/model.pkl

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]