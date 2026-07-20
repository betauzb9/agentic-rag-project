FROM python:3.11-slim

WORKDIR /app

# Sistem kutubxonalari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Agar repository'da document.pdf bo'lsa nusxalaydi, bo'lmasa xato bermay o'tib ketadi
COPY document.pd[f] .

EXPOSE 7860

CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "7860"]
