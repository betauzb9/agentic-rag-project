FROM python:3.11-slim

WORKDIR /app

# Sistem kutubxonalari (PyMuPDF uchun)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# PDF faylingizni shu papkaga qo'yib, nomini document.pdf deb qoldiring
# yoki app.py dagi PDF_PATH env o'zgaruvchisini o'zgartiring
COPY document.pdf .

EXPOSE 7860

CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "7860"]
