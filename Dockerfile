FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN grep -v -E '^(torch|torchvision)==|^--' requirements.txt > requirements-runtime.txt \
    && pip install --no-cache-dir -r requirements-runtime.txt \
    && pip install --no-cache-dir torch==2.12.0+cpu torchvision==0.27.0+cpu \
       --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY ml ./ml
COPY static ./static
COPY models ./models

RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/rohanchennupati-sudo/BirDiD-bird-identifier/releases/download/v1.1.0/best_model_finetuned.pth', 'models/best_model_finetuned.pth')"

EXPOSE 8000

# --proxy-headers so the rate limiter sees the client IP, not Render's proxy
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]