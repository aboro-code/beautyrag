FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch build first: the default PyPI wheel pulls in CUDA libraries that
# aren't needed here and push memory well past free-tier hosting limits (512MB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
