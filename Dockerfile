FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server_optimized.py .

EXPOSE 8765

CMD ["python", "server_optimized.py"]
