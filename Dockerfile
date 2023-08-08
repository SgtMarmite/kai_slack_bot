FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=3000

EXPOSE $PORT

# Run the application
CMD uvicorn app:api --host=0.0.0.0 --port=$PORT
