FROM python:3.11-slim

WORKDIR /app

COPY src/ ./src/
COPY api/ ./api/
COPY pyproject.toml .
COPY requirements.txt .

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["flask", "--app", "api.main:create_app", "run", \
     "--host", "0.0.0.0", "--port", "5000", "--debug"]