FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps (optional: git for model downloads if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY src/data_factory/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (mounted at runtime; copy ensures scripts exist if mount missing)
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
