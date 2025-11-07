# Dockerfile (CLI tools)
FROM python:3.10-slim

# System deps for numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libatlas-base-dev libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY src /app/src
COPY data /app/data
COPY .env /app/.env  || true

# Python deps
RUN pip install --no-cache-dir numpy

# Optional (some tools use matplotlib only in simulator; skip here)
# RUN pip install --no-cache-dir matplotlib

ENV PYTHONPATH=/app/src

# Default command prints help; you can override with `-m radarlink.tools.control_hub` etc
CMD ["python3", "-c", "print('radarlink CLI container ready. Use -m radarlink.tools.*')"]
