FROM python:3.10-slim

# GDAL/rasterio system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 300 --workers 1 web_app:app
