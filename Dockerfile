# 1) Base Python Image
# Start with a small Linux system that already has Python 3.11 installed
FROM python:3.11-slim

# 2) Set working directory inside container
# cd /app
WORKDIR /app

# 3) Install dependencies first (better caching)
# Copy requirements.txt into container
COPY requirements.txt .
# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your application code + model artifacts
COPY app ./app
COPY model ./model

# 5) Expose the port FastAPI will run on
# This app will use port 8000 inside the container.
EXPOSE 8000

# 6) Run the API
CMD ["sh", "-c", "uvicorn app.main_copy1:app --host 0.0.0.0 --port ${PORT:-8000}"]

# 0.0.0.0 means: accept requests from outside container
# otherwise API will not be accessible

# Dockerfile → build → Image → run → Container → API running
