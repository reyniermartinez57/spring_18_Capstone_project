# Use an official lightweight Ubuntu base
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.14 via Deadsnakes PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.14 \
    python3.14-venv \
    python3.14-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file first to leverage Docker caching layers
COPY requirements.txt .

# Create a virtual environment and install pinned dependencies + core application packages
RUN python3.14 -m venv /v_env
ENV PATH="/v_env/bin:$PATH"
RUN /v_env/bin/pip install --no-cache-dir --upgrade pip && \
    /v_env/bin/pip install --no-cache-dir -r requirements.txt uvicorn requests fastapi
# Copy the application code and artifacts
COPY src/ ./src/
COPY configs/ ./configs/
COPY mlflow.db .
COPY mlruns/ ./mlruns/
COPY tests/ ./tests/

# Expose the port FastAPI runs on
EXPOSE 8000

# Launch uvicorn as a python module from the virtual environment python binary
CMD ["/v_env/bin/python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]