# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies (without project)
RUN pip install uv

RUN uv pip install --system -r pyproject.toml

# Copy the rest of the code
COPY . .

# Install the project in editable mode
RUN uv pip install -e .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]