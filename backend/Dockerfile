FROM python:3.11

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* ./

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi \
    && pip install python-multipart
# Creating folders, and files for a project:
COPY . .

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]