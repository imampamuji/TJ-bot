# Gunakan base image Python
FROM python:3.12-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy semua file project ke container
COPY . /app

# Install dependencies dari pyproject.toml menggunakan uv
RUN uv sync --frozen

# Expose port untuk Hugging Face
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Jalankan main.py dengan uv
CMD ["uv", "run", "python", "main.py"]
