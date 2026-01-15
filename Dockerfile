FROM python:3.11-slim

# 1) Set working directory inside container
WORKDIR /app

# 2) Install system deps (minimal; PyMuPDF usually works without extra)
# If you ever hit PyMuPDF runtime issues, we can add extra libs.
RUN pip install --no-cache-dir --upgrade pip

# 3) Copy requirements first (faster rebuilds)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4) Copy the project code
COPY . /app

# 5) Expose port
EXPOSE 8000

# 6) Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
