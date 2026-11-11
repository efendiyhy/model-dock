FROM python:3.12-slim
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY custom_modules/ ./custom_modules/
COPY examples/iris_classifier.joblib ./examples/iris_classifier.joblib
RUN mkdir -p /workspace/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
