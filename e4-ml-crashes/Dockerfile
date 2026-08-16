# Build desde la raiz del repo (necesita el common module como hermano):
#   docker build -f e4-ml-crashes/Dockerfile -t e4-ml .
# Ejecutar (data y outputs como volumenes):
#   docker run \
#     -v ./e4-ml-crashes/data:/app/e4-ml-crashes/data \
#     -v ./e4-ml-crashes/outputs:/app/e4-ml-crashes/outputs \
#     e4-ml python run_models.py
FROM python:3.12-slim

WORKDIR /app/e4-ml-crashes

COPY e4-ml-crashes/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY e4-ml-crashes/*.py ./

CMD ["python", "run_models.py", "--no-wandb"]
