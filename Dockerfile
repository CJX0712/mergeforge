FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core ./core
COPY data ./data
COPY models ./models
COPY finetune ./finetune
COPY merging ./merging
COPY eval ./eval
COPY pipeline ./pipeline
COPY tests ./tests
COPY cli.py ./
COPY examples ./examples

RUN pip install --no-cache-dir -e ".[dev]"

# 完整基准 + 门禁（CPU，~50s）
CMD ["python", "cli.py", "benchmark"]
