# Python 3.11 to match Grigorov book (host may be 3.12, Docker pins 3.11)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Pin CUDA runtime label (CPU-first; GPU optional via --gpus all + matching torch extra)
LABEL cuda="12.1" python="3.11" benchmark="universe-0.1.0"
WORKDIR /work
# Non-root user for security (best practice 2026: never run bench as root)
RUN useradd -m -u 10001 bench && mkdir -p /work/results && chown -R bench:bench /work
COPY requirements.txt /work/requirements.txt
RUN pip install --upgrade pip && pip install -r /work/requirements.txt
COPY pyproject.toml README.md /work/
COPY src/ /work/src/
COPY benchmarks/ /work/benchmarks/
COPY scripts/ /work/scripts/
COPY tests/ /work/tests/
COPY docs/ /work/docs/
USER bench
# Deterministic defaults
ENV PYTHONHASHSEED=0 UB_SEED=1337 UB_FROZEN_PROMPTS=1
CMD ["pytest", "-q"]
