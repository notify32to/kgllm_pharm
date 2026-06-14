FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/kgllm_pharm

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY diagrams ./diagrams
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["kgllm-pharm"]
CMD ["--help"]
