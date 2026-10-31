# pdf-mcp as an OCI image: the server plus system Tesseract (English).
# Build:  podman build -t pdf-mcp .
# Run:    podman run -i --rm --userns=keep-id -v "$HOME/docs:$HOME/docs:Z" pdf-mcp
# Tool paths are host paths, so mount each directory at the same path.
FROM docker.io/library/python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/nfsarch33/pdf-mcp-server"

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY pdf_mcp ./pdf_mcp
RUN pip install --no-cache-dir ".[ocr]" && rm -rf /src

RUN useradd --create-home --uid 1000 pdfmcp
USER pdfmcp
WORKDIR /home/pdfmcp

ENTRYPOINT ["pdf-mcp"]
CMD ["serve"]
