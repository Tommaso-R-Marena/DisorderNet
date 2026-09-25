# The library and CLI. No GPU, no model weights, no network at run time.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY disordernet ./disordernet
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.12-slim
LABEL org.opencontainers.image.title="disordernet" \
      org.opencontainers.image.description="Benchmark capacity and the pairwise scoring protocol" \
      org.opencontainers.image.source="https://github.com/Tommaso-R-Marena/DisorderNet" \
      org.opencontainers.image.licenses="MIT"
RUN useradd -m -u 1000 app
COPY --from=build /src/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
USER app
WORKDIR /work
ENTRYPOINT ["disordernet"]
CMD ["table"]
