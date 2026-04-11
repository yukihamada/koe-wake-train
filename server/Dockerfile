# syntax=docker/dockerfile:1.6
FROM rust:1.88-bookworm AS chef
RUN cargo install cargo-chef --locked --version 0.1.71
WORKDIR /app

FROM chef AS planner
COPY server/Cargo.toml server/Cargo.lock* ./
COPY server/src ./src
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY server/Cargo.toml server/Cargo.lock* ./
COPY server/src ./src
RUN cargo build --release --bin koe-wake-train

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/koe-wake-train /usr/local/bin/koe-wake-train
EXPOSE 8080
ENV RUST_LOG=info
CMD ["koe-wake-train"]
