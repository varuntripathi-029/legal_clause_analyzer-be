#!/bin/bash
set -e

# Start Redis as a local, loopback-only sidecar (no persistence — HF Spaces disk
# is ephemeral anyway, and chat sessions are short-lived/TTL-based by design).
redis-server --daemonize yes --bind 127.0.0.1 --port 6379 --save "" --appendonly no

exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT}" --proxy-headers --forwarded-allow-ips "*"
