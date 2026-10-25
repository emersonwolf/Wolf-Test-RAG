#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT/ragvenv/bin/python"
RUN_LLM=0
if [[ "${1:-}" == "--llm" ]]; then RUN_LLM=1; fi

[[ -x "$VENV_PY" ]] || { echo "ERROR: ragvenv not found at $ROOT/ragvenv" >&2; exit 1; }
source "$ROOT/ragvenv/bin/activate"
export FASTEMBED_CACHE_PATH="$ROOT/models/fastembed_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$ROOT"

echo "=== Python dependency consistency ==="
"$VENV_PY" -m pip check

echo
echo "=== Native llama.cpp library dependencies ==="
if command -v ldd >/dev/null 2>&1; then
  FAILED=0
  while IFS= read -r -d '' shared_object; do
    missing="$(ldd "$shared_object" 2>&1 | grep 'not found' || true)"
    if [[ -n "$missing" ]]; then
      echo "Unresolved shared library for $shared_object:"
      echo "$missing"
      FAILED=1
    fi
  done < <(find "$ROOT/ragvenv" -path '*site-packages/llama_cpp*' -type f \( -name '*.so' -o -name '*.so.*' \) -print0)
  [[ "$FAILED" -eq 0 ]] || { echo "ERROR: one or more native libraries are missing." >&2; exit 1; }
  echo "llama.cpp shared-library dependency check: PASS"
else
  echo "WARNING: ldd is unavailable; skipped shared-library check."
fi

echo
echo "=== Python imports and model files ==="
"$VENV_PY" - <<'PY'
from pathlib import Path
import fastembed
import qdrant_client
import llama_cpp

root = Path.cwd()
model = root / "models" / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
cache = root / "models" / "fastembed_cache"
assert model.is_file() and model.stat().st_size > 500_000_000, f"Missing/invalid LLM model: {model}"
assert cache.is_dir() and any(cache.rglob("*")), f"FastEmbed cache is empty: {cache}"
print("fastembed:", getattr(fastembed, "__version__", "installed"))
print("qdrant-client:", getattr(qdrant_client, "__version__", "installed"))
print("llama-cpp-python:", getattr(llama_cpp, "__version__", "installed"))
print("Local LLM file: PASS")
print("Local FastEmbed cache exists: PASS")
PY

echo
echo "=== Offline embedding test ==="
"$VENV_PY" - <<'PY'
from fastembed import TextEmbedding
vectors = list(TextEmbedding(model_name="BAAI/bge-small-en-v1.5").embed([
    "Verify local embeddings work without downloading model files."
]))
assert len(vectors) == 1
assert len(vectors[0]) == 384
print(f"FastEmbed local-only test: PASS ({len(vectors[0])} dimensions)")
PY

echo
echo "=== Persistent Qdrant database test ==="
"$VENV_PY" - <<'PY'
from qdrant_client import QdrantClient
client = QdrantClient(path="rag_db")
try:
    assert client.collection_exists("text_chunks"), "text_chunks collection missing"
    count = client.count(collection_name="text_chunks", exact=True).count
    assert count > 0, "text_chunks has no vectors"
    print(f"Qdrant local database test: PASS ({count} vectors)")
finally:
    client.close()
PY

if [[ "$RUN_LLM" -eq 1 ]]; then
  echo
  [[ -f "$ROOT/llm_smoke_test.py" ]] || { echo "ERROR: llm_smoke_test.py is missing." >&2; exit 1; }
  echo "=== Local LLM generation test ==="
  "$VENV_PY" "$ROOT/llm_smoke_test.py"
fi

echo
echo "OFFLINE VERIFICATION PASSED"
