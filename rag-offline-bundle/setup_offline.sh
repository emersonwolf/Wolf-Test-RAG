#!/usr/bin/env bash
set -Eeuo pipefail

# Safe installer for a prepared offline RAG bundle.
# Deliberately does NOT install bundled system RPMs automatically.
# Optional: RAG_INSTALL_DIR="$HOME/rag-poc-airgap" ./setup_offline.sh

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${RAG_INSTALL_DIR:-$HOME/rag-poc}"

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$BUNDLE_DIR/SHA256SUMS" ]] || fail "SHA256SUMS not found; this is not a completed bundle."
[[ -f "$BUNDLE_DIR/metadata/BUILD_INFO.txt" ]] || fail "Build metadata missing."
[[ -d "$BUNDLE_DIR/wheelhouse" ]] || fail "Python wheelhouse missing."
[[ -d "$BUNDLE_DIR/payload/rag-poc" ]] || fail "Project payload missing."

get_info() {
  awk -F= -v key="$1" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' \
    "$BUNDLE_DIR/metadata/BUILD_INFO.txt"
}

echo "=== Verifying transferred files ==="
(
  cd "$BUNDLE_DIR"
  sha256sum -c SHA256SUMS
) || fail "Checksum validation failed. Do not install a corrupted or incomplete bundle."

EXPECTED_ARCH="$(get_info source_arch)"
EXPECTED_PY_MINOR="$(get_info source_python_minor)"
EXPECTED_GLIBC="$(get_info source_glibc | awk '{print $2}')"
CURRENT_ARCH="$(uname -m)"

[[ "$CURRENT_ARCH" == "$EXPECTED_ARCH" ]] || fail \
  "Architecture mismatch: bundle=$EXPECTED_ARCH, target=$CURRENT_ARCH. Build a bundle for the target architecture."

if [[ -r /etc/redhat-release ]]; then
  if ! grep -Eq 'release 8([.[:space:]]|$)' /etc/redhat-release; then
    fail "This bundle was prepared for RHEL 8-compatible systems; target reports: $(cat /etc/redhat-release)"
  fi
else
  fail "/etc/redhat-release is missing; manually verify this is an approved RHEL 8-compatible target."
fi

if [[ -z "$EXPECTED_GLIBC" || "$EXPECTED_GLIBC" == "unknown" ]]; then
  echo "WARNING: Build metadata does not specify a glibc version; native wheel compatibility must be verified manually."
else
  TARGET_GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
  [[ -n "$TARGET_GLIBC" ]] || fail "Could not determine target glibc version."
  # Wheels tagged manylinux_2_28 require glibc 2.28 or newer.
  OLDEST="$(printf '%s\n%s\n' "$EXPECTED_GLIBC" "$TARGET_GLIBC" | sort -V | head -n1)"
  [[ "$OLDEST" == "$EXPECTED_GLIBC" ]] || fail \
    "Target glibc $TARGET_GLIBC is older than bundle baseline $EXPECTED_GLIBC. Do not install this wheelhouse on this target."
  echo "glibc compatibility baseline: bundle=$EXPECTED_GLIBC, target=$TARGET_GLIBC"
fi

command -v python3.12 >/dev/null 2>&1 || fail \
  "Python 3.12 is not installed. This installer intentionally will not install every RPM in the bundle. Provision the approved Python 3.12, pip/ensurepip, venv, and native runtime prerequisites using your organization's offline RPM process, then rerun setup."

CURRENT_PY_MINOR="$(python3.12 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$CURRENT_PY_MINOR" == "$EXPECTED_PY_MINOR" ]] || fail \
  "Python mismatch: bundle=$EXPECTED_PY_MINOR, target=$CURRENT_PY_MINOR. Prepare the bundle using the same Python minor version."

python3.12 -m pip --version >/dev/null 2>&1 || fail \
  "Python 3.12 pip is unavailable. Provision the matching pip package through the approved offline process."
python3.12 -m venv --help >/dev/null 2>&1 || fail \
  "Python 3.12 venv module is unavailable. Provision the matching venv/ensurepip components through the approved offline process."

[[ ! -e "$INSTALL_DIR" ]] || fail \
  "Install directory already exists: $INSTALL_DIR. Choose a new directory, e.g. RAG_INSTALL_DIR=\"\$HOME/rag-poc-airgap\" ./setup_offline.sh. Existing files were not modified."

mkdir -p "$(dirname "$INSTALL_DIR")"
cp -a "$BUNDLE_DIR/payload/rag-poc" "$INSTALL_DIR"

# Create a new venv on the target; never copy a venv across hosts.
echo
echo "=== Creating isolated Python environment ==="
python3.12 -m venv "$INSTALL_DIR/ragvenv" || fail \
  "Could not create the target venv. Ensure Python 3.12 venv/ensurepip packages are installed; no RPMs were installed automatically."

VENV_PY="$INSTALL_DIR/ragvenv/bin/python"
"$VENV_PY" -m pip --version >/dev/null 2>&1 || fail \
  "The new virtual environment has no pip. Provision matching Python 3.12 pip/ensurepip components and retry in a clean install directory."

echo
echo "=== Installing pinned Python packages from local wheelhouse (no index) ==="
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 "$VENV_PY" -m pip install \
  --no-index \
  --find-links "$BUNDLE_DIR/wheelhouse" \
  --requirement "$BUNDLE_DIR/metadata/requirements-lock.txt"
"$VENV_PY" -m pip check

# Launchers enforce the project-local embedding cache and prevent Hub requests.
cat > "$INSTALL_DIR/run_rag.sh" <<'RUNRAG'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/ragvenv/bin/activate"
export FASTEMBED_CACHE_PATH="$ROOT/models/fastembed_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$ROOT"
exec "$ROOT/ragvenv/bin/python" "$ROOT/rag_app.py" "$@"
RUNRAG

cat > "$INSTALL_DIR/run_search.sh" <<'RUNSEARCH'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/ragvenv/bin/activate"
export FASTEMBED_CACHE_PATH="$ROOT/models/fastembed_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$ROOT"
exec "$ROOT/ragvenv/bin/python" "$ROOT/search_documents.py" "$@"
RUNSEARCH

cat > "$INSTALL_DIR/refresh_documents_offline.sh" <<'RUNREFRESH'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/ragvenv/bin/activate"
export FASTEMBED_CACHE_PATH="$ROOT/models/fastembed_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$ROOT"
exec "$ROOT/refresh_documents.sh" "$@"
RUNREFRESH

chmod +x "$INSTALL_DIR/run_rag.sh" "$INSTALL_DIR/run_search.sh" "$INSTALL_DIR/refresh_documents_offline.sh"
cp -p "$BUNDLE_DIR/README_OFFLINE.md" "$INSTALL_DIR/AIRGAP_README.md"
cp -p "$BUNDLE_DIR/verify_offline.sh" "$INSTALL_DIR/verify_offline.sh"
chmod +x "$INSTALL_DIR/verify_offline.sh"

export FASTEMBED_CACHE_PATH="$INSTALL_DIR/models/fastembed_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo
echo "=== Checking bundled vector database ==="
if ! "$VENV_PY" - "$INSTALL_DIR/rag_db" <<'PY'
import sys
from qdrant_client import QdrantClient
client = QdrantClient(path=sys.argv[1])
try:
    if not client.collection_exists("text_chunks"):
        raise RuntimeError("text_chunks collection missing")
    count = client.count(collection_name="text_chunks", exact=True).count
    if count < 1:
        raise RuntimeError("text_chunks collection is empty")
    print(f"Existing vector database is valid ({count} vectors).")
finally:
    client.close()
PY
then
  echo "The copied database is missing or unusable; attempting an offline rebuild from bundled documents."
  (cd "$INSTALL_DIR" && "$VENV_PY" "$INSTALL_DIR/ingest_documents.py") || fail \
    "Bundled database validation and offline rebuild both failed. Inspect the installation directory before retrying."
fi

echo
echo "=== Running offline verification ==="
"$INSTALL_DIR/verify_offline.sh"

echo
echo "=== OFFLINE RAG SETUP SUCCESSFUL ==="
echo "Install location: $INSTALL_DIR"
echo "Start interactive RAG:  $INSTALL_DIR/run_rag.sh"
echo "Search documents:        $INSTALL_DIR/run_search.sh \"your query\""
echo "Refresh documents:       $INSTALL_DIR/refresh_documents_offline.sh"
echo "Verify full LLM runtime: $INSTALL_DIR/verify_offline.sh --llm"
