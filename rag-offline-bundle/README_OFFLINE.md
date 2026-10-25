# Offline RAG Bundle

This toolkit packages the RAG application, Python wheels, local embedding model cache, GGUF language model, source documents, and Qdrant database for deployment to a compatible RHEL 8-family Linux host.

## Compatibility requirements

- Linux `x86_64` target for this bundle.
- Python 3.12 (same major/minor as the build host), with pip and the `venv`/`ensurepip` support needed to create an environment.
- glibc at least as new as the build baseline in `metadata/BUILD_INFO.txt` (this bundle was built against glibc 2.28; several wheels are tagged `manylinux_2_28`).
- Compatible native runtime libraries. The offline verification script checks `llama.cpp` shared-library dependencies using `ldd`.
- Enough disk space for the model files, Python environment, database, and documents.

RHEL 8.10 and Rocky Linux 8.10 are similar RHEL-family environments, but verify the actual target's glibc version and native-library behavior rather than relying on the distribution name alone.

## Important RPM safety note

The builder skips RPM collection by default. To opt in, an administrator must explicitly set `RAG_INCLUDE_RPM_BUNDLE=1`; this may collect a broad dependency closure and requires manual review. Any RPMs are retained as optional offline provisioning material only. **`setup_offline.sh` deliberately does not install any RPMs automatically.** Do not pass all bundled RPMs to DNF as a batch: the directory can contain core system packages such as glibc, systemd, rpm, and grub. If the target lacks Python 3.12, pip/venv support, or a native library, identify the exact missing prerequisite and install only approved, compatible RPMs through your organization's system administration process.

## Prepare on the connected development VM

1. Stop the RAG app so Qdrant's local database isn't locked.
2. Unpack this toolkit somewhere under the project, for example `~/rag-poc/offline-tools/`.
3. Create the bundle from the project directory:

   ```bash
   cd ~/rag-poc
   RAG_OFFLINE_BUNDLE_DIR="$HOME/rag-offline-bundle" \
     ./offline-tools/rag_offline_toolkit/prepare_offline_bundle.sh "$HOME/rag-poc"
   ```

4. The builder downloads compatible Python wheels, stages the embedding model cache, and copies the app, documents, model, and Qdrant DB. It skips OS RPM collection by default, records compatibility metadata, and writes `SHA256SUMS`. Do not enable RPM collection unless the resulting packages will be reviewed by an administrator.
5. Confirm the wheelhouse resolves without internet on the development VM:

   ```bash
   source ~/rag-poc/ragvenv/bin/activate
   python -m pip install --dry-run --ignore-installed --no-index \
     --find-links "$HOME/rag-offline-bundle/wheelhouse" \
     --only-binary=:all: \
     --requirement "$HOME/rag-offline-bundle/metadata/requirements-lock.txt"
   ```

## Transfer and install

Transfer the *entire* `rag-offline-bundle` directory using your approved media/process. Treat it as potentially sensitive: it includes source-document text and a vector database derived from it.

On the target, first inspect:

```bash
cat /etc/redhat-release
uname -m
getconf GNU_LIBC_VERSION
python3.12 --version
python3.12 -m pip --version
python3.12 -m venv --help >/dev/null && echo "venv available"
```

Then enter the bundle directory and run:

```bash
chmod +x setup_offline.sh verify_offline.sh
./setup_offline.sh
```

By default, the project installs at `~/rag-poc`. If that path exists, the installer stops rather than overwriting it. Use a new path if appropriate:

```bash
RAG_INSTALL_DIR="$HOME/rag-poc-airgap" ./setup_offline.sh
```

The setup script validates checksums, architecture, RHEL 8 family/release, Python minor version, and the glibc baseline. It builds a new venv on the target and installs Python requirements using `pip --no-index` from the bundled wheelhouse. It does not contact package indexes and does not install RPMs automatically.

If the script stops because a prerequisite is missing, do not install every RPM in `rpms/`. Identify the missing component, obtain the approved matching RPMs through the site's offline process, install only those selected prerequisites, and rerun in a clean install directory.

## Launch and operate

From the installed project directory:

```bash
./run_rag.sh
./run_search.sh "How are new script commands created?"
./refresh_documents_offline.sh
./verify_offline.sh
./verify_offline.sh --llm
```

Use the launch scripts: they set `FASTEMBED_CACHE_PATH` to the bundled cache and enable Hugging Face offline flags. The `--llm` test can take time on a low-core CPU.

Stop the interactive app before refreshing documents because Qdrant local mode locks the database. Editing/adding/removing `.txt` files under `documents/` and rerunning the refresh script rebuilds the vector collection.

## Integrity and acceptance testing

- `SHA256SUMS` validates the transferred files.
- `verify_offline.sh` checks Python dependencies, native shared libraries, local embedding inference, model files, and persisted Qdrant data.
- `verify_offline.sh --llm` additionally loads the local GGUF and performs a short generation test.
- For final acceptance, disconnect/disable the network (according to site policy) and run both verification levels. Do not rely on a test where network connectivity remains available.

## Model/licensing note

The bundle includes `BAAI/bge-small-en-v1.5` through FastEmbed and a quantized GGUF distribution of `Qwen2.5-1.5B-Instruct`. Retain upstream model cards and license notices, and confirm the model weights, packages, and source documents are approved for the intended environment.
