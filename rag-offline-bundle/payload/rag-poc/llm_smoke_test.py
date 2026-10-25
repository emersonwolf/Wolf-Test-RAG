from pathlib import Path
import os

from llama_cpp import Llama


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)

if not MODEL_PATH.is_file():
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

if MODEL_PATH.stat().st_size < 500_000_000:
    raise SystemExit(
        "Model file is unexpectedly small. Check the download."
    )

threads = max(1, min(os.cpu_count() or 2, 4))

print(f"Model file: {MODEL_PATH}")
print(f"CPU threads: {threads}")
print("Loading model. This may take a little while...")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=threads,
    n_gpu_layers=0,
    verbose=False,
)

print("Model loaded successfully.")
print("Generating a response...\n")

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful technical assistant. "
                "Answer concisely and clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                "In two or three sentences, explain what a vector "
                "database does in a retrieval-augmented generation "
                "(RAG) system."
            ),
        },
    ],
    temperature=0.2,
    max_tokens=128,
)

answer = response["choices"][0]["message"]["content"]

if not answer or not answer.strip():
    raise RuntimeError("The model returned an empty response.")

print("=== MODEL RESPONSE ===")
print(answer.strip())
print("\nLOCAL LLM TEST PASSED")
