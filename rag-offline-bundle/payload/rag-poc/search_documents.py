from pathlib import Path
import sys

from fastembed import TextEmbedding
from qdrant_client import QdrantClient


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PROJECT_DIR / "rag_db"

COLLECTION_NAME = "text_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5
PREVIEW_LENGTH = 600


def main() -> None:
    if not DATABASE_DIR.exists():
        raise SystemExit(
            f"Database directory not found: {DATABASE_DIR}"
        )

    # Accept a question as a command-line argument or prompt for one.
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Enter your search question: ").strip()

    if not query:
        raise SystemExit("No search question provided.")

    print(f"\nQuestion: {query}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = TextEmbedding(model_name=MODEL_NAME)

    # The ingestion script used the 'passage:' prefix.
    # Use the corresponding 'query:' prefix for this initial test.
    query_vector = list(
        model.embed([f"query: {query}"])
    )[0]

    client = QdrantClient(path=str(DATABASE_DIR))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise SystemExit(
                f"Collection not found: {COLLECTION_NAME}"
            )

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=TOP_K,
            with_payload=True,
        )

        results = response.points

        print(f"\n=== TOP {len(results)} RESULTS ===")

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            source = payload.get("source", "<unknown>")
            chunk_index = payload.get("chunk_index", "?")
            text = payload.get("text", "")

            # Flatten whitespace to make results easier to read.
            preview = " ".join(text.split())

            if len(preview) > PREVIEW_LENGTH:
                preview = preview[:PREVIEW_LENGTH] + "..."

            print(f"\n--- Result {rank} ---")
            print(f"Score:  {result.score:.4f}")
            print(f"Source: {source}")
            print(f"Chunk:  {chunk_index}")
            print(f"Text:   {preview}")

        if not results:
            print("No results returned.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
