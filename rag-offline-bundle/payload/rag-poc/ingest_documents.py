from pathlib import Path
import re

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
DATABASE_DIR = PROJECT_DIR / "rag_db"

COLLECTION_NAME = "text_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250
EMBED_BATCH_SIZE = 32
EXPECTED_VECTOR_SIZE = 384


_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Fig|No|Inc|Ltd|Dept|approx|U\.S|U\.K)\.",
    re.IGNORECASE,
)


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into approximate sentences without extra packages."""
    # Avoid treating common abbreviations as sentence endings.
    protected = _ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "\ue000"),
        paragraph,
    )

    # Keep closing quotes and brackets attached to their sentences.
    pieces = re.findall(
        r"\S.*?(?:[.!?]+[\"'”’)\]]*(?=\s+|$)|$)",
        protected,
    )

    return [
        piece.replace("\ue000", ".").strip()
        for piece in pieces
        if piece.strip()
    ]


def split_long_sentence(sentence: str) -> list[str]:
    """Split oversized sentences at word boundaries."""
    if len(sentence) <= CHUNK_SIZE:
        return [sentence]

    words = sentence.split()
    pieces = []
    current_words = []

    for word in words:
        candidate = " ".join(current_words + [word])

        if current_words and len(candidate) > CHUNK_SIZE:
            pieces.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def render_chunk_units(units: list[tuple[str, int]]) -> str:
    """Join sentence units while retaining paragraph boundaries."""
    parts = []
    previous_paragraph = None

    for text, paragraph_index in units:
        if parts:
            separator = (
                "\n\n"
                if paragraph_index != previous_paragraph
                else " "
            )
            parts.append(separator)

        parts.append(text)
        previous_paragraph = paragraph_index

    return "".join(parts)


def chunk_text(text: str) -> list[str]:
    """Create sentence-aware, overlapping chunks bounded by CHUNK_SIZE."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    # Each unit contains text and its original paragraph index.
    units = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            for piece in split_long_sentence(sentence):
                units.append((piece, paragraph_index))

    chunks = []
    current = []

    for unit in units:
        if not current:
            current = [unit]
            continue

        candidate = render_chunk_units(current + [unit])

        if len(candidate) <= CHUNK_SIZE:
            current.append(unit)
            continue

        chunks.append(render_chunk_units(current))

        # Keep whole trailing sentences for overlap, targeting roughly
        # CHUNK_OVERLAP characters. Limit overlap to twice that target
        # to avoid duplicating most of a chunk.
        overlap = []

        for previous in reversed(current):
            trial = [previous] + overlap
            trial_size = len(render_chunk_units(trial))

            if trial_size > CHUNK_OVERLAP * 2:
                break

            overlap = trial

            if trial_size >= CHUNK_OVERLAP:
                break

        # Ensure the next chunk stays within the size limit.
        while (
            overlap
            and len(render_chunk_units(overlap + [unit])) > CHUNK_SIZE
        ):
            overlap.pop(0)

        current = overlap + [unit]

    if current:
        chunks.append(render_chunk_units(current))

    return [chunk for chunk in chunks if chunk.strip()]

def main() -> None:
    if not DOCUMENTS_DIR.exists():
        raise SystemExit(
            f"Documents directory does not exist: {DOCUMENTS_DIR}"
        )

    files = sorted(DOCUMENTS_DIR.rglob("*.txt"))

    if not files:
        raise SystemExit(
            f"No .txt files found in {DOCUMENTS_DIR}"
        )

    records = []

    # Read files and produce chunk records.
    for file_path in files:
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        file_chunks = chunk_text(text)
        relative_source = str(file_path.relative_to(DOCUMENTS_DIR))

        print(
            f"Read {relative_source}: "
            f"{len(text)} characters, {len(file_chunks)} chunks"
        )

        for chunk_index, chunk in enumerate(file_chunks, start=1):
            records.append(
                {
                    "source": relative_source,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not records:
        raise SystemExit(
            "No nonempty text chunks found. Check the input files."
        )

    print(f"\nFiles found: {len(files)}")
    print(f"Total chunks: {len(records)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    # Load the embedding model before modifying the database.
    model = TextEmbedding(model_name=MODEL_NAME)

    client = QdrantClient(path=str(DATABASE_DIR))

    try:
        # Rebuild only our project collection on each ingestion run.
        # This prevents stale or duplicate chunks after document changes.
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EXPECTED_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        # Embed and upsert small batches to control memory usage.
        for start in range(0, len(records), EMBED_BATCH_SIZE):
            batch = records[start:start + EMBED_BATCH_SIZE]

            # BGE supports passage prefixes for retrieval tasks.
            model_inputs = [
                f"passage: {record['text']}"
                for record in batch
            ]

            vectors = list(model.embed(model_inputs))

            if len(vectors) != len(batch):
                raise RuntimeError(
                    "The embedding count did not match the chunk count."
                )

            points = []

            for offset, (record, vector) in enumerate(
                zip(batch, vectors)
            ):
                if len(vector) != EXPECTED_VECTOR_SIZE:
                    raise RuntimeError(
                        f"Unexpected vector size: {len(vector)}"
                    )

                point_id = start + offset + 1

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=record,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            print(
                f"Stored {min(start + len(batch), len(records))}"
                f"/{len(records)} chunks"
            )

        # Verify what was persisted in Qdrant.
        stored_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        if stored_count != len(records):
            raise RuntimeError(
                f"Expected {len(records)} points, "
                f"but found {stored_count}."
            )

        print("\n=== INGESTION SUCCESS ===")
        print(f"Source files: {len(files)}")
        print(f"Chunks embedded: {len(records)}")
        print(f"Vectors stored: {stored_count}")
        print(f"Vector dimensions: {EXPECTED_VECTOR_SIZE}")
        print(f"Database directory: {DATABASE_DIR}")
        print(f"Collection: {COLLECTION_NAME}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
