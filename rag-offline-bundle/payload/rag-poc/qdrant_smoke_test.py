from qdrant_client import QdrantClient, models

DB_PATH = "qdrant_smoke_db"
COLLECTION = "smoke_test"

# Open a persistent, local Qdrant database.
client = QdrantClient(path=DB_PATH)

try:
    # Reset only this test collection if the script has run before.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    # Create a collection that accepts 3-dimensional vectors.
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        ),
    )

    # Insert three test vectors and their descriptive payloads.
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={"label": "exact match"},
            ),
            models.PointStruct(
                id=2,
                vector=[0.9, 0.1, 0.0],
                payload={"label": "similar match"},
            ),
            models.PointStruct(
                id=3,
                vector=[0.0, 1.0, 0.0],
                payload={"label": "unrelated match"},
            ),
        ],
    )

    # Search for vectors most similar to our query.
    results = client.query_points(
        collection_name=COLLECTION,
        query=[1.0, 0.0, 0.0],
        limit=3,
    ).points

    print("=== Initial similarity search ===")
    for result in results:
        print(
            f"ID={result.id} | "
            f"Score={result.score:.4f} | "
            f"Payload={result.payload}"
        )

    assert results and results[0].id == 1, (
        "FAIL: The exact-match vector was not ranked first."
    )

    print("\nInitial search: PASS")

finally:
    client.close()


# Reopen the database to verify that the data persisted on disk.
client = QdrantClient(path=DB_PATH)

try:
    assert client.collection_exists(COLLECTION), (
        "FAIL: The collection did not persist."
    )

    results = client.query_points(
        collection_name=COLLECTION,
        query=[1.0, 0.0, 0.0],
        limit=1,
    ).points

    assert results and results[0].id == 1, (
        "FAIL: The stored vector could not be retrieved after reopening."
    )

    print("Persistence after reopening: PASS")

finally:
    client.close()

print("\nALL TESTS PASSED")
