import math
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"

texts = [
    "DNS resolves hostnames into IP addresses.",
    "Domain name resolution translates a website name to an IP.",
    "Bread recipes use flour, water, and yeast.",
]

print(f"Loading embedding model: {MODEL_NAME}")
print("The first run may download the model files.")

model = TextEmbedding(model_name=MODEL_NAME)
vectors = list(model.embed(texts))

# Verify one vector was produced for every input.
assert len(vectors) == len(texts), (
    f"Expected {len(texts)} vectors, got {len(vectors)}"
)

# This model should produce 384-dimensional vectors.
dimensions = {len(vector) for vector in vectors}
assert dimensions == {384}, (
    f"Unexpected vector dimensions: {dimensions}"
)

# Make sure the vectors contain valid numeric values.
for vector in vectors:
    assert all(
        math.isfinite(float(value)) for value in vector
    ), "Found an invalid numeric value"

print(f"\nModel: {MODEL_NAME}")
print(f"Input texts: {len(texts)}")
print(f"Vectors generated: {len(vectors)}")
print(f"Vector dimensions: {len(vectors[0])}")
print(f"First 5 values: {vectors[0][:5]}")

print("\nALL EMBEDDING TESTS PASSED")
