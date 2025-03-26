from tqdm import tqdm
import json
import os
import time

METADATA_PATH = "courses.json"
EMBEDDING_PATH = "3D_vectors.json"
OUTPUT_PATH = "output_vectors.json"

with open(EMBEDDING_PATH, "r") as f:
    embeddings = json.load(f)
print(f"Loaded embeddings: {len(embeddings)}")
embedding_map = {e["number"]: e for e in embeddings}
print(f"Created embedding map!")
results = []
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
    start = time.time()
    for course in tqdm(metadata):
        embedding = embedding_map[course["courseNumber"]]
        embedding_vector = [embedding["x"], embedding["y"], embedding["z"]]
        course_object = {
            "number": course["courseNumber"],
            "metadata": {
                "title": course["courseName"],
                "description": course["courseDescription"]
            },
            "embedding": embedding_vector
        }
        results.append(course_object)
    end = time.time()
print(f"Added all data objects to memory: {len(results)}\nTime used: {end - start:.4f}\n")
    
with open(OUTPUT_PATH, "w") as output_file:
    json.dump(results, output_file, indent = 2)
print(f"Saved to file: {OUTPUT_PATH}")




