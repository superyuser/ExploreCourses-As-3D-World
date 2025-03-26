from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pinecone import Index as PineconeIndex
import json
import os
import time
import numpy as np
import pinecone
import plotly.express as px
import pandas as pd
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

cache_vectors = []

DATA_PATH = "courses.json"
COORDINATES_PATH = "3D_vectors.json"

embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", # 384
    model_kwargs={
        'device': 'cpu'
    },
    encode_kwargs={
        'normalize_embeddings': False
    }
)

def load_courses(k):
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        if k == "all":
            to_return = data
        else:
            to_return = data[:k]
    return to_return

# helper (embed_courses)
def get_embedding(description):
    return embedding_fn.embed_query(description)

# @lru_cache(maxsize = None)
def embed_courses(courses):
    global cache_vectors
    course_description = [course["courseDescription"] for course in courses]
    course_nums = [course["courseNumber"] for course in courses]
    course_embeddings = embedding_fn.embed_documents(course_description)
    course_vectors = [
        {
            "number": course_nums[i],
            "values": embedding
        }
        for i, embedding in tqdm(list(enumerate(course_embeddings)), desc="😎embedding courses")
    ]
    cache_vectors = course_vectors
    return course_vectors

# helper to plot output 3D embeddings
def visualize_3D(data):
    df = pd.DataFrame(data)
    fig = px.scatter_3d(
        df,
        x = "x",
        y = "y",
        z = "z",
        hover_name = "number",
        opacity = 0.7
    )
    fig.update_traces(textposition = "top center", marker = dict(size=4))
    fig.update_layout(title = "Course embeddings in 3D space")
    fig.show()

# @lru_cache(maxsize = None)
def reduce_dims(embeddings, t_sne = False, pca = True):
    X = np.array([embedding["values"] for embedding in embeddings])
    course_nums = [embedding["number"] for embedding in embeddings]
    print(f"Embedding shape: {X.shape}")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    if pca:
        alg = PCA(
            n_components = 3,
            copy = True,
            random_state = 42
        )
        X_pca = alg.fit_transform(scaled_data)
        reduced = X_pca
    elif t_sne:
        alg = TSNE(
            n_components = 2,
            perplexity = 30,
            random_state = 42
        )
        X_tsne = alg.fit_transform(scaled_data)
        reduced = X_tsne
    print(f"Reduced-dims shape: {reduced.shape}")
    data_obj = [{"number": course_num, 
                 "x": coordinates[0],
                 "y": coordinates[1],
                 "z": coordinates[2]} for course_num, coordinates in zip(course_nums, reduced)]
    visualize_3D(data_obj)
    dump_3d_embeddings(data_obj)
    print("Dumped 3D embeddings!")
    return data_obj

# output to json, very helper
def dump_3d_embeddings(data, path = COORDINATES_PATH):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# helper to print out courses after initial load
def print_metadata(courses):
    for course in courses:
        print(f"Number: {course['number']}\nSize: {len(course['values'])}")

if __name__ == "__main__":
    start = time.time()
    data = load_courses("all")
    embeddings = embed_courses(data)
    reduced = reduce_dims(cache_vectors)
    end = time.time()
    print(f"⌚Taken {end - start:.4f} seconds to run.")
    # with open(COORDINATES_PATH, 'r') as f:
    #     data = json.load(f)
    #     visualize_3D(data)
