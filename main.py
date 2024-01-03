# main.py
import torch
from src.embeddings.without_gpu import WithoutGPU
from src.embeddings.with_gpu import WithGPU
from src.database.annoy_database import AnnoyDatabase
from src.database.faiss_database import FaissDatabase
from src.query.query import QueryProcessor
import json

def main():
    # Initialize embeddings (either WithGPU or WithoutGPU based on availability)
    # embeddings = WithGPU() if torch.cuda.is_available() else WithoutGPU()

    # Load data
    data_path = "data/combined_questions_filtered.json"
    data = load_data(data_path)
    docs = [triplet["answer"] for triplet in data][:10]

    # Initialize database (AnnoyDatabase or FaissDatabase)
    database = AnnoyDatabase(num_trees=10, documents=docs)

def load_data(data_path):
    with open(data_path, 'r', encoding = "utf-8") as json_file:
        data = json.load(json_file)
    return data

if __name__ == "__main__":
    main()
