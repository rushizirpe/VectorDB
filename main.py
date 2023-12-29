# from src.query.query import Query
# from src.database.annoy_database import AnnoyDatabase


# db_inst = AnnoyDatabase()

# db_inst = Query()

# main.py
import torch
from src.embeddings.without_gpu import WithoutGPU
from src.embeddings.with_gpu import WithGPU
from src.database.annoy_database import AnnoyDatabase
from src.database.faiss_database import FaissDatabase
from src.query.query import QueryProcessor
import json

def main():
    # Initialize embeddings (choose either WithGPU or WithoutGPU based on availability)
    embeddings = WithGPU() if torch.cuda.is_available() else WithoutGPU()

    # Load data
    data_path = "data/combined_questions_filtered.json"
    data = load_data(data_path)

    # Initialize database (choose either AnnoyDatabase or FaissDatabase)
    database = AnnoyDatabase(embeddings, num_trees=10)

    # Initialize query processor
    query_processor = QueryProcessor(embeddings, database)

    # Get user input (prompt)
    user_input = input("Enter a prompt: ")

    # Process the query and get similar meanings
    similar_meanings = query_processor.process_query(user_input, num_neighbors=5)

    # Display the results
    print("Similar Meanings:")
    for meaning in similar_meanings:
        print(meaning)

def load_data(data_path):
    # Implement your actual data loading logic here
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def populate_database(database, data):
    # Add data to the database
    for doc_id, doc_text in enumerate(data):
        embedding = embeddings.get_document_embedding(doc_text)
        database.add_item(doc_id, embedding)

if __name__ == "__main__":
    main()
