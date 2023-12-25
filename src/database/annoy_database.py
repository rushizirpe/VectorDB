# src/database/annoy_database.py
from annoy import AnnoyIndex
import torch
import numpy as np
from ..embeddings.with_gpu import WithGPU
from ..embeddings.without_gpu import WithoutGPU 

class AnnoyDatabase:
    def __init__(self, vector_dim, num_trees, documents):
        self.annoy_index = AnnoyIndex(vector_dim, 'angular')
        self.documents = documents

        for doc_id, doc_text in enumerate(documents):
            embedding = self.get_document_embedding(doc_text)
            self.annoy_index.add_item(doc_id, embedding)

        self.annoy_index.build(num_trees)

    def get_document_embedding(self, text, type = "cpu"):
        # Implement the embedding function based on your chosen method
        if type.lower() =="cpu":
            emb = WithoutGPU()
            emb.get_document_embedding(text)
        elif type.lower() =="gpu":
            emb = WithGPU()
            emb.get_document_embedding(text)
        else:
            AssertionError("Device not mentioned")
            
    def query_database(self, query_text, num_neighbors=5):
        query_embedding = self.get_document_embedding(query_text)
        neighbor_ids = self.annoy_index.get_nns_by_vector(query_embedding, num_neighbors, search_k=-1)
        similar_documents = [self.documents[doc_id] for doc_id in neighbor_ids]
        return similar_documents
