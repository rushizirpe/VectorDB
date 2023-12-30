# src/database/annoy_database.py
from annoy import AnnoyIndex
import torch
import numpy as np
from ..embeddings.with_gpu import WithGPU
from ..embeddings.without_gpu import WithoutGPU 

class AnnoyDatabase:
    def __init__(self, num_trees, documents, vector_dim=768):
        self.annoy_index = AnnoyIndex(vector_dim, 'angular')
        self.documents = documents
        
        for doc_id, doc_text in enumerate(documents):
            embedding = self.get_document_embedding(doc_text)
            self.annoy_index.add_item(doc_id, embedding)

        self.annoy_index.build(num_trees)

    def get_document_embedding(self, text, dev_type = "cpu"):
        # Implement the embedding function based on your chosen method
        if dev_type.lower() =="cpu":
            emb = WithoutGPU()
        elif dev_type.lower() =="gpu":
            emb = WithGPU()
        else:
            AssertionError("Device not mentioned")
        return emb.get_document_embedding(text)
            
    def query_database(self, query_text, num_neighbors=5):
        query_embedding = self.get_document_embedding(query_text)
        neighbor_ids = self.annoy_index.get_nns_by_vector(query_embedding, num_neighbors, search_k=-1)
        similar_documents = [self.documents[doc_id] for doc_id in neighbor_ids]
        return similar_documents
