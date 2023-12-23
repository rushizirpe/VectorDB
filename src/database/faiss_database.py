# src/database/faiss_database.py
import faiss
import torch
import numpy as np

class FaissDatabase:
    def __init__(self, vector_dim, num_clusters, documents):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(vector_dim), vector_dim, num_clusters, faiss.METRIC_INNER_PRODUCT)
        self.documents = documents

        document_embeddings = np.array([self.get_document_embedding(doc)[:vector_dim] for doc in documents], dtype=np.float32)
        self.index.train(document_embeddings)
        self.index.add(document_embeddings)

    def get_document_embedding(self, text):
        # Implement the embedding function based on your chosen method
        pass

    def query_database(self, query_text, num_neighbors=5):
        query_embedding = self.get_document_embedding(query_text)[:vector_dim]
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        query_embedding_gpu = torch.tensor(query_embedding, device=self.device)
        _, neighbor_indices = self.index.search(np.expand_dims(query_embedding_gpu.cpu().numpy(), axis=0), num_neighbors)
        neighbor_indices_cpu = neighbor_indices[0].cpu().numpy()
        nearest_neighbor_embeddings = np.array([self.documents[i] for i in neighbor_indices_cpu])
        return nearest_neighbor_embeddings
