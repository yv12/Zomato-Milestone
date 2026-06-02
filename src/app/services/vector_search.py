import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.data.repository import RestaurantRepository
from app.models.domain import Restaurant

class VectorSearchEngine:
    def __init__(self, repository: RestaurantRepository):
        self.repository = repository
        self.model_name = settings.embedding_model_name
        self.db_dir = os.path.dirname(settings.absolute_data_path)
        self.vector_path = os.path.join(self.db_dir, "embeddings.npy")
        
        # Load local model and embedding vectors
        print(f"Loading local SentenceTransformer model '{self.model_name}' inside search engine...")
        self.model = SentenceTransformer(self.model_name)
        
        if not os.path.exists(self.vector_path):
            raise FileNotFoundError(
                f"Embedding vectors file not found at {self.vector_path}. Please run the data ingestion pipeline first."
            )
            
        print(f"Loading dense vector representations from {self.vector_path}...")
        self.embeddings = np.load(self.vector_path)
        print(f"Successfully loaded embeddings of shape {self.embeddings.shape}.")

    def search(self, query: str, candidates: List[Restaurant], top_n: int) -> List[Restaurant]:
        """Performs local vector semantic search, ranking candidate subsets by cosine similarity."""
        if not candidates:
            return []
            
        # 1. Edge Case: Empty qualitative query (Bypass semantic search)
        if not query or not query.strip():
            print("Qualitative preferences query is empty. Bypassing vector search and maintaining hierarchical ranking.")
            return candidates[:top_n]

        # 2. Vectorize the query locally
        query_text = query.strip()
        query_vector = self.model.encode(query_text, convert_to_numpy=True)
        
        # 3. Retrieve embedding indices for candidate subset
        candidate_ids = [c.id for c in candidates]
        
        # Create a mapping from restaurant ID to its index in the cached repository dataframe
        id_to_index = {row['id']: idx for idx, row in self.repository._df.iterrows()}
        candidate_indices = [id_to_index[cid] for cid in candidate_ids if cid in id_to_index]
        
        if not candidate_indices:
            print("Warning: Candidates could not be mapped to vector indices. Serving database defaults.")
            return candidates[:top_n]
            
        # 4. Slice the dense embeddings array
        candidate_vectors = self.embeddings[candidate_indices]
        
        # 5. Compute Cosine Similarity between query vector (1, D) and candidate vectors (M, D)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            query_norm = 1.0
        normalized_query = query_vector / query_norm
        
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
        candidate_norms[candidate_norms == 0] = 1.0
        normalized_candidates = candidate_vectors / candidate_norms
        
        # Scores are between -1.0 and 1.0 representing similarity
        scores = np.dot(normalized_candidates, normalized_query)
        
        # 6. Pair, Rank, and Cap to top_n
        paired_results = list(zip(candidates, scores))
        # Sort descending by similarity score
        paired_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Semantic search ranked {len(paired_results)} candidates. Top similarity score: {paired_results[0][1]:.4f}")
        
        ranked_candidates = [res for res, score in paired_results[:top_n]]
        return ranked_candidates
