import os
import sqlite3
import numpy as np
import pandas as pd

from app.config import settings
from .loader import DatasetLoader
from .normalizer import SchemaNormalizer

def run_ingestion():
    print("=== Starting Zomato Local Vector Ingestion Pipeline ===")
    
    # 1. Fetch
    loader = DatasetLoader()
    raw_df = loader.fetch_as_dataframe()
    
    # 2. Normalize
    normalizer = SchemaNormalizer(
        budget_low_max=settings.budget_low_max, 
        budget_medium_max=settings.budget_medium_max
    )
    clean_df = normalizer.normalize(raw_df)
    
    # 3. Local Embedding Generation
    print(f"Loading local sentence embedding model '{settings.embedding_model_name}'...")
    # This automatically downloads/loads the MiniLM model locally
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.embedding_model_name)
    
    print("Generating dense vector representations of restaurant fields locally...")
    texts = clean_df['text_representation'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings array of shape {embeddings.shape}.")
    
    # 4. Save Structured Records to SQLite
    db_path = settings.absolute_data_path
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"Connecting to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # For SQLite storage, we serialize list columns to comma-separated strings
    sqlite_df = clean_df.copy()
    sqlite_df['cuisines'] = sqlite_df['cuisines'].apply(lambda x: ",".join(x))
    sqlite_df['dish_liked'] = sqlite_df['dish_liked'].apply(lambda x: ",".join(x))
    
    print("Writing metadata records to table 'restaurants'...")
    sqlite_df.to_sql(name="restaurants", con=conn, if_exists="replace", index=False)
    
    # Create indexes for fast hierarchical filtering
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON restaurants(location)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_budget ON restaurants(budget_band)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON restaurants(rating)")
    conn.commit()
    conn.close()
    
    # 5. Save Embedding Vectors
    vector_path = os.path.join(db_dir, "embeddings.npy")
    print(f"Saving dense embeddings array to {vector_path}...")
    np.save(vector_path, embeddings)
    
    print("=== Local Vector Ingestion Pipeline Complete ===")

if __name__ == "__main__":
    run_ingestion()
