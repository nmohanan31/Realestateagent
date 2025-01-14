import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
import lancedb
import os
import pyarrow as pa
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
def load_and_prepare_data(file_path='berlin_listings.json'):
    try:
        df = pd.read_json(file_path)
        required_columns = ['Description', 'Location', 'Price', 'Bedrooms', 'Bathrooms', 'Size', 'Neighborhood']
        
        # Validate columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # # Add numeric ID
        # df['id'] = range(1, len(df) + 1)
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_embeddings(df):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Combine text fields
        text_data = (
            df['Description'].fillna('') + ' ' + 
            df['Location'].fillna('') + ' ' + 
            df['Neighborhood'].fillna('')
        )
        
        # Validate input text
        if text_data.empty or text_data.isna().all():
            raise ValueError("No valid text data to encode")
            
        # Generate embeddings
        embeddings = model.encode(text_data.tolist(), normalize_embeddings=True)
        
        # Convert to float32 and verify shape
        embeddings = embeddings.astype(np.float32)
        
        # Additional validations
        if embeddings.shape[0] != len(df):
            raise ValueError(f"Embedding count ({embeddings.shape[0]}) doesn't match dataframe length ({len(df)})")
        
        if embeddings.shape[1] != 384:  # all-MiniLM-L6-v2 produces 384-dimensional vectors
            raise ValueError(f"Unexpected embedding dimension: {embeddings.shape[1]}")
            
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embeddings dtype: {embeddings.dtype}")
        
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")
            
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

# def calculate_similarities(embeddings):
#     try:
#         similarity_matrix = cosine_similarity(embeddings)
#         return similarity_matrix
#     except Exception as e:
#         logger.error(f"Error calculating similarities: {str(e)}")
#         raise

def store_in_lancedb(df, embeddings):
    """Store listings with vector embeddings"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'lance_db')

        # Clear existing DB
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)

        # Create PyArrow schema for fixed-size vector
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32(), 384)),  # Fixed 384-dim vector
            pa.field("description", pa.string()),
            pa.field("price", pa.string()),
            pa.field("location", pa.string()),
            pa.field("bedrooms", pa.string()),
            pa.field("bathrooms", pa.string()),
            pa.field("size", pa.string())
        ])

        # Create records
        records = []
        for idx, row in df.iterrows():
            record = {
                "id": idx,
                "embedding": embeddings[idx].astype(np.float32),
                "description": row['Description'],
                "price": str(row['Price']),
                "location": row['Location'],
                "bedrooms": str(row['Bedrooms']),
                "bathrooms": str(row['Bathrooms']),
                "size": str(row['Size'])
            }
            records.append(record)

        # Create table with schema
        db = lancedb.connect(db_path)
        table = db.create_table(
            "listings",
            schema=schema,
            data=records,
            mode="overwrite"
        )

        # # Create vector index
        # table.create_index(
        #     "embedding",
        #     #metric_type="cosine",
        #     replace=True
        #)

        logger.info(f"Created table with {len(records)} records")
        return table

    except Exception as e:
        logger.error(f"Error storing data in LanceDB: {str(e)}")
        raise

def main():
    try:
        logger.info("Loading data...")
        df = load_and_prepare_data()
        
        logger.info("Generating embeddings...")
        embeddings = create_embeddings(df)
        
        # logger.info("Calculating similarities...")
        # similarity_matrix = calculate_similarities(embeddings)
        
        # similarity_df = pd.DataFrame(
        #     similarity_matrix,
        #     #index=df['id'],
        #     #columns=df['id']
        # )

        logger.info("Storing data in LanceDB...")
        lancedb_table = store_in_lancedb(df, embeddings)
        lance_table_df = lancedb_table.to_pandas()
        
        print("\nOriginal Dataset:")
        print(df)
        print("\nEmbeddings (10x10):")
        print(embeddings[:10, :10])
        # print("\nSimilarity Matrix (10x10):")
        # print(similarity_df.iloc[:10, :10])
        print("\nEntries from LanceDB:")
        print(lance_table_df)      
                
        # return similarity_df
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()