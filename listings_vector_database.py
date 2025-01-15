import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
import logging
import lancedb
import os
import pyarrow as pa
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='berlin_listings.json'):
    try:
        # Load data from JSON file
        df = pd.read_json(file_path)
        
        # Define required columns
        required_columns = ['Description', 'Location', 'Price', 'Bedrooms', 'Bathrooms', 'Size', 'Neighborhood']
        
        # Validate columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Return the dataframe
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_embeddings(df):
    try:
        # Load pre-trained SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Combine text fields into a single string for each row
        text_data = (
            df['Description'].fillna('') + ' ' + 
            df['Location'].fillna('') + ' ' + 
            df['Neighborhood'].fillna('') + ' ' +
            df['Bedrooms'].fillna('').astype(str) + ' '
        )
        
        # Validate input text
        if text_data.empty or text_data.isna().all():
            raise ValueError("No valid text data to encode")
        
        # Generate embeddings
        embeddings = model.encode(text_data.tolist(), normalize_embeddings=True)
        
        # Convert embeddings to float32
        embeddings = embeddings.astype(np.float32)
        
        # Validate embeddings shape
        if embeddings.shape[0] != len(df):
            raise ValueError(f"Embedding count ({embeddings.shape[0]}) doesn't match dataframe length ({len(df)})")
        
        # Validate embedding dimensions
        if embeddings.shape[1] != 384:  # all-MiniLM-L6-v2 produces 384-dimensional vectors
            raise ValueError(f"Unexpected embedding dimension: {embeddings.shape[1]}")
        
        # Log embeddings information
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embeddings dtype: {embeddings.dtype}")
        
        # Check for NaN values in embeddings
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")
        
        # Return embeddings
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def store_in_lancedb(df, embeddings):
    """Store listings with vector embeddings"""
    try:
        # Get current directory and set database path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'lance_db')

        # Clear existing database if it exists
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
            pa.field("size", pa.string()),
            pa.field("neighborhood", pa.string())
        ])

        # Create records from dataframe and embeddings
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
                "size": str(row['Size']),
                "neighborhood": row['Neighborhood']
            }
            records.append(record)

        # Connect to LanceDB and create table with schema
        db = lancedb.connect(db_path)
        table = db.create_table(
            "listings",
            schema=schema,
            data=records,
            mode="overwrite"
        )

        # Log the number of records created
        logger.info(f"Created table with {len(records)} records")
        
        # Return the table
        return table

    except Exception as e:
        logger.error(f"Error storing data in LanceDB: {str(e)}")
        raise
