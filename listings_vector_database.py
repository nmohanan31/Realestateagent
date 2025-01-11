import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import logging
import lancedb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='berlin_listings.csv'):
    try:
        df = pd.read_csv(file_path)
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
        
        # Combine text fields with proper column names
        text_data = (
            df['Description'].fillna('') + ' ' + 
            df['Location'].fillna('') + ' ' + 
            df['Neighborhood'].fillna('')
        )
        
        embeddings = model.encode(text_data.tolist())
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def calculate_similarities(embeddings):
    try:
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        logger.error(f"Error calculating similarities: {str(e)}")
        raise

def store_in_lancedb(df, embeddings):
    try:
        
        # Create or connect to a database
        db = lancedb.connect("listings_db")
        
        # Prepare data for LanceDB
        data = []
        for i, row in df.iterrows():
            data.append({
                "description": row["Description"],
                "location": row["Location"],
                "price": row["Price"],
                "bedrooms": row["Bedrooms"],
                "bathrooms": row["Bathrooms"],
                "size": row["Size"],
                "neighborhood": row["Neighborhood"],
                "vector": embeddings[i]
            })
        
        # Create or overwrite table
        table = db.create_table("listings", data=data, mode="overwrite")
        logger.info("Data successfully stored in LanceDB")
        
        # # Display some entries from the database
        # table_pd = table.to_pandas()
        # print("\nEntries from LanceDB:")
        # print(table_pd)
        
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
        
        logger.info("Calculating similarities...")
        similarity_matrix = calculate_similarities(embeddings)
        
        similarity_df = pd.DataFrame(
            similarity_matrix,
            #index=df['id'],
            #columns=df['id']
        )

        logger.info("Storing data in LanceDB...")
        lancedb_table = store_in_lancedb(df, embeddings)
        lance_table_df = lancedb_table.to_pandas()
        
        print("\nOriginal Dataset:")
        print(df)
        print("\nEmbeddings (10x10):")
        print(embeddings[:10, :10])
        print("\nSimilarity Matrix (10x10):")
        print(similarity_df.iloc[:10, :10])
        print("\nEntries from LanceDB:")
        print(lance_table_df)      
                
        return similarity_df
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()