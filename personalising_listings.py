import logging
import os
import openai
import lancedb

from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database connection"""
    try:
        # Get the current directory and create a path for the database
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'lance_db')
        os.makedirs(db_path, exist_ok=True)
        
        # Connect to the LanceDB database
        db = lancedb.connect(db_path)
        return db
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def retrieve_top_recommendations(preference_questions, preference_answers) -> Tuple[List[Dict]]:
    """
    Retrieve top recommendations and associated preference data from LanceDB.
    """
    try:
        # Initialize the database connection
        db = initialize_database()
        table = db.open_table("top_recommendations")
        logger.info("Connected to database")

        # Convert table to pandas DataFrame for iteration
        df = table.to_pandas()
        
        if df.empty:
            logger.warning("No recommendations found in database")
            return [], [], []

        top_recommendations = []

        # Process each row in the DataFrame
        for _, row in df.iterrows():
            recommendation = {
                'id': row['id'],
                'description': row['description'],
                'neighborhood': row['neighborhood'],
                'price': row['price'],
                'location': row['location'],
                'bedrooms': row['bedrooms'],
                'bathrooms': row['bathrooms'],
                'size': row['size']
            }
            top_recommendations.append(recommendation)
        
        return top_recommendations

    except Exception as e:
        logger.error(f"Error retrieving recommendations: {str(e)}")
        raise

def personalize_property_descriptions(preference_questions, preference_answers, top_recommendations: List[Dict]) -> List[Dict]:
    """Generate personalized descriptions using OpenAI"""
    augmented_recommendations = []
    
    # Combine preference questions and answers into a single context string
    preference_pairs = [f"Q: {q} A: {a}" for q, a in zip(preference_questions, preference_answers)]
    preference_context = " | ".join(preference_pairs)
    logger.info(f"Generated preference context")
                    
    for property in top_recommendations:
        try:
            # Use OpenAI API to generate personalized property descriptions
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a professional real estate agent. Your task is to:
                    1. ONLY use facts directly stated in the original property description and property details
                    2. DO NOT add any features or amenities that are not explicitly mentioned
                    3. DO NOT incorporate buyer preferences unless they exactly match features mentioned in the property description
                    4. Rewrite the description into a paragraph in a more elaborate and engaging style while maintaining 100% factual accuracy
                    5. If uncertain about any detail, exclude it from the description"""},
                    {"role": "user", "content": f"""
                        Property Details (only include these if explicitly mentioned):
                        - Location: {property.get('location', 'N/A')}
                        - Size: {property.get('size', 'N/A')}
                        - Bedrooms: {property.get('bedrooms', 'N/A')}
                        - Bathrooms: {property.get('bathrooms', 'N/A')}
                        
                        Original property description: {property.get('description', '')}
                        Neighborhood description: {property.get('neighborhood', '')}
                        
                        Note: Below are buyer preferences for context only. DO NOT include these unless they exactly match features in the property description above:
                        {preference_context}
                    """}
                ],
                temperature=0.2,  # Reduced temperature for more consistent output
                max_tokens=300
            )
            
            # Add the personalized description to the property
            property['personalized_description'] = response.choices[0].message['content'].strip()
            augmented_recommendations.append(property)
            
        except Exception as e:
            logger.error(f"Error personalizing description: {str(e)}")
            property['personalized_description'] = property.get('description', '')
            augmented_recommendations.append(property)
    
    return augmented_recommendations


def display_augmented_recommendations(properties: List[Dict]) -> None:
    """Display formatted property recommendations"""
    print("\nTop Property Recommendations:")
    print("=" * 80)

    for idx, property in enumerate(properties, 1):
        print(f"\nProperty #{idx}")
        print("-" * 40)
        print(f"Location: {property.get('location', 'N/A')}")
        # Fix price formatting
        price = property.get('price', 'N/A')
        print(f"Price: {price}")
        print(f"Size: {property.get('size', 'N/A')}")
        print(f"Bedrooms: {property.get('bedrooms', 'N/A')}")
        print(f"Bathrooms: {property.get('bathrooms', 'N/A')}")
        print("\nOriginal Description:")
        print(property.get('description', 'No description available'))
        print("\nOriginal Neighborhood Description:")
        print(property.get('neighborhood', 'No description available'))
        print("\nPersonalized Description:")
        print(property.get('personalized_description', 'No description available'))
        print("-" * 80)