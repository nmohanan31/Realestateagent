import logging
import os
import openai
import logging
import warnings
import logging

from generate_listings import generate_listings_llm
from listings_vector_database import load_and_prepare_data, create_embeddings, store_in_lancedb
from userinterface_and_search import similarity_search_store, get_listings
from personalising_listings import personalize_property_descriptions, retrieve_top_recommendations, display_augmented_recommendations
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings('ignore')

# Template for generating real estate listings
example_template = """
Location: {location}
Price: {price}
Bedrooms: {bedrooms}
Bathrooms: {bathrooms}
Size: {size}

Description: {description}

Neighborhood: {neighborhood}
"""

# Examples for few-shot learning
examples = [
    {
        "location": "Prenzlauer Berg, Berlin",
        "price": "€750,000",
        "bedrooms": "3",
        "bathrooms": "2",
        "size": "120 sqm",
        "description": "Charming pre-war apartment with high ceilings and modern amenities. Renovated kitchen and bathrooms. Balcony with garden views. Parking space included.",
        "neighborhood": "Trendy area with cafes and parks. The U-bhan station is a 5-minute walk away. Tram stop right outside the building."
    },
    {
        "location": "Mitte, Berlin",
        "price": "€950,000",
        "bedrooms": "4",
        "bathrooms": "2.5",
        "size": "150 sqm",
        "description": "Luxury penthouse with panoramic city views and private terrace. High-end finishes and designer furniture. Concierge service and fitness center in the building. Parking available.",
        "neighborhood": "Central location with excellent transport links. The S-bhan station is a 10-minute walk away. Bus stop right outside the building."
    }
]

# Define preference questions and answers
preference_questions = [
    "How big do you want your house to be?",
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?"
]

preference_answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

# OpenAI setup
os.environ["OPENAI_API_KEY"] = "voc-12568918961266773670401673e1acb29caf9.60086405"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

def main():

    # Generate listings and print them
    listings = generate_listings_llm(example_template, examples)
    for i, listing in enumerate(listings, 1):
        print(f"\nListing {i}:")
        print(listing)
        print("-" * 50)

    # Run the main function
    logger.info("Loading data...")
    df = load_and_prepare_data()
    
    logger.info("Generating embeddings...")
    embeddings = create_embeddings(df)
    
    logger.info("Storing data in LanceDB...")
    lancedb_table = store_in_lancedb(df, embeddings)
    lance_table_df = lancedb_table.to_pandas()
    
    # Print the original dataset, embeddings, and entries from LanceDB
    print("\nOriginal Dataset:")
    print(df)
    print("\nEmbeddings (10x10):")
    print(embeddings[:10, :10])
    print("\nEntries from LanceDB:")
    print(lance_table_df)

    # Get property listings based on preference answers
    properties = get_listings(preference_answers)
    # Store and display top recommendations
    similarity_search_store(properties)

    # Retrieve top recommendations from the database
    recommendations = retrieve_top_recommendations(preference_questions, preference_answers)
    # Generate personalized descriptions for the recommendations
    augmented_recommendations = personalize_property_descriptions(preference_questions, preference_answers, recommendations)
    # Display the augmented recommendations
    display_augmented_recommendations(augmented_recommendations)

if __name__ == "__main__":
    main()