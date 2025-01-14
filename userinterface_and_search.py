import os
import csv
import logging
import re
import numpy as np
import lancedb

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List
from sentence_transformers import SentenceTransformer

# Environment variables (placeholders)
os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"
os.environ["OPENAI_API_BASE"] = "https://openai.example.com/v1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def initialize_database():
    """Initialize database connection"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'lance_db')
        os.makedirs(db_path, exist_ok=True)
        db = lancedb.connect(db_path)
        return db
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define weighting per category
SEARCH_WEIGHTS = {
    "size": 0.25,      # Size and layout
    "location": 0.25,  # Location features
    "amenities": 0.20, # Property amenities
    "transport": 0.15, # Transportation options
    "urban": 0.15      # Urban/suburban balance
}

def create_category_queries(preference_answers):
    """Map preference answers to search categories."""
    return {
        "size": preference_answers[0],
        "location": preference_answers[1],
        "amenities": preference_answers[2],
        "transport": preference_answers[3],
        "urban": preference_answers[4]
    }

def convert_to_rating(score):
    """Convert similarity score to 1-5 rating with weight scaling"""
    # Calculate average weight to use as scaling factor
    avg_weight = sum(SEARCH_WEIGHTS.values()) / len(SEARCH_WEIGHTS)
    # Apply weight-based scaling with exponential factor
    scaled_score = score * (1 + avg_weight) ** 2
    # Convert to 1-5 range with rounding
    return min(max(round(scaled_score * 5), 1), 5)

def get_similar_listings(preference_answers, k=5):
    try:
        db = initialize_database()
        table = db.open_table("listings")
        logger.info("Connected to database")
        
        category_queries = create_category_queries(preference_answers)
        property_scores = {}
        
        for category, query in category_queries.items():
            query_vector = model.encode(query).astype(np.float32)
            results = table.search(
                query=query_vector,
                vector_column_name="embedding"
            ).limit(k).to_list()
            
            for r in results:
                if '_distance' in r:
                    property_id = r['id']
                    similarity_score = 1 / (1 + r['_distance'])
                    weighted_score = similarity_score * SEARCH_WEIGHTS[category]
                    
                    if property_id not in property_scores:
                        property_scores[property_id] = {
                            'total_score': 0,
                            'category_scores': {},
                            'details': {
                                'description': r['description'],
                                'price': r['price'],
                                'location': r['location'],
                                'bedrooms': r['bedrooms'],
                                'bathrooms': r['bathrooms'],
                                'size': r['size']
                            }
                        }
                    
                    property_scores[property_id]['category_scores'][category] = convert_to_rating(weighted_score)
                    property_scores[property_id]['total_score'] += weighted_score

        # Convert total scores to 1-10 rating
        for prop_data in property_scores.values():
            prop_data['rating'] = convert_to_rating(prop_data['total_score'])

        # Sort by rating and get top property
        sorted_properties = sorted(
            property_scores.items(),
            key=lambda x: x[1]['rating'],
            reverse=True
        )

        if not sorted_properties:
            return []

        # Display all properties with ratings
        print("\n=== ALL PROPERTIES RATINGS ===")
        for prop_id, prop_data in sorted_properties:
            print(f"\nProperty ID: {prop_id}")
            print(f"Location: {prop_data['details']['location']}")
            print(f"Price: {prop_data['details']['price']}")
            print(f"Size: {prop_data['details']['size']}")
            print(f"Bedrooms: {prop_data['details']['bedrooms']}")
            print(f"Bathrooms: {prop_data['details']['bathrooms']}")
            print("\nCategory Ratings (1-5):")
            for cat, score in prop_data['category_scores'].items():
                print(f"- {cat.title()}: {score}/5")
            print(f"OVERALL RATING: {prop_data['rating']}/5")
            print("-" * 30)

        # Display top 3 recommendations
        print("\n" + "=" * 50)
        print("TOP 3 RECOMMENDATIONS".center(50))
        print("=" * 50)

        top_recommendations = []
        for i, (prop_id, prop_data) in enumerate(sorted_properties[:3], 1):
            print(f"\n{i}. PROPERTY RECOMMENDATION")
            print("-" * 20)
            print(f"Property ID: {prop_id}")
            print(f"Location: {prop_data['details']['location']}")
            print(f"Price: {prop_data['details']['price']}")
            print(f"Size: {prop_data['details']['size']}")
            print(f"Bedrooms: {prop_data['details']['bedrooms']}")
            print(f"Bathrooms: {prop_data['details']['bathrooms']}")
            print("\nCategory Ratings (1-5):")
            for cat, score in prop_data['category_scores'].items():
                print(f"- {cat.title()}: {score}/5")
            print(f"\nOVERALL RATING: {prop_data['rating']}/5")
            print("=" * 50)

            top_recommendations.append({
                'id': prop_id,
                **prop_data['details'],
                'rating': prop_data['rating'],
                'category_ratings': prop_data['category_scores']
            })

        return top_recommendations

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []
if __name__ == "__main__":
    recommendations = get_similar_listings(preference_answers)