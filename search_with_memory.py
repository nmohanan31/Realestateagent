import os
import csv
import logging
import re

# Environment variables
os.environ["OPENAI_API_KEY"] = "voc-12568918961266773670401673e1acb29caf9.60086405"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List

import lancedb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "gpt-3.5-turbo"
temperature = 0.0
llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=1000)

# Fix preference_questions as multiple items in a list
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

max_rating = 10

def initialize_database():
    """Initialize database connection and create table if needed"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, 'lance_db')
        os.makedirs(db_path, exist_ok=True)
        db = lancedb.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        return db
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def format_listing(listing: Dict) -> str:
    """Format listing dictionary as readable string"""
    return f"""
    Location: {listing.get('location', 'N/A')}
    Price: {listing.get('price', 'N/A')}
    Size: {listing.get('size', 'N/A')}
    Bedrooms: {listing.get('bedrooms', 'N/A')}
    Description: {listing.get('description', 'N/A')}
    """
def get_no_of_listings():
    """Get all available listings from LanceDB."""
    try:
        db = initialize_database()
        if 'listings' not in db.table_names():
            logger.error("Listings table not found")
            raise FileNotFoundError("Please run listings_vector_database.py first")

        table = db.open_table("listings")
        results = table.to_pandas().to_dict('records')
        logger.info(f"Found {len(results)} listings")
        return results
    except Exception as e:
        logger.error(f"Error retrieving listings: {str(e)}")
        raise


def format_property_rating(property_details: str, response: str, rating: int) -> str:
    """Format property details with single consistent rating"""
    return f"""
    {property_details}
    Analysis: {response}
    """

def extract_rating(response: str) -> int:
    """Extract numerical rating from response"""
    if match := re.search(r'Final Rating:\s*(\d+)/10', response):
        return min(int(match.group(1)), 10)
    return 0

def user_interface(preference_questions, preference_answers):
    rating_template = """
    Based on these user preferences:
    {preferences}
    
    Rate this property:
    {property_details}
    
    Provide a detailed analysis and end with exactly one rating out of 10.
    Format: "Final Rating: X/10"
    """
    
    rating_prompt = PromptTemplate(
        input_variables=["preferences", "property_details"],
        template=rating_template
    )
    
    rating_chain = LLMChain(
        llm=llm,
        prompt=rating_prompt,
        verbose=True
    )
    
    listings = get_no_of_listings()
    recommendations = []
    
    preferences = "\n".join([
        f"Q: {q}\nA: {a}" 
        for q, a in zip(preference_questions, preference_answers)
    ])
    
    # Process each listing
    for listing in listings:
        try:
            property_details = format_listing(listing)
            response = rating_chain.invoke({
                "preferences": preferences,
                "property_details": property_details
            })
            
            # Extract response text and rating
            analysis = response['text'] if isinstance(response, dict) else str(response)
            rating = extract_rating(analysis)
            
            # Store as dictionary
            recommendations.append({
                "listing": listing,
                "details": property_details,
                "analysis": analysis,
                "rating": rating
            })
            
        except Exception as e:
            logger.error(f"Error processing listing: {e}")
            continue
    
    # Display properties
    print("\n=== Properties and Their Ratings ===\n")
    for i, rec in enumerate(recommendations, 1):
        print("="*80)
        print(f"Property {i}:")
        print(rec["details"])
        print(f"Rating: {rec['rating']}/10")
        print("="*80 + "\n")
    
    # Get final recommendation for top properties
    if recommendations:
        # Sort recommendations
        recommendations.sort(key=lambda x: x["rating"], reverse=True)
        top_listing = recommendations[0]
        
        # Create final recommendation template
        final_template = """Based on these rated properties, recommend the best match:
        {recommendations}
        
        Explain why this is the best match for the user's preferences.
        """
        
        # Setup final recommendation chain
        final_prompt = PromptTemplate(
            input_variables=["recommendations"],
            template=final_template
        )
        
        final_chain = LLMChain(llm=llm, prompt=final_prompt)
        
        # Generate final recommendation
        properties_summary = "\n\n".join([
            f"Property: {rec['details']}\nRating: {rec['rating']}/10"
            for rec in recommendations[:3]  # Top 3 properties
        ])
        
        final_recommendation = final_chain.invoke({
            "recommendations": properties_summary
        })
        
        # Format final output with top 3 properties
        final_output = "=== Top 3 Recommended Properties ===\n\n"
        
        for i, listing in enumerate(recommendations[:3], 1):
            final_output += f"""
            {'-'*80}
            Recommendation #{i}:
            {listing['details']}

            Analysis:
            {listing['analysis']}
            {'-'*80}
            """
        
        return final_output
        
    return "No suitable properties found"


if __name__ == "__main__":
    recommendation = user_interface(preference_questions, preference_answers)
    print(recommendation)