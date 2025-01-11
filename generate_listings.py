import os
import csv

os.environ["OPENAI_API_KEY"] = "voc-12568918961266773670401673e1acb29caf9.60086405"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Initialize chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Example template
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

def generate_listings(num_listings=10):
    example_prompt = PromptTemplate(
        input_variables=["location", "price", "bedrooms", "bathrooms", "size", "description", "neighborhood"],
        template=example_template
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Generate Berlin real estate listings which are unique and different from each other with the following format:",
        suffix="New Listing:\n\n{input}",
        input_variables=["input"],
        example_separator="\n\n"
    )

    listings = []
    for _ in range(num_listings):
        prompt = few_shot_prompt.format(input="")
        response = chat.predict(prompt)
        listings.append(response)
       
        # Convert listings to structured format
        structured_listings = []
        for listing in listings:
            lines = listing.strip().split('\n')
            listing_dict = {}
            current_field = None
            
            for line in lines:
                if line.strip():
                    if ':' in line:
                        field, value = line.split(':', 1)
                        current_field = field.strip()
                        listing_dict[current_field] = value.strip()
                    elif current_field:
                        listing_dict[current_field] = listing_dict.get(current_field, '') + ' ' + line.strip()
            
            structured_listings.append(listing_dict)

        # Write to CSV
        with open('berlin_listings.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Location', 'Price', 'Bedrooms', 'Bathrooms', 'Size', 'Description', 'Neighborhood'])
            writer.writeheader()
            writer.writerows(structured_listings)
    return listings

if __name__ == "__main__":
    listings = generate_listings()
    for i, listing in enumerate(listings, 1):
        print(f"\nListing {i}:")
        print(listing)
        print("-" * 50)