import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate



def generate_listings_llm(example_template, examples ,num_listings=10):

    # Initialize chat model with specified parameters
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

    # Create a prompt template for the examples
    example_prompt = PromptTemplate(
        input_variables=["location", "price", "bedrooms", "bathrooms", "size", "description", "neighborhood"],
        template=example_template
    )

    # Create a few-shot prompt template using the examples
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""Generate real estate listings for different locations in Berlin, ensuring:
        1. At least 5 different Berlin neighborhoods (including but not limited to: Mitte, Prenzlauer Berg, Kreuzberg, Charlottenburg, Friedrichshain)
        2. EXACTLY 3 properties must have 3 bedrooms
        3. Each listing must be COMPLETELY UNIQUE in terms of:
           - Location
           - Price
           - Description
           - Number of rooms
           - Neighborhood details
        4. NO duplicate listings or similar descriptions
        
        Use the following format:\n\n""",
        suffix="New Listing:\n\n{input}",
        input_variables=["input"],
        example_separator="\n\n"
    )

    listings = []
    for _ in range(num_listings):
        # Format the prompt for generating a new listing
        prompt = few_shot_prompt.format(input="")
        # Get the response from the chat model
        response = chat.predict(prompt)
        listings.append(response)
    
    # Convert listings to structured format
    structured_listings = []
    for i, listing in enumerate(listings):
        lines = listing.strip().split('\n')
        listing_dict = {}
        current_field = None
        
        # Add ID field
        listing_dict['ID'] = f'BER{str(i+1).zfill(4)}'
        
        for line in lines:
            if line.strip():
                if ':' in line:
                    field, value = line.split(':', 1)
                    current_field = field.strip()
                    listing_dict[current_field] = value.strip()
            elif current_field:
                listing_dict[current_field] = listing_dict.get(current_field, '') + ' ' + line.strip()
        
        structured_listings.append(listing_dict)

    # Write the structured listings to a JSON file
    with open('berlin_listings.json', 'w', encoding='utf-8') as f:
        json.dump(structured_listings, f, indent=4, ensure_ascii=False)
    
    return listings
