# Real Estate Agent System

An AI-powered real estate recommendation system that generates, personalizes, and matches property listings based on user preferences.

## Features

- Property listing generation using LLM
- Vector similarity search
- Personalized property descriptions
- Preference-based matching
- Rating system for properties

## Prerequisites

- Python 3.10+
- OpenAI API key
- CUDA-capable GPU (optional, for faster embeddings)

## Installation

1. Clone the repository:

git clone git@github.com:nmohanan31/Realestateagent.git
cd RealestateAgent

2. Create and activate virtual environment:
conda create -n vector_env python=3.10
conda activate vector_env

3. Install required packages:
pip install -r requirements.txt


## Configuration

Set up OpenAI API credentials:
xport OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://openai.vocareum.com/v1"

## Project Structure

realestateAgent/
├── realestate_agent.py       # Main orchestration
├── generate_listings.py      # Listing generation
├── listings_vector_database.py # Vector DB operations
├── userinterface_and_search.py # Search interface
└── personalising_listings.py  # Personalized descriptions of Top recommendations 

## Usage

Run the main program:

    python realestate_agent.py

## Sample output for check

cat real_estate_output.txt

## Features Explained

- Listing Generation: Creates property listings using OpenAI's GPT model
- Vector Search: Implements similarity search using LanceDB
- Personalization: Customizes descriptions based on user preferences
- Rating System: Scores properties on a 1-10 scale across categories
- Sample Output

## The program generates:

- Property listings with details
- Similarity search results
- Personalized recommendations
- Category-based ratings
- Error Handling


## If you encounter OpenAI API errors:

- Verify API key is set correctly
- Check API base URL
- Dependencies
- openai
- lancedb
- sentence-transformers
- pandas
- numpy