import logging
from generate_listings import generate_listings
from listings_vector_database import load_and_prepare_data, create_embeddings, store_in_lancedb
from userinterface_and_search import similarity_search_store, get_listings, preference_answers
from personalising_listings import personalize_property_descriptions, retrieve_top_recommendations, display_augmented_recommendations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    # Generate listings and print them
    listings = generate_listings()
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
    top_recommendations = similarity_search_store(properties)

    # Retrieve top recommendations from the database
    recommendations = retrieve_top_recommendations()
    # Generate personalized descriptions for the recommendations
    augmented_recommendations = personalize_property_descriptions(recommendations)
    # Display the augmented recommendations
    display_augmented_recommendations(augmented_recommendations)

if __name__ == "__main__":
    main()