import pandas as pd
import sys

import data_cleaning_feature_extraction
from recommender import ProductRecommender


def main():
    """
    Main function to load descriptions, generate recommendations,
    and save them to a new CSV file.
    """

    # --- Configuration ---
    # Using the file paths provided by the user and inferred from recommender.py
    descriptions_filepath = "data/b_structured_descriptions.csv"
    # This path is based on the 'main' function in the provided 'recommender.py'
    catalog_filepath = "data/b_product_catalog.csv"
    output_filename = "data/description_recommendations.csv"

    try:
        # Load the descriptions to process
        descriptions_df = pd.read_csv(descriptions_filepath)
        print(f"Successfully loaded descriptions from '{descriptions_filepath}'")
    except FileNotFoundError:
        print(f"Error: Descriptions file not found at '{descriptions_filepath}'")
        return
    except Exception as e:
        print(f"Error loading '{descriptions_filepath}': {e}")
        return

    # Initialize the recommender
    # This will print its own success or error message
    try:
        recommender = ProductRecommender(catalog_filepath)
    except FileNotFoundError:
        print(f"Error: Product catalog file not found at '{catalog_filepath}'")
        print("Please ensure the catalog file is in the correct location.")
        return
    except Exception as e:
        print(f"Error initializing ProductRecommender: {e}")
        print("Please ensure 'recommender.py', 'data_cleaning_feature_extraction.py',")
        print(f"and the catalog '{catalog_filepath}' are accessible.")
        return

    # A list to store results for each description
    all_recommendations_data = []

    print(f"Processing {len(descriptions_df)} descriptions...")

    # Iterate through each row in the descriptions DataFrame
    for index, row in descriptions_df.iterrows():
        desc_id = row['description_id']

        clean_series = data_cleaning_feature_extraction.clean_data_extract_features(row['unstructured_description'])

        # The recommend function in recommender.py already sorts by score
        recs_df = recommender.recommend(clean_series, 3)

        # 3. Prepare the new row for the output DataFrame
        new_row = {"DESC_ID": desc_id}

        # 4. Populate REC_1, REC_2, REC_3
        rec_list = []
        if not recs_df.empty:
            for i, rec_row in recs_df.iterrows():
                try:
                    # Ensure SKU and match_score exist in the returned df
                    sku = rec_row['SKU']
                    score = rec_row['match_score']
                    rec_list.append((sku, score))
                except KeyError:
                    print(f"Warning: 'SKU' or 'match_score' not in recommendations DataFrame for {desc_id}")
                    rec_list.append((None, None))

        # Fill in the REC columns, padding with (None, None) if fewer than 3 recs
        for i in range(3):
            if i < len(rec_list):
                new_row[f"REC_{i + 1}"] = rec_list[i]
            else:
                new_row[f"REC_{i + 1}"] = (None, None)

        all_recommendations_data.append(new_row)

    # 5. Create the final DataFrame from our list of dictionaries
    final_df = pd.DataFrame(all_recommendations_data)

    # 6. Save the result to a new CSV
    try:
        final_df.to_csv(output_filename, index=False)
        print("\n--- Processing Complete ---")
        print(f"Recommendations saved to '{output_filename}'")
        print("\nFinal DataFrame Head:")
        print(final_df.head())
    except Exception as e:
        print(f"\nError saving final recommendations to '{output_filename}': {e}")


if __name__ == "__main__":
    main()