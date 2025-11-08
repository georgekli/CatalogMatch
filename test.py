import json

import pandas as pd
import sys
import os

# Import the functions/classes from the files you provided
try:
    import data_cleaning_feature_extraction
    from recommender import ProductRecommender
except ImportError:
    print("Error: Make sure recommender.py and data_cleaning_feature_extraction.py are in the same directory.")
    sys.exit(1)

# File paths (These files are required)
CATALOG_FILE = "data/b_product_catalog.csv"
DESCRIPTIONS_FILE = "data/b_unstructured_descriptions.csv"

def run_analysis():
    """
    Runs the full evaluation pipeline using the specified functions.
    """
    print("Starting analysis...")
    with open("data/testset.json", "r") as file:
        test_set = json.load(file)
    # --- 1. Load Data ---
    # Check if data directory and files exist
    if not os.path.exists("data"):
        print("Error: 'data/' directory not found.")
        print("Please create a 'data/' directory and add your CSV files.")
        return

    try:
        descriptions_df = pd.read_csv(DESCRIPTIONS_FILE)
        # Normalize columns for lookup
        descriptions_df.columns = descriptions_df.columns.str.lower()
        descriptions_df = descriptions_df.set_index("description_id")
    except FileNotFoundError:
        print(f"Error: Required file not found: {DESCRIPTIONS_FILE}")
        return
    except KeyError:
        print(f"Error: 'description_id' column not found in {DESCRIPTIONS_FILE}.")
        return

    # --- 2. Initialize Recommender ---
    # This loads the catalog and builds the TF-IDF matrix
    try:
        recommender = ProductRecommender(CATALOG_FILE)
        if recommender.catalog_df.empty:
            print("Recommender failed to initialize. Check catalog file.")
            return
    except FileNotFoundError:
        print(f"Error: Required file not found: {CATALOG_FILE}")
        return

    print("Recommender initialized. Starting evaluation loop...")

    # --- 3. Run Evaluation Loop ---
    results_log = []
    total_precision_compares = 0
    for desc_id, expected_sku in test_set.items():
        try:
            # Get the raw description text
            raw_text = descriptions_df.loc[desc_id]['unstructured_description']
        except KeyError:
            print(f"Warning: Description_ID '{desc_id}' not found in {DESCRIPTIONS_FILE}. Skipping.")
            continue

        # a) Use clean_data_extract_features to process the query
        # This function returns a 1-row DataFrame
        clean_series = data_cleaning_feature_extraction.clean_data_extract_features(raw_text)

        # b) Use recommend to get top 3 matches
        # The 'recommend' function returns 3 sampled results by default
        recommendations_df = recommender.recommend(clean_series, top_n=10)

        # c) Log results
        if recommendations_df.empty:
            top_1_sku = None
            top_3_skus = []
        else:
            top_3_skus = recommendations_df['SKU'].tolist()
            top_1_sku = top_3_skus[0] if len(top_3_skus) > 0 else None

        is_top_1_correct = (top_1_sku in expected_sku)
        if expected_sku.__class__ != str:
            is_top_3_correct = any(item in top_3_skus for item in expected_sku)
        else:
            is_top_3_correct = (expected_sku in top_3_skus)

        # Compute how many items to compare — whichever list is shorter
        n_compare = min(len(expected_sku), len(top_3_skus))
        total_precision_compares += n_compare
        # Slice the predictions to that size
        precision_matches = sum(item in expected_sku for item in top_3_skus[:n_compare])

        results_log.append({
            "desc_id": desc_id,
            "expected_sku": expected_sku,
            "top_1_sku": top_1_sku,
            "is_top_1_correct": is_top_1_correct,
            "is_top_3_correct": is_top_3_correct,
            "precision_matches": precision_matches
        })

    print("...Evaluation loop complete.")

    # --- 4. Calculate and Print Metrics ---
    if not results_log:
        print("No results were generated. Check your TEST_SET and data files.")
        return

    results_df = pd.DataFrame(results_log)
    total_queries = len(results_df)

    top_1_hits = results_df['is_top_1_correct'].sum()
    top_3_hits = results_df['is_top_3_correct'].sum()

    top_1_accuracy = top_1_hits / total_queries
    top_3_accuracy = top_3_hits / total_queries  # (Recall@3)

    # Precision@3 = (Total relevant items found in top 3) / (Total items recommended)
    precision_matches_at_3 = results_df['precision_matches'].sum()
    precision_at_3 = precision_matches_at_3 / total_precision_compares


    print("\n--- Performance Metrics ---")
    print(f"Total Queries:         {total_queries}")
    print(f"Top-1 Accuracy:        {top_1_hits} / {total_queries} = {top_1_accuracy:.1%}")
    print(f"Top-3 Accuracy (Recall@3): {top_3_hits} / {total_queries} = {top_3_accuracy:.1%}")
    print(f"Precision@3:           {precision_matches_at_3} / {total_precision_compares} = {precision_at_3:.1%}")
    print("---------------------------\n")

    # --- 5. Analyze Results (Error Analysis) ---
    print("--- Error Analysis (Where the system struggles) ---")
    failed_matches = results_df[~results_df['is_top_3_correct']]

    if failed_matches.empty:
        print("No failed matches! All items in the test set were found in the Top 3.")
    else:
        print(f"Failed to find the correct SKU in the Top 3 for {len(failed_matches)} queries:")
        for _, row in failed_matches.iterrows():
            print(f"  - ID: {row['desc_id']}, Expected: {row['expected_sku']}, Got (Top 1): {row['top_1_sku']}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_analysis()