import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import data_cleaning_feature_extraction

class ProductRecommender:
    """
    A hybrid recommender that uses strict filtering for exact-match attributes
    and TF-IDF cosine similarity for text-based attributes.
    """

    def __init__(self, catalog_filepath):
        """
        Initializes the recommender by loading and processing the catalog
        from the given file path.
        """
        self.catalog_df = self._preprocess_catalog(catalog_filepath)
        # Define which columns are for strict filtering vs. text ranking
        self.strict_filter_cols = ['Size']
        # The "embedding" part: create the TF-IDF matrix
        self.vectorizer = TfidfVectorizer(lowercase=True, analyzer='word')
        # Create and fit the vectorizer on our "feature string"
        self.tfidf_matrix = self._fit_vectorizer()

    def _preprocess_catalog(self, catalog_filepath):
        """
        Loads and preprocesses the catalog data from a CSV file path.
        """
        # Load the data
        try:
            # --- MODIFICATION: Read directly from file path ---
            df = pd.read_csv(catalog_filepath)
            print(f"Successfully loaded catalog from '{catalog_filepath}'")
        except FileNotFoundError:
            print(f"Error: Catalog file not found at '{catalog_filepath}'")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return pd.DataFrame()
        # Exclude Price
        if 'Price' in df.columns:
            df = df.drop(columns=['Price'])
        # Combine Category and Subcategory (as requested)
        df['Category_Global'] = df['Category'].fillna('') + ' ' + df['Subcategory'].fillna('')
        # Process the multi-value 'Features' column
        # Replace '|' with a space
        df['Features_Proc'] = df['Features'].fillna('').str.replace('|', ' ', regex=False)
        # Create the master "feature string" for each product
        df['feature_string'] = (df['Category_Global'].fillna('') + ' ' + df['Features_Proc'].fillna('') + ' ' +
                                df['Material'].fillna('') + ' ' + df['Color'].fillna('') + ' ' + df['Season'].fillna('')
                                + ' ' + df['Brand'].fillna(''))
        print("--- Catalog Preprocessing Complete ---")
        return df

    def _fit_vectorizer(self):
        """
        Fits the TF-IDF vectorizer to the 'feature_string' column.
        This creates the "feature space" you asked about.
        """
        if 'feature_string' in self.catalog_df.columns and not self.catalog_df.empty:
            print("Fitting TF-IDF vectorizer (creating feature space)...")
            return self.vectorizer.fit_transform(self.catalog_df['feature_string'])
        else:
            print("Error: 'feature_string' not found or catalog is empty. TF-IDF fitting failed.")
            return None

    def recommend(self, query_series, top_n=10):
        """
        Recommends products based on a query dictionary.

        :param query_features: A dict of features, e.g.,
                             {'Size': 'M', 'Category': 'Coat', 'Features': 'Quick-dry'}
        :param top_n: The maximum number of recommendations from which to sample the last 3 to return.
        :return: A pandas DataFrame of matching products, ranked by score.
        """
        query_features = extract_features(query_series)
        # Hard Filtering
        filtered_df = self.catalog_df.copy()
        strict_features_queried = []
        for col in self.strict_filter_cols:
            if col in query_features and query_features[col] is not None:
                query_val = str(query_features[col]).lower()
                strict_features_queried.append(col)
                # Apply the strict filter
                filtered_df = filtered_df[filtered_df[col].str.lower() == query_val]

        print(f"Applied strict filters on: {strict_features_queried}")

        if filtered_df.empty:
            print("No products matched the strict criteria.")
            return pd.DataFrame(columns=self.catalog_df.columns.tolist() + ['match_score'])
        # Build Text Query for Ranking
        text_query_parts = []
        if 'Category' in query_features and query_features['Category'] is not None:
            text_query_parts.append(str(query_features['Category']))
        if 'Features' in query_features and query_features['Features'] is not None:
            text_query_parts.append(
                str(query_features['Features']).replace('|', ' ')
            )
        if 'Material' in query_features and query_features['Material'] is not None:
            text_query_parts.append(str(query_features['Material']))
        if 'Brand' in query_features and query_features['Brand'] is not None:
            text_query_parts.append(str(query_features['Brand']))
        if 'Color' in query_features and query_features['Color'] is not None:
            text_query_parts.append(str(query_features['Color']))
        if 'Season' in query_features and query_features['Season'] is not None:
            text_query_parts.append(str(query_features['Season']))
        query_string = ' '.join(text_query_parts).lower().strip()
        # Rank or Return
        if not query_string:
            print("No text query provided. Returning all hard-filtered results.")
            filtered_df['match_score'] = 1.0
            return filtered_df.head(top_n).sample(3)
        print(f"Ranking remaining products based on text: '{query_string}'")
        # TF-IDF Ranking (The "Embedding" Match)
        filtered_indices = filtered_df.index.tolist()
        filtered_tfidf_matrix = self.tfidf_matrix[filtered_indices]
        query_vector = self.vectorizer.transform([query_string])
        cosine_sim = cosine_similarity(query_vector, filtered_tfidf_matrix)
        scores = cosine_sim[0]
        ranked_df = filtered_df.copy()
        ranked_df['match_score'] = scores
        final_results = ranked_df.sort_values(by='match_score', ascending=False)
        # try:
        #     recs = pd.concat([final_results.head(1), final_results.iloc[1:top_n].sample(2)], axis=0)
        # except IndexError:
        recs = final_results.head(3)
        recommendations = recs
        recommendations.loc[:, 'match_score'] = recommendations['match_score'] * 100
        recommendations = recommendations.set_index("SKU")
        recommendations = recommendations.sort_values(by="match_score", ascending=False)
        recommendations = recommendations.drop(['Category_Global', 'Features_Proc', 'feature_string'], axis=1)
        for series in recommendations.iterrows():
            print(series[0], ': ', series[1]['Product_Name'])
        print("--- Product Recommendations Complete ---")
        return recs

def extract_features(query_series):
    features_dict = {}
    product_features = ["Category", "Color", "Size", "Brand", "Material", "Features", "Season"]
    series_features = ["category", "color", "size", "brand", "material", "features", "season"]
    for i, feature in enumerate(series_features):
        value = query_series.get(feature)
        if pd.notna(value[0]) and value[0] != "":
            features_dict[product_features[i]] = value[0]
    return features_dict


def main():
    # Check if a text argument was provided
    if len(sys.argv) < 2:
        print("Usage: python recommender.py <query>")
        return
    # Combine all command-line arguments into a single string
    input_text = " ".join(sys.argv[1:])
    # Initialize recommender
    recommender = ProductRecommender("data/b_product_catalog.csv")
    # Get recommendations
    clean_series = data_cleaning_feature_extraction.clean_data_extract_features(input_text)
    recommender.recommend(clean_series)  # Assuming you have a 'recommend' method


if __name__ == "__main__":
    main()
