Here is a complete `README.md` file for your project, based on the provided code, data, and assignment documentation.

-----
Product Catalog Matching System
-----

## Overview

This project solves a critical data-matching bottleneck for a retail company. [cite\_start]Currently, sales teams and customers use unstructured, natural-language descriptions (e.g., "Need a blak polo shirt, size M"), while the inventory system operates on structured SKUs (e.g., `SKU1000172`, `Essential Shirt`, `Black`, `S`)[cite: 89, 90].

[cite\_start]The existing manual matching process is slow (2-3 hours/day) and causes costly fulfillment errors and lost sales [cite: 92-95].

This system automates the process by:

1.  [cite\_start]**Extracting** structured attributes (like color, size, brand) from the messy text[cite: 109].
2.  [cite\_start]**Matching** these attributes against the 800-SKU product catalog[cite: 104].
3.  [cite\_start]**Ranking** and returning the Top 3 most likely SKU matches with a confidence score[cite: 107].

## Key Features

  * **Intelligent Attribute Extraction:** Goes beyond simple keyword matching.
      * **Fuzzy Matching:** Handles misspellings (e.g., "bevge" -\> "beige", "Classzc" -\> "classic") using `rapidfuzz`.
      * **Synonym Normalization:** Maps common variations to catalog standards (e.g., "extra large" -\> "xl", "medium" -\> "m").
      * **Advanced Color Correction:** Uses `webcolors` and RGB Euclidean distance to find the perceptually closest catalog color, even for non-standard names (e.g., "crimson" -\> "red", "maroon" -\> "brown").
  * **Hybrid Matching Algorithm:** A robust two-stage "filter-then-rank" model.
    1.  **Strict Filtering:** Applies a hard filter on non-negotiable attributes (like `Size`) to create a small, relevant candidate pool.
    2.  **TF-IDF Ranking:** Ranks the remaining candidates by text similarity (`cosine_similarity`) against a master `feature_string` built from all catalog text attributes (Category, Brand, Color, Material, etc.).
  * **End-to-End Pipeline:** Includes scripts to process raw descriptions, generate recommendations for all items, and validate performance.

## Performance

The system was validated against a 25-item test set (`data/testset.json`) with known ground-truth answers. The results demonstrate that the system effectively solves the business problem.

```
--- Performance Metrics ---
Total Queries:           25
Top-1 Accuracy:        24 / 25 = 96.0%
Top-3 Accuracy (Recall@3): 25 / 25 = 100.0%
Precision@3:           33 / 34 = 97.1%
---------------------------
```

[cite\_start]The **100% Top-3 Accuracy (Recall@3)** is the key business metric[cite: 133]. [cite\_start]It means that for 100% of queries, the correct item is presented to the user in the top 3 suggestions, completely eliminating the need for manual search and solving the core pain point[cite: 92].

## Project Structure

```
/
|-- data_cleaning_feature_extraction.py  # Python module for NLP cleaning and attribute extraction
|-- recommender.py                     # Python module with the ProductRecommender class
|-- recommend_on_every_desc.py         # Main script to run batch recommendations on all data
|-- test.py                            # Main script to run validation and print performance
|-- requirements.txt                   # Project dependencies
|
|-- data/
|   |-- b_product_catalog.csv            # INPUT: The 800-SKU master product catalog
|   |-- b_unstructured_descriptions.csv  # INPUT: The ~250 unstructured queries
|   |-- testset.json                     # INPUT: Ground-truth validation set
|   |
|   |-- b_structured_descriptions.csv    # OUTPUT: Unstructured text after attribute extraction
|   |-- description_recommendations.csv  # OUTPUT: Final Top-3 SKU recommendations
|
|-- Assignment_.docx                   # The original assignment brief
|-- README.md                          # This file
```

-----

## Setup & Installation

This project is written in Python 3.

1.  **Clone the repository:**

    ```sh
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment (Recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies from `requirements.txt`:**

    ```sh
    pip install -r requirements.txt
    ```

    *If `requirements.txt` is missing, here are the likely contents based on imports:*

    ```
    pandas
    rapidfuzz
    webcolors
    numpy
    nltk
    scikit-learn
    ```

4.  **Download NLTK data:**
    The cleaning script requires the NLTK `stopwords` package. You must run this command once to download them.

    ```sh
    python -c "import nltk; nltk.download('stopwords')"
    ```

-----

## How to Run

The system provides several entry points for processing, testing, and validation. All outputs are saved to the `/data/` directory.

### 1\. Run Validation & Performance Test (Recommended First Step)

This script runs the full end-to-end model pipeline against the `data/testset.json` validation file. It will print the final performance metrics (Top-1, Top-3, Precision) to the console.

```sh
python test.py
```

### 2\. Run Batch Recommendations

This script reads the pre-processed `data/b_structured_descriptions.csv`, runs the recommender for every single entry, and saves the final Top-3 SKU recommendations to `data/description_recommendations.csv`.

```sh
python recommend_on_every_desc.py
```

### 3\. Test a Single, Ad-Hoc Query

You can test any custom query directly from your command line using `recommender.py`. The script will process your string and print the Top-3 recommendations.

```sh
# Usage: python recommender.py "your query string"

# --- EXAMPLES ---
python recommender.py "Need blue sundress that is breathable and flexible for Fall 2024, size M."

python recommender.py "Looking for bevge Snakers by Classzc, size S please."

python recommender.py "blak shirth esesntial size s"
```

-----

## Methodology

The matching algorithm is a two-stage hybrid model:

1.  **Attribute Extraction (`data_cleaning_feature_extraction.py`):**

      * The raw query is cleaned (lowercase, punctuation removed, stopwords removed).
      * Fuzzy matching (`rapidfuzz`) and synonym maps are used to find and normalize `Size`, `Brand`, `Category`, `Material`, and `Season`.
      * A special color-matching function finds the closest catalog color using RGB color-space distance.
      * This produces a structured, machine-readable version of the query.

2.  **Hybrid Matching (`recommender.py`):**

      * **Strict Filtering:** The system first applies a hard filter using the most important attribute: `Size`. If the query asks for "size L", all non-L items are immediately discarded. This drastically narrows the search space and improves accuracy.
      * **TF-IDF Ranking:** The `ProductRecommender` class pre-builds a TF-IDF matrix from a `feature_string` (a concatenation of all text fields) for every item in the catalog. The user's query (minus the size) is vectorized, and its `cosine_similarity` is calculated against the remaining candidates.
      * The Top 3 items by similarity score are returned as the final recommendations.

## Known Limitations & Troubleshooting

  * [cite\_start]**Catalog Dependency:** The model's accuracy is highly dependent on the quality of the `data/b_product_catalog.csv`[cite: 104]. If an item is missing its `Brand`, `Material`, or `Features` in the catalog, it is less likely to be matched by the text ranker.
  * **Strict Size Filter:** The `Size` filter is deliberately strict. If a user enters the wrong size (e.g., "S" when they meant "M"), the system will *not* find the "M" item. This is a business logic trade-off, prioritizing accuracy for a correct query over "guessing" a different size.
  * **Out-of-Vocabulary (OOV) Attributes:** If a query contains a new brand or color not present anywhere in the catalog (e.g., a query for "Nike" when the brand is not in the catalog), the model will not be able to extract it. It will, however, still attempt to match based on the other attributes (e.g., "sneakers", "blue").
  * **Troubleshooting `FileNotFoundError`:** All scripts assume they are being run from the root directory of the project. If you get an error that `data/b_product_catalog.csv` cannot be found, ensure you are running the python commands from the main project folder.