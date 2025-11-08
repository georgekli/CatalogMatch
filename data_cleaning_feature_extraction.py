"""
Data cleaning and normalization for unstructured product descriptions using NLTK.

Features:
- Lowercasing and punctuation removal
- Lemmatization and stemming with NLTK
- Color correction using webcolors + fuzzy matching
- Size and brand normalization using catalog lists
- Outputs cleaned CSV with color, size, brand columns
"""

import pandas as pd
import re
from rapidfuzz import process, fuzz
import webcolors
import numpy as np
import nltk
from nltk.corpus import stopwords
# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Exclude size tokens from stopwords
size_tokens = {"s", "m", "l", "xl", "xxl"}
stop_words = stop_words - size_tokens  # remove size tokens from stopwords

# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    """Lowercase, remove punctuation, normalize spaces and remove English stopwords."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    text = " ".join(tokens)
    return text


# =========================
# FUZZY MATCHING UTILITIES
# =========================

def fuzzy_correct(word, vocabulary, cutoff=80):
    """
    Fuzzy-match a word against a vocabulary with safeguards for short words.

    Parameters
    ----------
    word : str
        Input word
    vocabulary : list of str
        List of valid terms
    cutoff : int
        Minimum similarity threshold (0-100)

    Returns
    -------
    str
        Best match from vocabulary or None
    """
    word = word.lower().strip()
    if len(word) <= 2:
        # For very short words, match only exact matches
        return word if word in vocabulary else None

    match = process.extractOne(word, vocabulary, score_cutoff=cutoff)
    return match[0] if match else None


def rgb_distance(rgb1, rgb2):
    """Compute Euclidean distance between two RGB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def closest_catalog_color_from_token(token, catalog_colors):
    """
    Correct a color token using fuzzy matching with webcolors, then
    select the closest catalog color based on RGB distance.

    Parameters
    ----------
    token : str
        Input color token (may be misspelled)
    catalog_colors : list of str
        List of catalog colors (normalized lowercase)

    Returns
    -------
    str or None
        Best matching catalog color
    """
    if not token:
        return None

    token = token.lower().strip()

    # Step 1: Fuzzy-correct against CSS3 webcolors
    webcolors_vocab = list(webcolors._html5._CSS3_NAMES_TO_HEX.keys())
    corrected = process.extractOne(token, webcolors_vocab, scorer=fuzz.ratio, score_cutoff=85)
    corrected_name = corrected[0] if corrected else token

    # Step 2: Convert corrected name to RGB
    try:
        rgb_token = webcolors.name_to_rgb(corrected_name)
    except ValueError:
        return None

    # Step 3: Find closest catalog color by RGB distance
    best_color = None
    min_dist = float("inf")
    for c in catalog_colors:
        try:
            rgb_catalog = webcolors.name_to_rgb(c)
            d = rgb_distance(rgb_token, rgb_catalog)
            if d < min_dist:
                min_dist = d
                best_color = c
        except ValueError:
            continue

    return best_color


# =========================
# SIZE NORMALIZATION
# =========================

size_synonyms = {
    "extra large": "xl", "x large": "xl", "x-large": "xl",
    "extra-large": "xl", "etra large": "xl", "xx large": "xxl",
    "medium": "m", "small": "s", "large": "l"
}


def normalize_size(text, sizes_catalog, annotated_text):
    """
    Normalize sizes from text to catalog-approved sizes and annotate text.

    Parameters
    ----------
    text : str
        Cleaned description
    sizes_catalog : list of str
        Catalog-approved sizes
    annotated_text : str
        Current annotated text

    Returns
    -------
    (str or None, str)
        (Normalized size, updated annotated text)
    """
    tokens = text.split()
    # Step 1: Match size synonyms
    for t in tokens:
        corr = process.extractOne(t, size_synonyms.keys(), scorer=fuzz.ratio, score_cutoff=90)
        if corr is not None:
            annotated_text = annotate_token_in_text(annotated_text, t, "(size:" + corr[0] + ")")
            return corr[0], annotated_text

    # Step 2: Match catalog sizes by length
    for t in tokens:
        token_len = len(t)
        filtered_vocab = [word for word in sizes_catalog if len(word) == token_len]
        if filtered_vocab is None:
            return None, annotated_text
        corr = process.extractOne(t, filtered_vocab, scorer=fuzz.ratio, score_cutoff=90)
        if corr is not None:
            annotated_text = annotate_token_in_text(annotated_text, t, "(size:" + corr[0] + ")")
            return corr[0], annotated_text

    return None, annotated_text


# =========================
# ATTRIBUTE EXTRACTION
# =========================

def extract_single_attribute(text, catalog_list, annotation, annotated_text, thres=85):
    """
    Extract a single attribute (brand, category, material, season) using fuzzy matching.

    Parameters
    ----------
    text : str
        Cleaned description
    catalog_list : list of str
        Catalog values for this attribute
    annotation : str
        Attribute name to annotate
    annotated_text : str
        Current annotated description
    thres : int
        Minimum fuzzy match threshold

    Returns
    -------
    (str or None, str)
        (Best matching attribute, updated annotated text)
    """
    # Exact match first
    for b in catalog_list:
        if re.search(rf"\b{b}\b", text):
            annotated_text = annotate_token_in_text(annotated_text, b, f"({annotation}:{b})")
            return b, annotated_text

    # Fuzzy match on tokens
    tokens = text.split()
    for t in tokens:
        corr = process.extractOne(t, catalog_list, scorer=fuzz.ratio, score_cutoff=thres)
        if corr is not None:
            annotated_text = annotate_token_in_text(annotated_text, t, f"({annotation}:{corr[0]})")
            return corr[0], annotated_text

    return None, annotated_text


def extract_features(text, features_catalog, annotated_text, thres=90):
    """
    Extract multiple features from text using fuzzy matching and annotate.

    Parameters
    ----------
    text : str
        Cleaned description
    features_catalog : list of str
        Catalog features
    annotated_text : str
        Current annotated text
    thres : int
        Minimum fuzzy match threshold

    Returns
    -------
    (str or None, str)
        (Pipe-separated matched features, updated annotated text)
    """
    text_tmp = text
    matched = []

    # Exact matches
    for f in features_catalog:
        if re.search(rf"\b{re.escape(f.lower())}\b", text_tmp):
            matched.append(f)
            annotated_text = annotate_token_in_text(annotated_text, f, f"(feature:{f})")
            text_tmp = text_tmp.replace(f.lower(), "")

    # Fuzzy matches
    tokens = text_tmp.split()
    for t in tokens:
        corr = process.extractOne(t, features_catalog, scorer=fuzz.ratio, score_cutoff=thres)
        if corr is not None:
            annotated_text = annotate_token_in_text(annotated_text, t, f"(feature:{corr[0]})")
            matched.append(corr[0])

    return ("|".join(matched), annotated_text) if matched else (None, annotated_text)


def extract_attributes(row, colors_catalog, sizes_catalog, brands_catalog, categories_catalog,
                       materials_catalog, features_catalog, season_catalog):
    """
    Extract all product attributes from a row of unstructured description.

    Returns normalized attributes dictionary and annotated text.
    """
    text = row["clean_text"]
    annotated_text = text
    attrs = {"category": None, "color": None, "size": None, "brand": None,
             "material": None, "features": None, "season": None, "annotated_text": None}

    # Color
    for token in text.split():
        corr = process.extractOne(token, colors_catalog, scorer=fuzz.ratio, score_cutoff=80)
        if corr is not None and corr[0] in colors_catalog:
            attrs["color"] = corr[0]
            annotated_text = annotate_token_in_text(annotated_text, token, f"(color:{corr[0]})")
            break
    if not attrs["color"]:
        for token in text.split():
            best_color = closest_catalog_color_from_token(token, colors_catalog)
            if best_color in colors_catalog:
                attrs["color"] = best_color
                annotated_text = annotate_token_in_text(annotated_text, token, f"(color:{best_color})")
                break

    # Other attributes
    attrs["season"], annotated_text = extract_single_attribute(text, season_catalog, "season", annotated_text)
    attrs["brand"], annotated_text = extract_single_attribute(text, brands_catalog, "brand", annotated_text, 80)
    attrs["material"], annotated_text = extract_single_attribute(text, materials_catalog, "material", annotated_text)
    attrs["category"], annotated_text = extract_single_attribute(text, categories_catalog, "category", annotated_text)
    attrs["features"], annotated_text = extract_features(text, features_catalog, annotated_text)
    attrs["size"], annotated_text = normalize_size(text, sizes_catalog, annotated_text)

    attrs["annotated_text"] = annotated_text
    return attrs


# =========================
# ANNOTATION HELPER
# =========================

def annotate_token_in_text(text, token, annotation):
    """
    Annotate a token in the original text by appending (attribute:value).

    Parameters
    ----------
    text : str
        Original description
    token : str
        Token to annotate
    annotation : str
        Text to append, e.g., "(color:navy)"

    Returns
    -------
    str
        Annotated text
    """
    def replacer(match):
        return f"{match.group(0)}{annotation}"

    pattern = rf"\b{re.escape(token)}\b"
    annotated_text = re.sub(pattern, replacer, text, count=1, flags=re.IGNORECASE)
    return annotated_text


# =========================
# MAIN
# =========================

def clean_data_extract_features(query=''):
    """
    Load unstructured product descriptions and product catalog, clean text,
    and extract structured attributes with annotations.

    Steps:
    1. Load datasets:
        - Unstructured descriptions CSV (b_unstructured_descriptions.csv)
        - Product catalog CSV (b_product_catalog.csv)
    2. Normalize column names to lowercase.
    3. Extract canonical lists for:
        - Colors, sizes, brands, categories, subcategories, materials, seasons
        - Features (split by '|' and normalized)
    4. Clean unstructured text:
        - Lowercase
        - Remove punctuation
        - Remove English stopwords (excluding size tokens)
    5. Extract attributes and annotate text using `extract_attributes`.
        - Attributes: color, size, brand, category, material, features, season
        - Annotated text includes original terms with parentheses showing matched attributes
    6. Convert attribute dictionaries to DataFrame and merge with original dataset.
    7. Save cleaned and structured CSV to 'data/b_unstructured_descriptions_cleaned.csv'.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves the cleaned and annotated CSV file.
        Prints confirmation of completion.
    """
    if query == '':
        # Load datasets
        unstruct = pd.read_csv("data/b_unstructured_descriptions.csv")
    else:
        unstruct = pd.DataFrame([('DESC_NEW', query, 'Unknown')],
                                columns=['Description_ID', 'Unstructured_Description', 'Source_Channel'])
    catalog = pd.read_csv("data/b_product_catalog.csv")

    # Normalize column names
    catalog.columns = catalog.columns.str.lower()
    unstruct.columns = unstruct.columns.str.lower()

    # Extract canonical lists from catalog
    colors_catalog = catalog["color"].dropna().str.lower().str.strip().unique().tolist()
    sizes_catalog = catalog["size"].dropna().str.lower().str.strip().unique().tolist()
    brands_catalog = catalog["brand"].dropna().str.lower().str.strip().unique().tolist()
    categories_catalog = (catalog["subcategory"].dropna().str.lower().str.strip().unique().tolist()
                          + catalog["category"].dropna().str.lower().str.strip().unique().tolist())
    materials_catalog = catalog["material"].dropna().str.lower().str.strip().unique().tolist()
    season_catalog = catalog["season"].dropna().str.lower().str.strip().unique().tolist()

    features_raw = catalog["features"].dropna().str.split("|").tolist()
    features_catalog = list(set([f.lower().strip() for sublist in features_raw for f in sublist]))

    # Clean text
    unstruct["clean_text"] = unstruct["unstructured_description"].apply(clean_text)

    # Extract attributes and annotations
    normalized_attrs = unstruct.apply(
        lambda row: extract_attributes(
            row, colors_catalog, sizes_catalog, brands_catalog, categories_catalog,
            materials_catalog, features_catalog, season_catalog
        ),
        axis=1
    )

    # Convert extracted dicts to DataFrame and merge with original
    normalized_df = pd.DataFrame(list(normalized_attrs))
    unstruct_final = pd.concat([unstruct, normalized_df], axis=1)

    # Save cleaned CSV
    if query == '':
        unstruct_final.to_csv("data/b_structured_descriptions.csv", index=False)
        print("Done! Saved as data/b_structured_descriptions.csv")
        return None
    else:
        print("Original Text:\t", unstruct_final["unstructured_description"][0])
        print("Features Extracted:\t", unstruct_final['annotated_text'][0])
        return unstruct_final


if __name__ == "__main__":
    clean_data_extract_features("Need blue sundress that is breathable and flexible for Fall 2024, size M.")
