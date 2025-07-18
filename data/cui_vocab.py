import os
import pandas as pd
from collections import Counter
from data.umls.uts_client import UMLSClient

def build_cui_vocab(
    df: pd.DataFrame,
    source: str,
    umls_lookup: dict = None,
    output_dir: str = "data/vocab",
    api_key: str = None):
    """
    Build a UMLS-enriched CUI vocabulary CSV from a dataset DataFrame.
    Caches lookups in a persistent CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract valid CUIs from dataset
    valid_cuis = []
    for row in df["concepts"]:
        if isinstance(row, list):
            valid_cuis.extend([cui for cui in row if isinstance(cui, str) and cui.strip()])

    unique_cuis = sorted(set(valid_cuis))
    freq = Counter(valid_cuis)

    # Load or initialize cache
    cache_path = os.path.join("data/umls", "cui_metadata_cache.csv")
    cache_df = pd.read_csv(cache_path, dtype=str) if os.path.exists(cache_path) else pd.DataFrame(columns=[
        "cui", "name", "definition", "semantic_type", "parents", "descendants"
    ])
    cache = {row["cui"]: row for _, row in cache_df.iterrows()}

    # Fetch missing CUIs if needed
    new_rows = []
    if umls_lookup is None:
        if not api_key:
            raise ValueError("UMLS API key is required if no lookup is passed.")
        umls = UMLSClient(api_key)

        for i, cui in enumerate(unique_cuis):
            if cui in cache:
                continue
            try:
                meta = umls.get_concept_metadata(cui)
                cache[cui] = meta
                new_rows.append(meta)
            except Exception as e:
                print(f"[!] Failed to fetch {cui}: {e}")

            if i % 25 == 0 and i > 0:
                print(f"  ...processed {i}/{len(unique_cuis)}")

        # Append new rows to cache file
        if new_rows:
            pd.DataFrame(new_rows).to_csv(
                cache_path, mode="a", header=not os.path.exists(cache_path), index=False
            )

    # Build final vocab DataFrame
    vocab_df = pd.DataFrame({
        "cui": unique_cuis,
        "name": [cache[cui]["name"] for cui in unique_cuis],
        "definition": [cache[cui].get("definition", "") for cui in unique_cuis],
        "semantic_type": [cache[cui].get("semantic_type", "") for cui in unique_cuis],
        "parents": [str(cache[cui].get("parents", [])) for cui in unique_cuis],
        "descendants": [str(cache[cui].get("descendants", [])) for cui in unique_cuis],
        "count": [freq[cui] for cui in unique_cuis],
        "source": source
    })

    out_path = os.path.join(output_dir, f"{source}_vocab.csv")
    vocab_df.to_csv(out_path)
    print(f"[✓] Saved enriched vocab to {out_path}")
    return vocab_df