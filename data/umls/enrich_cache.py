import os
import pandas as pd
import ast
from data.umls.uts_client import UMLSClient

CACHE_PATH = os.path.join("data", "umls", "cui_metadata_cache.csv")


def enrich_metadata_cache(api_key: str):
    """
    Iteratively enrich cui_metadata_cache.csv by adding parent CUIs
    not already present in the cache. Saves each new CUI incrementally.
    Stops when no new parents are discovered.
    """
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Metadata cache not found: {CACHE_PATH}")

    # Load cache as dictionary
    cache_df = pd.read_csv(CACHE_PATH, dtype=str).fillna("")
    cache = {}
    for _, row in cache_df.iterrows():
        cache[row["cui"]] = {
            "cui": row["cui"],
            "name": row["name"],
            "definition": row["definition"],
            "semantic_type": row["semantic_type"],
            "parents": ast.literal_eval(row["parents"]) if row["parents"] else [],
            "descendants": ast.literal_eval(row["descendants"]) if row["descendants"] else [],
        }

    client = UMLSClient(api_key)

    # Iteratively add parents until no new CUIs
    while True:
        current_cuis = set(cache.keys())
        new_cuis_to_add = set()

        # Collect parent CUIs missing from cache
        for cui, meta in cache.items():
            for parent in meta["parents"]:
                parent_cui = parent.split(", ")[-1].strip("}")
                if parent_cui not in current_cuis and parent_cui != "N/A":
                    new_cuis_to_add.add(parent_cui)

        if not new_cuis_to_add:
            print("[✓] No new parent CUIs found. Cache enrichment complete.")
            break

        print(f"[→] Found {len(new_cuis_to_add)} new parent CUIs. Fetching metadata...")

        # Fetch and save each new CUI incrementally
        for parent_cui in sorted(new_cuis_to_add):
            if parent_cui in cache:
                continue
            try:
                meta = client.get_concept_metadata(parent_cui)
                cache[parent_cui] = meta

                # Save immediately
                new_row = {
                    "cui": parent_cui,
                    "name": meta["name"],
                    "definition": meta["definition"],
                    "semantic_type": meta["semantic_type"],
                    "parents": str(meta["parents"]),
                    "descendants": str(meta["descendants"]),
                }
                pd.DataFrame([new_row]).to_csv(
                    CACHE_PATH, mode="a", header=False, index=False
                )

            except Exception as e:
                print(f"[!] Failed to fetch metadata for {parent_cui}: {e}")