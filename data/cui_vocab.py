# data/cui_vocab.py
import os, ast
import pandas as pd
from collections import Counter
from data.umls.uts_client import UMLSClient
import json

def build_cui_vocab(
    df: pd.DataFrame,
    source: str,
    umls_lookup: dict = None,
    output_dir: str = "data/vocab",
    api_key: str = None
):
    os.makedirs(output_dir, exist_ok=True)

    # === Extract CUIs ===
    valid_cuis = []
    for row in df["concepts"]:
        if isinstance(row, list):
            valid_cuis.extend([c for c in row if isinstance(c, str) and c.strip()])

    unique_cuis = sorted(set(valid_cuis))
    freq = Counter(valid_cuis)

    # === Load cache file ===
    cache_path = os.path.join("data/umls", "cui_metadata_cache.csv")
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path, dtype=str).fillna("")
    else:
        cache_df = pd.DataFrame(columns=["cui", "name", "definition", "semantic_type", "parents", "descendants"])

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

    # === Fetch missing CUIs incrementally ===
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

                # --- Save immediately after each CUI ---
                new_row = {
                    "cui": cui,
                    "name": meta["name"],
                    "definition": meta["definition"],
                    "semantic_type": meta["semantic_type"],
                    "parents": str(meta["parents"]),
                    "descendants": str(meta["descendants"]),
                }
                pd.DataFrame([new_row]).to_csv(
                    cache_path, mode="a", header=not os.path.exists(cache_path), index=False
                )

            except Exception as e:
                print(f"[!] Failed to fetch {cui}: {e}")

            if (i + 1) % 10 == 0:
                print(f"  ...processed {i+1}/{len(unique_cuis)}")

    # === Build final vocab DataFrame (merge with cache) ===
    vocab_rows = []
    for cui in unique_cuis:
        meta = cache[cui]
        vocab_rows.append({
            "cui": cui,
            "name": meta["name"],
            "definition": meta["definition"],
            "semantic_type": meta["semantic_type"],
            "parents": str(meta["parents"]),
            "descendants": str(meta["descendants"]),
            "count": freq[cui],
            "source": source
        })

    vocab_df = pd.DataFrame(vocab_rows)

    # Save vocab CSV
    csv_path = os.path.join(output_dir, f"{source}_vocab.csv")
    vocab_df.to_csv(csv_path, index=False)

    print(f"[✓] Saved enriched vocab CSV to {csv_path}")

    return vocab_df