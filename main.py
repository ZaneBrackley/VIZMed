import os, ast
import pandas as pd

from VIZMed.preprocess.padchest_parser import load_padchest
from VIZMed.preprocess.chexpert_parser import load_chexpert
from VIZMed.data.cui_vocab import build_cui_vocab

CACHE_DIR = os.path.join("VIZMed", "cache")

def load_or_preprocess(dataset_name: str, split: str = None) -> pd.DataFrame:
    """Load from CSV cache or preprocess and cache."""
    suffix = f"{dataset_name}_{split}" if split else dataset_name
    cache_path = os.path.join(CACHE_DIR, f"{suffix}-meta.csv")

    if os.path.exists(cache_path):
        print(f"[✓] Found cached {suffix} CSV.")
        df = pd.read_csv(cache_path)
        # Convert 'concepts' back to list
        df["concepts"] = df["concepts"].apply(ast.literal_eval)
        return df

    print(f"[→] Preprocessing {suffix}...")

    os.makedirs(CACHE_DIR, exist_ok=True)

    if dataset_name == "padchest":
        df = load_padchest()
    elif dataset_name == "chexpert":
        if split not in ["train", "valid"]:
            raise ValueError("CheXpert split must be 'train' or 'valid'")
        df = load_chexpert(split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Save to CSV with list as string
    df.to_csv(cache_path, index=False)
    print(f"[✓] Saved {suffix} CSV to {cache_path}")
    return df


if __name__ == "__main__":
    # Load or preprocess PadChest
    padchest_df = load_or_preprocess("padchest")

    # Load or preprocess CheXpert train + valid
    chexpert_train_df = load_or_preprocess("chexpert", split="train")
    chexpert_valid_df = load_or_preprocess("chexpert", split="valid")

    print("\n=== Dataset Summary ===")
    print(f"PadChest samples:      {len(padchest_df)}")
    print(f"CheXpert (train):      {len(chexpert_train_df)}")
    print(f"CheXpert (valid):      {len(chexpert_valid_df)}")
    print(f"Combined total:        {len(padchest_df) + len(chexpert_train_df) + len(chexpert_valid_df)}")

    chexpert_df = pd.concat([chexpert_train_df, chexpert_valid_df])

    padchest_vocab = build_cui_vocab(padchest_df, source="padchest")
    chexpert_vocab = build_cui_vocab(chexpert_df, source="chexpert")