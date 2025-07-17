import os
import pandas as pd
from preprocess.utils.cui_mappings import CHEXPERT_CUI_MAP

def load_chexpert(split):
    file_path = os.path.join("datasets", "CheXpert", f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CheXpert CSV not found at {file_path}")

    df = pd.read_csv(file_path, quoting=3, escapechar="\\")

    # Drop rows with missing or empty reports
    df = df[df["Report"].notna() & df["Report"].str.strip().ne("")]
    
    def extract_cuis(row):
        cuis = []
        for label, cui in CHEXPERT_CUI_MAP.items():
            value = row.get(label, 0)
            if value in [1, -1]:  # positive or uncertain
                cuis.append(cui)
        return list(set(cuis))

    def resolve_image_path(row):
        return os.path.join("datasets", "CheXpert", row["Path"])

    return pd.DataFrame({
        "image_path": df.apply(resolve_image_path, axis=1),
        "concepts": df.apply(extract_cuis, axis=1),
        "report": df["Report"].tolist(),
        "source": "chexpert"
    })
