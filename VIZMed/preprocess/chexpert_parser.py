import os
import pandas as pd
from VIZMed.preprocess.utils.cui_mappings import CHEXPERT_CUI_MAP

def load_chexpert(split="train"):
    file_path = os.path.join("datasets", "CheXpert", f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CheXpert CSV not found at {file_path}")

    df = pd.read_csv(file_path)
    df = df.fillna(0)

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
        "report": "",  # No reports in CheXpert
        "source": "chexpert"
    })
