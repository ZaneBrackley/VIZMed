# preprocess/padchest_parser.py
import os, ast
import pandas as pd

PAD_CSV_PATH = os.path.join("datasets", "PadChest", "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
BASE_IMAGE_DIR = os.path.join("datasets", "PadChest")

def load_padchest():
    if not os.path.exists(PAD_CSV_PATH):
        raise FileNotFoundError(f"PadChest CSV not found at {PAD_CSV_PATH}")
    
    df = pd.read_csv(PAD_CSV_PATH, low_memory=False)

    def extract_cuis(row):
        label_cui_raw = row.get("labelCUIS")
        labels_raw = row.get("Labels")

        # If labelCUIS is empty/missing
        if pd.isna(label_cui_raw) or str(label_cui_raw).strip() in ["", "[]"]:
            try:
                labels = ast.literal_eval(labels_raw)
                if isinstance(labels, list) and "normal" in [l.lower() for l in labels]:
                    return ["C0205307"]
            except:
                pass
            return []

        try:
            cell = str(label_cui_raw).replace("'", '"').replace(' ', ', ')
            cuis = list(set(ast.literal_eval(cell)))
            return cuis if cuis else []
        except Exception:
            return []

    def resolve_image_path(row):
        return os.path.join(BASE_IMAGE_DIR, str(row["ImageDir"]), row["ImageID"])

    # Apply transformations
    processed_df = pd.DataFrame({
        "image_path": df.apply(resolve_image_path, axis=1),
        "concepts": df.apply(extract_cuis, axis=1),
        "report": df["Report"].fillna("").astype(str),
        "source": "padchest"
    })

    return processed_df
