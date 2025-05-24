import re
import pandas as pd

# Load your vocab
vocab_path = "VIZMed/data/vocab/padchest_vocab.csv"
vocab_df = pd.read_csv(vocab_path)
seen_cuis = set(vocab_df["cui"].dropna().astype(str))

# Load the full PadChest hierarchy text
with open("datasets/PadChest/Extras/Radiographic Findings.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Extract all valid CUIs from the hierarchy text
expected_cuis = set(re.findall(r'C\d{7}', full_text))

# Find what you're missing
missing_cuis = expected_cuis - seen_cuis

print(f"[✓] Found {len(expected_cuis)} CUIs in hierarchy")
print(f"[✓] Found {len(seen_cuis)} CUIs in vocab")
print(f"[!] Missing {len(missing_cuis)} CUIs in vocab:\n")
print(sorted(missing_cuis))
