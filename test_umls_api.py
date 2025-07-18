import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("UMLS_API_KEY")

UMLS_AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SERVICE = "http://umlsks.nlm.nih.gov"
VERSION = "2025AA"
CUI = "C4728208"

os.makedirs("api_output", exist_ok=True)

def get_tgt(api_key):
    response = requests.post(UMLS_AUTH_URL, data={"apikey": api_key})
    if response.status_code == 201:
        return response.headers["location"].split("/")[-1]
    raise RuntimeError(f"Failed to get TGT: {response.status_code} - {response.text}")

def get_st(tgt):
    response = requests.post(f"{UMLS_AUTH_URL}/{tgt}", data={"service": SERVICE})
    if response.status_code == 200:
        return response.text
    raise RuntimeError(f"Failed to get ST: {response.status_code} - {response.text}")

def fetch_with_ticket(endpoint, ticket):
    response = requests.get(endpoint, params={"ticket": ticket})
    if response.ok:
        return response.json()
    else:
        print(f"Failed to fetch {endpoint}: {response.status_code} - {response.text}")
        return None

def save_json(data, name):
    with open(f"api_output/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {name}.json")

def resolve_name_and_cui(entry, tgt):
    ui = entry.get("ui", "N/A")
    name = entry.get("name", "N/A")
    sabs = entry.get("rootSource", "N/A")
    if not ui.startswith("C"):
        search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{VERSION}"
        st = get_st(tgt)
        params = {
            "string": ui,
            "inputType": "sourceUi",
            "sabs": sabs,
            "searchType": "exact",
            "ticket": st,
        }
        resp = requests.get(search_url, params=params)
        if resp.ok:
            results = resp.json().get("result", {}).get("results", [])
            if results:
                name = results[0]["name"] 
                cui = results[0]["ui"]
            else:  
                cui = "N/A"
            return f"{{{name}, {cui}}}"
    return f"{{({name}), {ui}}}"

def extract_summary(cui, concept_info, definition_info, parents, descendants, tgt):
    name = concept_info.get("name", "N/A")
    semantic = "; ".join([stype.get("name", "") for stype in concept_info.get("semanticTypes", [])])

    definitions = definition_info.get("result", [])
    if definitions:
        msh_defs = [d for d in definitions if d.get("rootSource") == "MSH"]
        definition = msh_defs[0].get("value", "N/A") if msh_defs else definitions[0].get("value", "N/A")
    else:
        definition = "N/A"

    parent_lines = [resolve_name_and_cui(p, tgt) for p in parents]
    descendant_lines = [resolve_name_and_cui(d, tgt) for d in descendants]

    print("\n📄 SUMMARY")
    print(f"Name:           {name}")
    print(f"CUI:            {cui}")
    print(f"Semantic Type:  {semantic}")
    print(f"Definition:     {definition}\n")

    print("Parent Concepts:")
    print("\n".join(parent_lines) or "N/A")
    print("\nDescendant Concepts:")
    print("\n".join(descendant_lines) or "N/A")

def main():
    if not API_KEY:
        print("❌ Missing UMLS_API_KEY in .env")
        return

    tgt = get_tgt(API_KEY)
    st = get_st(tgt)

    concept_url = f"https://uts-ws.nlm.nih.gov/rest/content/{VERSION}/CUI/{CUI}"
    concept = fetch_with_ticket(concept_url, st).get("result", {})
    save_json({"result": concept}, f"{CUI}_concept")

    def_url = concept.get("definitions")
    if def_url == "NONE":
        print("⚠️ No definition URL found.")
        defs = {
            "result": [{
                "value": "No description provided",
                "rootSource": "N/A"
            }]
        }
    else:
        st = get_st(tgt)
        defs = fetch_with_ticket(def_url, st)
        save_json(defs, f"{CUI}_definitions")

    # === Atoms → SourceDescriptor ===
    atoms_url = concept.get("atoms")
    st = get_st(tgt)
    atoms = fetch_with_ticket(atoms_url, st).get("result", [])
    save_json({"result": atoms}, f"{CUI}_atoms")

    src_desc_url = None

    for atom in atoms:
        sd = atom.get("sourceDescriptor")
        if atom.get("rootSource") == "MSH" and atom.get("obsolete") != "true" and sd and sd != "NONE" and atom.get("language") == "ENG":
            src_desc_url = sd
            break

    if not src_desc_url:
        print("⚠️ No non-obsolete MSH atom found, searching other atoms.")
        for atom in atoms:
            sd = atom.get("sourceDescriptor")
            if atom.get("obsolete") != "true" and sd and sd != "NONE":
                src_desc_url = sd
                break

    if not src_desc_url:
        print("❌ No usable atom found with sourceDescriptor. Skipping parents/descendants.")
        parents = []
        descendants = []
    else:
        st = get_st(tgt)
        parents_response = fetch_with_ticket(f"{src_desc_url}/parents", st)
        parents = parents_response.get("result", []) if parents_response else []
        save_json(parents, f"{CUI}_parents")

        st = get_st(tgt)
        descendants_response = fetch_with_ticket(f"{src_desc_url}/descendants", st)
        descendants = descendants_response.get("result", []) if descendants_response else []
        save_json(descendants, f"{CUI}_descendants")

    extract_summary(CUI, concept, defs, parents, descendants, tgt)

if __name__ == "__main__":
    main()
    tgt = get_tgt(API_KEY)
    st = get_st(tgt)
    print(fetch_with_ticket("https://uts-ws.nlm.nih.gov/rest/content/2025AA/source/MDR/10081792", st))