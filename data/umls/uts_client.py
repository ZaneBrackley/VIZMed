# data/umls/uts_client.py
import requests
from typing import Optional

UMLS_AUTH_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
UMLS_API_BASE = "https://uts-ws.nlm.nih.gov/rest"
SERVICE = "http://umlsks.nlm.nih.gov"

class UMLSClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tgt = self._get_tgt()

    def _get_tgt(self):
        """Get Ticket Granting Ticket (TGT) from UMLS."""
        r = requests.post(UMLS_AUTH_URL, data={"apikey": self.api_key})
        if r.status_code == 201:
            return r.headers["location"].split("/")[-1]
        else:
            raise RuntimeError("Failed to authenticate with UMLS API")

    def _get_st(self):
        """Get Service Ticket (ST) using TGT."""
        r = requests.post(
            f"{UMLS_AUTH_URL}/{self.tgt}",
            data={"service": SERVICE}
        )
        if r.status_code == 200:
            return r.text
        else:
            raise RuntimeError("Failed to obtain service ticket")
        
    def resolve_name_and_cui(self, entry) -> str:
        ui = entry.get("ui", "N/A")
        name = entry.get("name", "N/A")
        sabs = entry.get("rootSource", "N/A")
        if not ui.startswith("C"):
            search_url = f"{UMLS_API_BASE}/search/current"
            st = self._get_st()
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
                return f"{{({name}), {cui}}}"
        return f"{{({name}), {ui}}}"            

    def get_concept_metadata(self, cui: str) -> dict:
        st = self._get_st()

        # === Step 1: Concept
        concept_url = f"{UMLS_API_BASE}/content/current/CUI/{cui}"
        concept = requests.get(concept_url, params={"ticket": st}).json().get("result", {})
        name = concept.get("name", "N/A")
        semantic = "; ".join([stype.get("name", "") for stype in concept.get("semanticTypes", [])])

        # === Step 2: Definitions
        def_url = concept.get("definitions")
        definition = "No description provided"
        if def_url and def_url != "NONE":
            st = self._get_st()
            defs = requests.get(def_url, params={"ticket": st}).json().get("result", [])
            msh_defs = [d for d in defs if d.get("rootSource") == "MSH"]
            if msh_defs:
                definition = msh_defs[0].get("value", definition)
            elif defs:
                definition = defs[0].get("value", definition)

        # === Step 3: Atoms → SourceDescriptor
        atoms_url = concept.get("atoms")
        st = self._get_st()
        atoms = requests.get(atoms_url, params={"ticket": st}).json().get("result", [])
        src_desc_url = None

        for atom in atoms:
            sd = atom.get("sourceDescriptor")
            if atom.get("rootSource") == "MSH" and atom.get("obsolete") != "true" and sd and sd != "NONE" and atom.get("language") == "ENG":
                src_desc_url = sd
                break
        if not src_desc_url:
            for atom in atoms:
                sd = atom.get("sourceDescriptor")
                if atom.get("obsolete") != "true" and sd and sd != "NONE":
                    src_desc_url = sd
                    break

        # === Step 4: Parents and Descendants
        parents = []
        descendants = []

        if src_desc_url:
            try:
                st = self._get_st()
                parent_entries = requests.get(f"{src_desc_url}/parents", params={"ticket": st}).json().get("result", [])
                st = self._get_st()
                descendant_entries = requests.get(f"{src_desc_url}/descendants", params={"ticket": st}).json().get("result", [])

                parents = [self.resolve_name_and_cui(p) for p in parent_entries]
                descendants = [self.resolve_name_and_cui(d) for d in descendant_entries]
            except Exception as e:
                print(f"[!] Failed to fetch parents/descendants for {cui}: {e}")

        return {
            "cui": cui,
            "name": name,
            "definition": definition,
            "semantic_type": semantic,
            "parents": parents,
            "descendants": descendants,
        }
