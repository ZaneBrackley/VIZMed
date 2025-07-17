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

    def get_concept_metadata(self, cui: str) -> dict:
        ticket = self._get_st()

        concept_url = f"{UMLS_API_BASE}/content/current/CUI/{cui}"
        concept_response = requests.get(concept_url, params={"ticket": ticket})
        concept = concept_response.json().get("result", {}) if concept_response.ok else {}

        return {
            "cui": cui,
            "name": concept.get("name", "N/A"),
            "semantic_type": "; ".join([stype.get("name", "") for stype in concept.get("semanticTypes", []) if isinstance(stype, dict)]) or "N/A",
        }