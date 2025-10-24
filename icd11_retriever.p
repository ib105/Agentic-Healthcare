import requests
import re
from typing import List, Dict, Optional
from functools import lru_cache
from difflib import get_close_matches
import unicodedata

# Root for your local ICD API (adjust if your local path differs)
ICD11_LOCAL_ROOT = "http://localhost:8382/icd"
ENTITY_ENDPOINT = f"{ICD11_LOCAL_ROOT}/entity"
# Linearization endpoint base for ICD-11 MMS linearization
MMS_RELEASE_BASE = f"{ICD11_LOCAL_ROOT}/release/11/mms"

HEADERS = {
    "API-Version": "v2",
    "Accept": "application/json",
    "Accept-Language": "en"
}

def normalize_term(term: str) -> str:
    """Simplify terms for flexible matching."""
    term = term.lower().strip()
    term = unicodedata.normalize("NFKD", term)
    term = re.sub(r"[^a-z0-9\s]", " ", term)
    term = re.sub(r"\s+", " ", term)
    return term

def search_icd11_local(term: str, limit: int = 5, linearization_version: Optional[str] = None) -> List[Dict]:
    """Flexible local ICD-11 search with fuzzy fallback."""
    url = f"{ENTITY_ENDPOINT}/search"
    params = {"q": term, "flatResults": "true", "includeKeywordResult": "true"}

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ICD-11 Local API error] {e}")
        return []

    results = []
    entities = data.get("destinationEntities", [])
    
    # --- Step 1: Direct matches ---
    for item in entities[:limit]:
        code = item.get("theCode") or item.get("code") or item.get("classificationCode")
        if not code or code in ("None", "N/A"):
            entity_id = (item.get("id") or "").split("/")[-1]
            code = lookup_mms_code_for_entity(entity_id, linearization_version) or entity_id

        results.append({
            "code": code,
            "title": clean_html(item.get("title", "Unknown")),
            "uri": item.get("id", None)
        })

    # --- Step 2: If empty, retry with fuzzy term matching ---
    if not results:
        base_term = normalize_term(term)
        alt_terms = [
            term.replace("syndrome", "").strip(),
            base_term.replace("long", "").replace("prolonged", "long").strip(),
            base_term.split()[0] if " " in base_term else base_term
        ]
        for alt in set(alt_terms):
            if not alt:
                continue
            params["q"] = alt
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=10)
                r.raise_for_status()
                data = r.json()
                entities = data.get("destinationEntities", [])
                if entities:
                    for item in entities[:limit]:
                        code = item.get("theCode") or item.get("code") or item.get("classificationCode")
                        if not code or code in ("None", "N/A"):
                            entity_id = (item.get("id") or "").split("/")[-1]
                            code = lookup_mms_code_for_entity(entity_id, linearization_version) or entity_id
                        results.append({
                            "code": code,
                            "title": clean_html(item.get("title", "Unknown")),
                            "uri": item.get("id", None)
                        })
                    break
            except Exception:
                continue

    return results

# ------------------------------
# Utility: clean up HTML tags
# ------------------------------
def clean_html(text: str) -> str:
    """Remove HTML tags (e.g., <em>) from ICD titles."""
    return re.sub(r"<[^>]+>", "", text or "")

# ------------------------------
# Utility: find likely ICD code in arbitrary JSON
# ------------------------------
ICD_CODE_RE = re.compile(r"^[0-9A-Z]{1,4}(?:[A-Z0-9]*)(?:\.[0-9A-Z]+)?$")  # matches e.g. 1A00, BA01.0, 1A

def find_code_in_json(obj) -> Optional[str]:
    """
    Recursively search a JSON-like object for a short string that looks like an ICD code.
    Returns the first match or None.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            # common keys to check quickly
            if isinstance(v, str) and ICD_CODE_RE.match(v):
                return v
            found = find_code_in_json(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_code_in_json(item)
            if found:
                return found
    elif isinstance(obj, str):
        if ICD_CODE_RE.match(obj):
            return obj
    return None

# ------------------------------
# Linearization lookup (cached)
# ------------------------------
@lru_cache(maxsize=1024)
def lookup_mms_code_for_entity(entity_id: str, version: Optional[str] = None) -> Optional[str]:
    """
    Try to fetch the MMS linearization representation for the entity and extract the ICD-11 short code.
    Handles /entity/, /foundation/, and /release/11/mms/ paths.
    """
    if not entity_id:
        return None

    entity_id = entity_id.strip().split("/")[-1]  # Ensure just the numeric or short ID
    candidates = [
        f"{MMS_RELEASE_BASE}/{entity_id}",
        f"{ICD11_LOCAL_ROOT}/release/11/foundation/{entity_id}",
        f"{ENTITY_ENDPOINT}/{entity_id}"
    ]
    if version:
        candidates.insert(0, f"{ICD11_LOCAL_ROOT}/release/11/{version}/mms/{entity_id}")

    for url in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()

            # Priority order for code extraction
            for key in ["theCode", "code", "classificationCode", "stemCode"]:
                if key in data and isinstance(data[key], str) and ICD_CODE_RE.match(data[key]):
                    return data[key]

            # fallback recursive search
            found = find_code_in_json(data)
            if found:
                return found

        except Exception:
            continue

    return None

# ------------------------------
# Search ICD-11 locally
# ------------------------------
def search_icd11_local(term: str, limit: int = 5, linearization_version: Optional[str] = None) -> List[Dict]:
    """
    Searches for ICD-11 entities using the local API endpoint.
    If a short ICD code is not present in the search hit, attempt to fetch the MMS linearization to obtain it.
    """
    url = f"{ENTITY_ENDPOINT}/search"
    params = {
        "q": term,
        "flatResults": "true",
        "includeKeywordResult": "true"
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ICD-11 Local API error] {e}")
        print(f"Response: {r.text[:300] if 'r' in locals() else 'No response'}")
        return []

    results = []
    for item in data.get("destinationEntities", [])[:limit]:
        # get a best-effort code from multiple fields
        code = (
            item.get("theCode")
            or item.get("code")
            or item.get("classificationCode")
        )

        # if still missing or None-like, attempt linearization lookup
        if not code or code in ("None", "N/A"):
            entity_uri = item.get("id") or ""
            # extract the trailing numeric id if present
            entity_id = entity_uri.split("/")[-1] if entity_uri else ""
            if entity_id:
                mms_code = lookup_mms_code_for_entity(entity_id, linearization_version)
                if mms_code:
                    code = mms_code

        # final fallback: entity id
        if not code or code in ("None",):
            code = (item.get("id", "").split("/")[-1]) if item.get("id") else "N/A"

        results.append({
            "code": code,
            "title": clean_html(item.get("title", "Unknown")),
            "uri": item.get("id", None)
        })

    return results

# ------------------------------
# Fetch ICD-11 entity details
# ------------------------------
def fetch_icd11_local_details(uri: str, linearization_version: Optional[str] = None) -> Optional[Dict]:
    """
    Fetches detailed ICD-11 entity data via the local API.
    Always routes WHO URIs through localhost, and if the short ICD code isn't present in the entity,
    attempts to fetch it from the MMS linearization.
    """
    # normalize uri -> local entity endpoint
    if uri.startswith("http://id.who.int/icd/entity/") or uri.startswith("https://id.who.int/icd/entity/"):
        entity_id = uri.split("/")[-1]
        uri = f"{ENTITY_ENDPOINT}/{entity_id}"
    elif not uri.startswith("http"):
        uri = f"{ENTITY_ENDPOINT}/{uri}"

    print(f"[ICD-11 Local Fetch] {uri}")

    try:
        r = requests.get(uri, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ICD-11 Detail Fetch Error] {e}")
        return None

    # attempt multiple field names
    code = (
        data.get("code")
        or data.get("theCode")
        or data.get("classificationCode")
    )

    # If code missing, try linearization
    if not code or code in ("None", "N/A"):
        # entity id fallback
        entity_id = uri.split("/")[-1] if uri else ""
        if entity_id:
            mms_code = lookup_mms_code_for_entity(entity_id, linearization_version)
            if mms_code:
                code = mms_code

    # last resort: use entity id
    if not code or code in ("None",):
        code = uri.split("/")[-1] if uri else "N/A"

    title_value = ""
    # title might be in a nested structure depending on endpoint
    if isinstance(data.get("title"), dict):
        title_value = data.get("title", {}).get("@value", "") or data.get("title", {}).get("value", "")
    elif isinstance(data.get("title"), str):
        title_value = data.get("title")
    else:
        title_value = ""

    definition_value = ""
    if isinstance(data.get("definition"), dict):
        definition_value = data.get("definition", {}).get("@value", "") or data.get("definition", {}).get("value", "")
    elif isinstance(data.get("definition"), str):
        definition_value = data.get("definition")

    return {
        "code": code,
        "title": clean_html(title_value),
        "definition": definition_value
    }

# ------------------------------
# Quick test (run as script)
# ------------------------------
if __name__ == "__main__":

    results = search_icd11_local("Dementia")
    for r in results:
        print(f"{r['code']} — {r['title']}")
        if r["uri"]:
            details = fetch_icd11_local_details(r["uri"])
            if details:
                print(f"  Definition: {details['definition'][:150]}...\n")
