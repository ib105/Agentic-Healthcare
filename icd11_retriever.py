import requests
import re
from typing import List, Dict, Optional
from functools import lru_cache
import unicodedata
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# WHO ICD-11 API endpoints
TOKEN_ENDPOINT = "https://icdaccessmanagement.who.int/connect/token"
API_BASE = "https://id.who.int/icd"
ENTITY_ENDPOINT = f"{API_BASE}/entity"
MMS_LINEARIZATION = f"{API_BASE}/release/11/2024-01/mms"

CLIENT_ID = os.getenv("ICD_CLIENT_ID")
CLIENT_SECRET = os.getenv("ICD_CLIENT_SECRET")

# Token cache
_token_cache = {
    "access_token": None,
    "expires_at": None
}

def get_access_token() -> str:
    """
    Obtain OAuth2 access token using client credentials.
    Caches the token until it expires.
    """
    global _token_cache
    
    # Return cached token if still valid
    if (_token_cache["access_token"] and 
        _token_cache["expires_at"] and 
        datetime.now() < _token_cache["expires_at"]):
        return _token_cache["access_token"]
    
    # Request new token
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "icdapi_access",
        "grant_type": "client_credentials"
    }
    
    try:
        response = requests.post(TOKEN_ENDPOINT, data=payload, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        
        # Cache the token
        _token_cache["access_token"] = token_data["access_token"]
        # Set expiry time (subtract 60 seconds for safety margin)
        expires_in = token_data.get("expires_in", 3600)
        _token_cache["expires_at"] = datetime.now() + timedelta(seconds=expires_in - 60)
        
        print(f"[OAuth2] Token obtained, expires in {expires_in}s")
        return _token_cache["access_token"]
        
    except Exception as e:
        print(f"[OAuth2 Error] Failed to obtain token: {e}")
        raise

def get_headers() -> Dict[str, str]:
    """Get headers with fresh access token."""
    token = get_access_token()
    return {
        "Authorization": f"Bearer {token}",
        "API-Version": "v2",
        "Accept": "application/json",
        "Accept-Language": "en"
    }

def clean_html(text: str) -> str:
    """Remove HTML tags from ICD titles."""
    return re.sub(r"<[^>]+>", "", text or "")

def extract_text_value(obj) -> str:
    """Extract text from various ICD-11 response formats."""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return obj.get("@value", "") or obj.get("value", "")
    elif isinstance(obj, list) and len(obj) > 0:
        return extract_text_value(obj[0])
    return ""

def normalize_term(term: str) -> str:
    """Simplify terms for flexible matching."""
    term = term.lower().strip()
    term = unicodedata.normalize("NFKD", term)
    term = re.sub(r"[^a-z0-9\s]", " ", term)
    term = re.sub(r"\s+", " ", term)
    return term

@lru_cache(maxsize=1024)
def get_mms_code_from_foundation(foundation_uri: str) -> Optional[str]:
    """
    Convert a foundation entity URI to its MMS linearization base code.
    Does not attempt to traverse into child/leaf codes.
    """
    try:
        entity_id = foundation_uri.split("/")[-1]
        mms_url = f"{MMS_LINEARIZATION}/{entity_id}"
        response = requests.get(mms_url, headers=get_headers(), timeout=10)

        if response.status_code == 200:
            data = response.json()
            code = (
                data.get("code")
                or data.get("theCode")
                or data.get("blockId")
                or data.get("codeRange")
            )
            return code
    except Exception as e:
        print(f"[MMS Lookup Error for {foundation_uri}] {e}")
    return None

def search_icd11(term: str, limit: int = 5, use_fuzzy: bool = True, filter_blocks: bool = True) -> List[Dict]:
    """
    Search ICD-11 entities using WHO API with OAuth2 authentication.
    Returns actual ICD-11 codes from MMS linearization.
    
    Args:
        term: Search term
        limit: Maximum number of results
        use_fuzzy: Enable flexible search
        filter_blocks: If True, filters out block/chapter codes and returns only leaf codes
    """
    url = f"{ENTITY_ENDPOINT}/search"
    params = {
        "q": term,
        "flatResults": "true",
        "useFlexisearch": "true" if use_fuzzy else "false"
    }
    
    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ICD-11 Search Error] {e}")
        return []
    
    results = []
    entities = data.get("destinationEntities", [])
    
    for item in entities:
        if len(results) >= limit:
            break
            
        foundation_uri = item.get("id", "")
        title = clean_html(item.get("title", "Unknown"))
        
        # Try to get the actual ICD-11 code from MMS linearization
        code = get_mms_code_from_foundation(foundation_uri)
        
        # Fallback to any code in the search result
        if not code:
            code = (
                item.get("theCode") or
                item.get("code") or
                item.get("blockId") or
                foundation_uri.split("/")[-1]  # Last resort: entity ID
            )
        
        # Filter out block codes if requested
        if filter_blocks and code and code.startswith("Block"):
            continue
        
        results.append({
            "code": code,
            "title": title,
            "uri": foundation_uri
        })
    
    # If no results and fuzzy search not used, try again with fuzzy
    if not results and not use_fuzzy:
        return search_icd11(term, limit, use_fuzzy=True, filter_blocks=filter_blocks)
    
    # If still no results, try simplified term
    if not results:
        normalized = normalize_term(term)
        simplified = normalized.split()[0] if " " in normalized else normalized
        if simplified != term.lower():
            return search_icd11(simplified, limit, use_fuzzy=True, filter_blocks=filter_blocks)
    
    return results

def fetch_icd11_details(uri: str) -> Optional[Dict]:
    """
    Fetch detailed information for an ICD-11 entity.
    Returns code, title, and definition.
    """
    try:
        # Ensure HTTPS
        uri = uri.replace("http://", "https://")
        
        response = requests.get(uri, headers=get_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Get the MMS code
        code = get_mms_code_from_foundation(uri)
        
        if not code:
            code = (
                data.get("code") or
                data.get("theCode") or
                uri.split("/")[-1]
            )
        
        # Extract title
        title = extract_text_value(data.get("title", {}))
        
        # Extract definition
        definition = extract_text_value(data.get("definition", {}))
        
        return {
            "code": code,
            "title": clean_html(title),
            "definition": clean_html(definition)
        }
        
    except Exception as e:
        print(f"[ICD-11 Details Error] {e}")
        return None

def search_by_code(code: str) -> Optional[Dict]:
    """
    Search for an ICD-11 entity by its code.
    Useful for code validation.
    """
    # First try the codeinfo endpoint
    url = f"{MMS_LINEARIZATION}/codeinfo/{code}"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            title = extract_text_value(data.get("title", {}))
            
            # If title is empty, try fetching from stemId
            if not title and data.get("stemId"):
                stem_response = requests.get(data["stemId"], headers=get_headers(), timeout=10)
                if stem_response.status_code == 200:
                    stem_data = stem_response.json()
                    title = extract_text_value(stem_data.get("title", {}))
            
            return {
                "code": code,
                "title": clean_html(title),
                "uri": data.get("stemId", ""),
                "definition": extract_text_value(data.get("definition", {}))
            }
    except Exception as e:
        print(f"[Code Search Error] {e}")
    
    # If codeinfo fails, try searching by the code as a term
    results = search_icd11(code, limit=1, filter_blocks=False)
    if results and results[0]["code"] == code:
        return results[0]
        
    return None

def browse_chapter(chapter_code: str, max_results: int = 20) -> List[Dict]:
    """
    Browse all codes within a chapter or block.
    Useful for exploring the hierarchy.
    """
    url = f"{MMS_LINEARIZATION}/{chapter_code}"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            children = data.get("child", [])
            
            results = []
            for child in children[:max_results]:
                child_uri = child if isinstance(child, str) else child.get("@id", "")
                if child_uri:
                    child_id = child_uri.split("/")[-1]
                    child_code = get_mms_code_from_foundation(child_uri)
                    
                    # Fetch child details
                    child_response = requests.get(child_uri, headers=get_headers(), timeout=10)
                    if child_response.status_code == 200:
                        child_data = child_response.json()
                        title = extract_text_value(child_data.get("title", {}))
                        
                        results.append({
                            "code": child_code or child_id,
                            "title": clean_html(title),
                            "uri": child_uri
                        })
            
            return results
    except Exception as e:
        print(f"[Browse Error] {e}")
    
    return []

def get_z_codes_for_condition(condition: str, limit: int = 3) -> List[Dict]:
    """
    Search for ICD-11 Z-codes (Chapter 24: Factors influencing health status).
    Z-codes start with 'Q' in ICD-11 (e.g., QA00-QE8Z).
    """
    z_query = f"{condition} factors influencing health status screening prevention"
    results = search_icd11(z_query, limit=limit*3, filter_blocks=True)
    
    # Filter for Chapter 24 codes (Q prefix)
    z_codes = [r for r in results if r['code'].startswith('Q')][:limit]
    
    # If no Q codes, try broader search
    if not z_codes:
        z_query = f"history of {condition}"
        results = search_icd11(z_query, limit=limit*2)
        z_codes = [r for r in results if r['code'].startswith('Q')][:limit]
    
    return z_codes

# ------------------------------
# Test function
# ------------------------------
if __name__ == "__main__":
    diabetes_results = search_icd11("Diabetes", limit=3)
    for idx, result in enumerate(diabetes_results, 1):
        print(f"{idx}. {result['code']} — {result['title']}")
    
    print("\n" + "=" * 60)
