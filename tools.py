"""
Medical coding tool functions.
These are the actual implementations called by the MCP server.
"""

from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes
from icd11_retriever import search_icd11, get_z_codes_for_condition

def get_comorbidities(condition: str, k: int = 3):
    """Find comorbidities for a medical condition"""
    results = search_comorbidities(condition, k=k)
    return [
        {
            "condition": doc.metadata.get("condition", "Unknown"),
            "icd10": doc.metadata.get("icd10", "N/A"),
            "distance": float(distance)
        }
        for doc, distance in results
    ]

def get_icd11(condition: str, limit: int = 3):
    """Get ICD-11 codes for a condition"""
    return search_icd11(condition, limit=limit, filter_blocks=True)

def get_cpt(procedure: str, k: int = 3):
    """Get CPT codes for a procedure"""
    return search_cpt_codes(procedure, k=k)

def get_z_codes(health_factor: str, limit: int = 3):
    """Get Z-codes for health status factors"""
    return get_z_codes_for_condition(health_factor, limit=limit)
