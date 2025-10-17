from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# Initialize Embedding Function
# ============================================================
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ============================================================
# Load FAISS CPT/HCPCS Vector Database
# ============================================================
vectorstore = FAISS.load_local(
    "CPT/cpt_faiss",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True  # only safe in trusted environments
)

# ============================================================
# Search Function
# ============================================================
def search_cpt_codes(query: str, k: int = 5):
    """
    Searches the CPT/HCPCS FAISS database for procedures related to a given condition.
    Returns a list of (Document, distance) tuples.
    """
    try:
        results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as e:
        print(f"[ERROR] CPT search failed for '{query}': {e}")
        return []

    if not results_with_scores:
        print(f"No relevant CPT codes found for query: {query}")
        return []

    # Optional: Clean and format results for readability
    formatted_results = []
    for doc, score in results_with_scores:
        code = doc.metadata.get("code") if "code" in doc.metadata else "Unknown Code"
        description = doc.page_content.strip().replace("\n", " ")
        formatted_results.append({
            "code": code,
            "description": description,
            "similarity": round(float(score), 4)
        })

    return formatted_results
