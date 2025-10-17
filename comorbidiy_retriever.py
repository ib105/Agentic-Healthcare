from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Initialize Embedding Function ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load FAISS Vector Store (previously saved using save_local("./faiss_db")) ---
vectorstore = FAISS.load_local(
    "./faiss_db",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True  # required for local pickle load
)

# --- Search Function ---
def search_comorbidities(query: str, k: int = 3):
    """
    Searches the comorbidity FAISS vector database for relevant entries.
    Returns a list of (Document, distance) tuples.
    """
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    if not results_with_scores:
        print("No relevant comorbidities found.")
        return []

    # Example printout (optional)
    # for i, (doc, distance) in enumerate(results_with_scores, 1):
    #     print(f"\n{i}. Condition: {doc.metadata.get('condition', 'Unknown')}")
    #     print(f"   ICD-10: {doc.metadata.get('icd10', 'N/A')}")
    #     print(f"   Distance: {distance:.4f}")
    #     print(f"   Text: {doc.page_content[:200]}...")

    return results_with_scores
