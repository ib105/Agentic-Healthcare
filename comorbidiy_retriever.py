from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Initialize Embedding Function ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load FAISS Vector Store ---
vectorstore = FAISS.load_local(
    "./comorbidities_faiss",
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

    return results_with_scores
