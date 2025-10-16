from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
    client_settings=Settings(anonymized_telemetry=False)
)

def search_comorbidities(query: str, k: int = 3):
    """
    Searches the comorbidity vector database for relevant entries.
    Returns a list of (Document, distance) tuples.
    """
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    if not results_with_scores:
        print("No relevant comorbidities found.")
        return []

    # print(f"\nTop {len(results_with_scores)} matching comorbidities for query: '{query}'")

    # for i, (doc, distance) in enumerate(results_with_scores, 1):
    #     print(f"\n{i}. Condition: {doc.metadata.get('condition', 'Unknown')}")
    #     print(f"   ICD-10: {doc.metadata.get('icd10', 'N/A')}")
    #     print(f"   Chunk ID: {doc.metadata.get('chunk_id', '-')}")
    #     print(f"   Distance: {distance:.4f}") 
    #     print(f"   Text: {doc.page_content[:200]}...")

    return results_with_scores
