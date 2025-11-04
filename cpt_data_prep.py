# ===========================================================
# Build CPT/HCPCS FAISS Vectorstore
# ===========================================================
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

df = pd.read_excel("CPT/2025_DHS_Code_List_Addendum_11_26_2024.xlsx", sheet_name=0)
df = df.dropna(how="all")

# --- Normalize columns ---
df.columns = ["CODE", "DESCRIPTION"]

# --- Keep only valid alphanumeric codes ---
df = df[df["CODE"].astype(str).str.match(r"^[A-Za-z0-9]+$")].reset_index(drop=True)

# --- Prepare text for embedding ---
df["text"] = df.apply(lambda x: f"Code: {x.CODE}. Description: {x.DESCRIPTION}", axis=1)

# --- Convert each row to a LangChain Document ---
docs = [
    Document(
        page_content=row["text"],
        metadata={"code": row["CODE"]}
    )
    for _, row in df.iterrows()
]

print(f"Created {len(docs)} CPT/HCPCS documents")

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_function)

vectorstore.save_local("CPT/cpt_faiss")

print("CPT FAISS vector database built and saved to CPT/cpt_faiss")
