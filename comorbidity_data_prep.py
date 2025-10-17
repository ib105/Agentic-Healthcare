import pandas as pd
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===========================================================
# Load and Clean Data
# ===========================================================

df = pd.read_excel("Comorbidities.xlsx")

# Drop fully empty rows
df.dropna(how="all", inplace=True)

# Forward-fill comorbid conditions and ICD codes
df["comorbidconditions"] = df["comorbidconditions"].ffill()
df["icd10codes"] = df["icd10codes"].ffill()

# Extract clean ICD-10 codes
def extract_icd_codes(text):
    if pd.isna(text):
        return None
    codes = re.findall(r"[A-Z]\d{2}\.\d", text)
    return ", ".join(codes) if codes else None

df["icd10codes_clean"] = df["icd10codes"].apply(extract_icd_codes)

# Combine consecutive rows belonging to the same (condition + ICD) group
group_cols = ["comorbidconditions", "icd10codes_clean"]
combined_df = (
    df.groupby(group_cols, dropna=True, sort=False)
      .agg({
          "problemsymptomslists": lambda x: " ".join(x.dropna().astype(str))
      })
      .reset_index()
)

# Build combined text for embeddings
combined_df["text"] = (
    "Comorbid Condition: " + combined_df["comorbidconditions"] + ". "
    + "ICD-10 Code: " + combined_df["icd10codes_clean"].fillna("") + ". "
    + "Details: " + combined_df["problemsymptomslists"]
)

# ===========================================================
# Text Chunking
# ===========================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)

docs = []
for _, row in combined_df.iterrows():
    chunks = splitter.split_text(row["text"])
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "condition": row["comorbidconditions"],
                    "icd10": row["icd10codes_clean"],
                    "chunk_id": i,
                }
            )
        )

print(f"Created {len(docs)} text chunks from {len(combined_df)} comorbid conditions")

# ===========================================================
# Build and Persist Vector Store (FAISS)
# ===========================================================
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS vector store
vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_function)

# Persist FAISS index locally
vectorstore.save_local("./comorbidities_faiss")

print("FAISS vector database built and saved to ./comorbidities_faiss")
