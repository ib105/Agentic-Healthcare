import os
import operator
from dotenv import load_dotenv
from typing import Annotated, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes
from icd11_retriever import search_icd11_local, fetch_icd11_local_details
from report_loading import extract_text_from_pdf, chunk_text, extract_text_from_image

# ============================
# Environment Setup
# ============================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ============================
# LLM Setup
# ============================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================
# Schema Definitions
# ============================
class Condition(BaseModel):
    name: str = Field(description="Detected medical condition.")
    description: str | None = Field(default=None, description="Short note or context about the condition.")

class ConditionExtraction(BaseModel):
    conditions: List[Condition] = Field(description="List of extracted medical conditions.")
    highlighted_report: str = Field(description="Original report with detected conditions highlighted using **Markdown bold**.")

condition_extractor = llm.with_structured_output(ConditionExtraction)

# ============================
# State Definitions
# ============================
class State(TypedDict):
    patient_report: str
    tasks: list[str]
    completed_sections: Annotated[list, operator.add]
    extracted_conditions: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    task: str
    patient_report: str
    extracted_conditions: list[str]
    completed_sections: Annotated[list, operator.add]

# ============================
# Nodes
# ============================
def orchestrator(state: State):
    """Define multiple tasks for medical analysis workflow."""
    tasks = [
        "Extract and highlight medical conditions from the patient report.",
        "Find potential comorbidities for extracted conditions using vector search."
    ]
    return {"tasks": tasks}


def condition_extractor_worker(state: WorkerState):
    """Worker 1: Extracts and highlights medical conditions."""
    result = condition_extractor.invoke([
        SystemMessage(content="You are a clinical NLP model that extracts all medical conditions mentioned in a report and highlights them using **Markdown bold**."),
        HumanMessage(content=f"Patient Report:\n\n{state['patient_report']}")
    ])

    if not result.conditions:
        summary_text = "No medical conditions detected."
        return {"completed_sections": [summary_text], "extracted_conditions": []}

    extracted_conditions = [c.name for c in result.conditions]
    summary_text = (
        "### Extracted Medical Conditions\n" +
        "\n".join([f"- **{c.name}**" for c in result.conditions]) +
        "\n\n### Highlighted Report:\n" + result.highlighted_report
    )

    return {"completed_sections": [summary_text], "extracted_conditions": extracted_conditions}


def comorbidity_worker(state: WorkerState):
    """Worker 2: Finds potential comorbidities for extracted conditions."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions available for comorbidity search."]}

    comorbidity_summary = ["### Potential Comorbidities"]

    for condition in state["extracted_conditions"]:
        comorbidity_summary.append(f"\n#### {condition}")
        results = search_comorbidities(condition, k=3)

        if not results:
            comorbidity_summary.append("- No related comorbidities found.")
        else:
            for doc,distance in results:
                comorbidity_summary.append(
                    f"- {doc.metadata.get('condition', 'Unknown')} (ICD-10: {doc.metadata.get('icd10', 'N/A')}, Distance: {distance:.4f})"
                )

    return {"completed_sections": ["\n".join(comorbidity_summary)]}

def cpt_worker(state: WorkerState):
    """Worker 3: Finds related CPT/HCPCS procedure codes for extracted conditions and generates clinical explanations."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions available for CPT code analysis."]}

    cpt_summary = ["### Suggested CPT/HCPCS Codes"]

    for condition in state["extracted_conditions"]:
        cpt_summary.append(f"\n#### {condition}")

        # Step 1: Retrieve top matches from FAISS
        results = search_cpt_codes(condition, k=5)

        if not results:
            cpt_summary.append("- No related CPT codes found.")
            continue

        # Step 2: Format retrieved data
        top_codes = [
            f"{item['code']} — {item['description']} (similarity: {item['similarity']})"
            for item in results
        ]
        formatted_codes = "\n".join([f"- {c}" for c in top_codes])

        # Step 3: Ask the LLM to summarize CPT mapping contextually
        prompt = f"""
        You are a clinical coding assistant. Based on the following retrieved CPT/HCPCS codes and their descriptions, explain what procedures might relate to the condition: "{condition}".
        If there is no single direct CPT code, say that explicitly and summarize the likely related categories.
        
        Retrieved codes:
        {formatted_codes}

        Respond in a concise markdown section with:
        - A short overview paragraph.
        - A bulleted list of relevant codes and their use.
        """

        llm_response = llm.invoke([HumanMessage(content=prompt)])

        cpt_summary.append(llm_response.content.strip())

    return {"completed_sections": ["\n".join(cpt_summary)]}

def icd11_local_worker(state: WorkerState):
    """Worker 4: Retrieves ICD-11 diagnostic codes and summaries."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions available for ICD-11 lookup."]}

    icd_sections = ["### ICD-11 Diagnostic Codes (Local API)"]

    for condition in state["extracted_conditions"]:
        icd_sections.append(f"\n#### {condition}")
        results = search_icd11_local(condition, limit=3)

        if not results:
            icd_sections.append("- No matching ICD-11 codes found.")
            continue

        for res in results:
            icd_sections.append(f"- **{res['code']}** — {res['title']}")
            if res["uri"]:
                detail = fetch_icd11_local_details(res["uri"])
                if detail and detail.get("definition"):
                    icd_sections.append(f"  *Definition:* {detail['definition'][:250]}...")

    return {"completed_sections": ["\n".join(icd_sections)]}


def synthesizer(state: State):
    """Combine all worker results into a final report."""
    sections = [s for s in state["completed_sections"] if s.strip()]
    combined_text = "\n\n---\n\n".join(sections)
    return {"final_report": [combined_text]}

# ============================
# Conditional Edge Function
# ============================
def assign_workers(state: State):
    return [Send("condition_extractor_worker", {"task": "Extract conditions", "patient_report": state["patient_report"]})]

# ============================
# Graph Assembly
# ============================
graph = StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("condition_extractor_worker", condition_extractor_worker)
graph.add_node("comorbidity_worker", comorbidity_worker)
graph.add_node("cpt_worker", cpt_worker)
graph.add_node("icd11_local_worker", icd11_local_worker)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_workers, ["condition_extractor_worker"])
graph.add_edge("condition_extractor_worker", "comorbidity_worker")
graph.add_edge("comorbidity_worker", "cpt_worker")
graph.add_edge("cpt_worker", "icd11_local_worker")
graph.add_edge("icd11_local_worker", "synthesizer")
graph.add_edge("synthesizer", END)

comorbidity_pipeline = graph.compile()

# ============================
# Display Graph
# ============================
graph_image = comorbidity_pipeline.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(graph_image)
print("Workflow graph saved as workflow.png")

if __name__ == "__main__":
    # ============================
    # Example Run
    # ============================
    input_path = "Med-reports/Cardiologist-Viskin-Report-page-1.jpg"

    # Detect file type
    ext = os.path.splitext(input_path)[-1].lower()

    if not os.path.exists(input_path):
        print(f"File not found at {input_path}.")
    elif ext == ".pdf":
        print(f"Extracting text from PDF: {input_path}")
        full_text = extract_text_from_pdf(input_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        print(f"Extracting text from image: {input_path}")
        full_text = extract_text_from_image(input_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf, .jpg, or .png file.")        


    chunks = chunk_text(full_text)
    print(f"Extracted {len(chunks)} chunk(s) from the report.")

    final_reports = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}...")
        state = comorbidity_pipeline.invoke({"patient_report": chunk})
        final_text = (
            "\n\n".join(state["final_report"])
            if isinstance(state["final_report"], list)
            else state["final_report"]
        )
        final_reports.append(f"## Chunk {i}\n{final_text}")

    # ---- Combine and Save ----
    combined_report = "\n\n---\n\n".join(final_reports)

    print("\n========== FINAL REPORT ==========\n")
    print(combined_report)
