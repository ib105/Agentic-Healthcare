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
from icd11_retriever import search_icd11, fetch_icd11_details, get_z_codes_for_condition
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
    extracted_conditions: list[str]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    patient_report: str
    extracted_conditions: list[str]

# ============================
# Nodes
# ============================
def condition_extractor_node(state: State):
    """Extracts and highlights medical conditions from patient report."""
    result = condition_extractor.invoke([
        SystemMessage(content="You are a clinical NLP model that extracts all medical conditions mentioned in a report and highlights them using **Markdown bold**."),
        HumanMessage(content=f"Patient Report:\n\n{state['patient_report']}")
    ])

    if not result.conditions:
        summary_text = "No medical conditions detected."
        return {
            "completed_sections": [summary_text], 
            "extracted_conditions": []
        }

    extracted_conditions = [c.name for c in result.conditions]
    summary_text = (
        "### Extracted Medical Conditions\n" +
        "\n".join([f"- **{c.name}**" for c in result.conditions]) +
        "\n\n### Highlighted Report:\n" + result.highlighted_report
    )

    return {
        "completed_sections": [summary_text], 
        "extracted_conditions": extracted_conditions
    }


def comorbidity_worker(state: WorkerState):
    """Worker: Finds potential comorbidities for extracted conditions."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions for comorbidity search."]}

    comorbidity_summary = ["### Potential Comorbidities"]

    for condition in state["extracted_conditions"]:
        comorbidity_summary.append(f"\n#### {condition}")
        results = search_comorbidities(condition, k=3)

        if not results:
            comorbidity_summary.append("- No related comorbidities found.")
        else:
            for doc, distance in results:
                comorbidity_summary.append(
                    f"- {doc.metadata.get('condition', 'Unknown')} (ICD-10: {doc.metadata.get('icd10', 'N/A')}, Distance: {distance:.4f})"
                )

    return {"completed_sections": ["\n".join(comorbidity_summary)]}


def cpt_worker(state: WorkerState):
    """Worker: Finds exact or best-match CPT/HCPCS procedure codes."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions for CPT code analysis."]}

    cpt_summary = ["### Suggested CPT/HCPCS Codes"]

    for condition in state["extracted_conditions"]:
        cpt_summary.append(f"\n#### {condition}")
        results = search_cpt_codes(condition, k=5)

        if not results:
            cpt_summary.append("- No related CPT codes found.")
            continue

        top_codes = [
            f"{item['code']} — {item['description']} (similarity: {item['similarity']})"
            for item in results
        ]
        formatted_codes = "\n".join([f"- {c}" for c in top_codes])

        prompt = f"""
You are a clinical coding expert. Analyze these retrieved CPT codes for the condition: "{condition}".

Retrieved codes:
{formatted_codes}

Return ONLY the most relevant code(s) in this exact format:
**Primary Code:** [CODE] — [Brief description]
**Alternative:** [CODE] — [Brief description] (only if multiple viable options exist)

If no exact match exists, state "No direct CPT code available" and suggest the closest category.
Be concise - maximum 2-3 lines.
"""
        llm_response = llm.invoke([HumanMessage(content=prompt)])
        cpt_summary.append(llm_response.content.strip())

    return {"completed_sections": ["\n".join(cpt_summary)]}


def icd11_worker(state: WorkerState):
    """Worker: Retrieves exact or best-match ICD-11 diagnostic codes."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions for ICD-11 lookup."]}

    icd_sections = ["### ICD-11 Diagnostic Codes"]

    for condition in state["extracted_conditions"]:
        icd_sections.append(f"\n#### {condition}")
        results = search_icd11(condition, limit=3, filter_blocks=True)

        if not results:
            icd_sections.append("- No matching ICD-11 codes found.")
            continue

        # Get best match (first result)
        best_match = results[0]
        icd_sections.append(f"- **Primary:** {best_match['code']} — {best_match['title']}")
        
        if best_match.get("uri"):
            detail = fetch_icd11_details(best_match["uri"])
            if detail and detail.get("definition"):
                icd_sections.append(f"  *{detail['definition'][:200]}...*")
        
        # Add alternative only if significantly different
        if len(results) > 1:
            alt = results[1]
            if alt['code'][:3] != best_match['code'][:3]:  # Different category
                icd_sections.append(f"- **Alternative:** {alt['code']} — {alt['title']}")

    return {"completed_sections": ["\n".join(icd_sections)]}


def z_code_worker(state: WorkerState):
    """Worker: Retrieves exact or best-match ICD-11 Z-codes with descriptions."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["No extracted conditions for Z-code analysis."]}

    z_sections = ["### ICD-11 Z-Codes (Health Status Factors)"]

    for condition in state["extracted_conditions"]:
        z_sections.append(f"\n#### {condition}")
        results = get_z_codes_for_condition(condition, limit=2)

        if not results:
            z_sections.append("- No relevant Z-codes found.")
            continue

        # Show best match with definition
        for idx, result in enumerate(results, 1):
            prefix = "**Primary:**" if idx == 1 else "**Alternative:**"
            z_sections.append(f"- {prefix} {result['code']} — {result['title']}")
            
            # Fetch definition for context
            if result.get("uri"):
                detail = fetch_icd11_details(result["uri"])
                if detail and detail.get("definition"):
                    z_sections.append(f"  *{detail['definition'][:150]}...*")

    return {"completed_sections": ["\n".join(z_sections)]}


def synthesizer(state: State):
    """Combine all worker results into a final report."""
    sections = [s for s in state["completed_sections"] if s.strip()]
    combined_text = "\n\n---\n\n".join(sections)
    return {"final_report": combined_text}


# ============================
# Smart Orchestrator (Parallel Routing)
# ============================
def smart_orchestrator(state: State):
    """
    Intelligently routes to relevant workers based on report content.
    Returns list of Send objects for parallel execution.
    """
    conditions = state.get("extracted_conditions", [])
    report_lower = state["patient_report"].lower()
    
    workers_to_run = []
    
    # Always run comorbidity if conditions exist
    if conditions:
        workers_to_run.append(
            Send("comorbidity_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": conditions
            })
        )
    
    # CPT codes: if procedures/treatments mentioned
    if any(kw in report_lower for kw in [
        "procedure", "surgery", "treatment", "therapy", "test", 
        "scan", "exam", "biopsy", "catheter", "endoscopy"
    ]):
        workers_to_run.append(
            Send("cpt_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": conditions
            })
        )
    
    # ICD-11: if conditions exist (diagnosis)
    if conditions:
        workers_to_run.append(
            Send("icd11_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": conditions
            })
        )
    
    # Z-codes: if preventive/status/screening factors mentioned
    if any(kw in report_lower for kw in [
        "screening", "prevention", "history of", "family history",
        "status", "vaccination", "counseling", "observation",
        "encounter", "follow-up", "risk factor"
    ]):
        workers_to_run.append(
            Send("z_code_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": conditions
            })
        )
    
    # If no workers selected, send empty signal to synthesizer
    if not workers_to_run:
        return [Send("synthesizer", state)]
    
    return workers_to_run


# ============================
# Graph Assembly
# ============================
graph = StateGraph(State)

# Add all nodes
graph.add_node("condition_extractor", condition_extractor_node)
graph.add_node("comorbidity_worker", comorbidity_worker)
graph.add_node("cpt_worker", cpt_worker)
graph.add_node("icd11_worker", icd11_worker)
graph.add_node("z_code_worker", z_code_worker)
graph.add_node("synthesizer", synthesizer)

# Sequential: START -> condition extraction
graph.add_edge(START, "condition_extractor")

# Parallel: condition extractor -> smart orchestrator -> multiple workers
graph.add_conditional_edges(
    "condition_extractor",
    smart_orchestrator,
    ["comorbidity_worker", "cpt_worker", "icd11_worker", "z_code_worker", "synthesizer"]
)

# All workers converge to synthesizer
graph.add_edge("comorbidity_worker", "synthesizer")
graph.add_edge("cpt_worker", "synthesizer")
graph.add_edge("icd11_worker", "synthesizer")
graph.add_edge("z_code_worker", "synthesizer")

# Final edge
graph.add_edge("synthesizer", END)

comorbidity_pipeline = graph.compile()

# ============================
# Display Graph
# ============================
try:
    graph_image = comorbidity_pipeline.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(graph_image)
    print("Workflow graph saved as workflow.png")
except Exception as e:
    print(f"Could not generate graph image: {e}")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    input_path = "Med-reports/Cardiologist-Viskin-Report-page-1.jpg"

    # Detect file type
    ext = os.path.splitext(input_path)[-1].lower()

    if not os.path.exists(input_path):
        print(f"File not found at {input_path}.")
        exit(1)
    elif ext == ".pdf":
        print(f"Extracting text from PDF: {input_path}")
        full_text = extract_text_from_pdf(input_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        print(f"Extracting text from image: {input_path}")
        full_text = extract_text_from_image(input_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf, .jpg, or .png file.")

    chunks = chunk_text(full_text)
    print(f"Extracted {len(chunks)} chunk(s) from the report.\n")

    final_reports = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}...")
        state = comorbidity_pipeline.invoke({"patient_report": chunk})
        
        final_text = state.get("final_report", "")
        if not final_text:
            final_text = "No analysis generated for this chunk."
            
        final_reports.append(f"## Chunk {i}\n{final_text}")

    # Combine and display
    combined_report = "\n\n---\n\n".join(final_reports)

    print("\n" + "="*60)
    print("FINAL MEDICAL CODING REPORT")
    print("="*60 + "\n")
    print(combined_report)
