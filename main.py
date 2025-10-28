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
    name: str = Field(description="Detected medical condition or diagnosis.")
    description: str | None = Field(default=None, description="Short note or context about the condition.")

class Procedure(BaseModel):
    name: str = Field(description="Medical procedure, service, or treatment performed.")
    description: str | None = Field(default=None, description="Context about the procedure.")

class HealthStatusFactor(BaseModel):
    name: str = Field(description="Health status factor (screening, history, risk factor, etc.).")
    description: str | None = Field(default=None, description="Context about the factor.")

class MedicalEntityExtraction(BaseModel):
    conditions: List[Condition] = Field(description="List of diseases, diagnoses, and medical conditions.")
    procedures: List[Procedure] = Field(description="List of procedures, services, tests, and treatments.")
    health_status_factors: List[HealthStatusFactor] = Field(description="List of health status factors (screening, history, preventive care, etc.).")
    highlighted_report: str = Field(description="Original report with entities highlighted: **conditions**, __procedures__, and *health_status_factors*.")

entity_extractor = llm.with_structured_output(MedicalEntityExtraction)

# ============================
# State Definitions
# ============================
class State(TypedDict):
    patient_report: str
    extracted_conditions: list[str]
    extracted_procedures: list[str]
    extracted_health_factors: list[str]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class ConditionWorkerState(TypedDict):
    patient_report: str
    extracted_conditions: list[str]

class ProcedureWorkerState(TypedDict):
    patient_report: str
    extracted_procedures: list[str]

class HealthFactorWorkerState(TypedDict):
    patient_report: str
    extracted_health_factors: list[str]

# ============================
# Nodes
# ============================
def entity_extractor_node(state: State):
    """Extracts conditions, procedures, and health status factors from patient report."""
    result = entity_extractor.invoke([
        SystemMessage(content="""You are a clinical NLP model that extracts three types of entities from medical reports:
1. **Conditions**: Diseases, diagnoses, medical conditions (e.g., diabetes, hypertension, fracture)
2. **Procedures**: Medical procedures, services, tests, treatments (e.g., MRI scan, blood test, surgery, chemotherapy)
3. **Health Status Factors**: Screening, preventive care, family history, risk factors, follow-up visits, vaccination status (e.g., "screening for cancer", "family history of diabetes")

Highlight them in the text using: **bold for conditions**, __underline for procedures__, and *italics for health status factors*."""),
        HumanMessage(content=f"Patient Report:\n\n{state['patient_report']}")
    ])

    extracted_conditions = [c.name for c in result.conditions] if result.conditions else []
    extracted_procedures = [p.name for p in result.procedures] if result.procedures else []
    extracted_health_factors = [h.name for h in result.health_status_factors] if result.health_status_factors else []

    summary_parts = ["### Extracted Medical Entities\n"]
    
    if extracted_conditions:
        summary_parts.append("**Conditions/Diagnoses:**\n" + "\n".join([f"- {c}" for c in extracted_conditions]))
    
    if extracted_procedures:
        summary_parts.append("\n**Procedures/Services:**\n" + "\n".join([f"- {p}" for p in extracted_procedures]))
    
    if extracted_health_factors:
        summary_parts.append("\n**Health Status Factors:**\n" + "\n".join([f"- {h}" for h in extracted_health_factors]))
    
    summary_parts.append("\n### Highlighted Report:\n" + result.highlighted_report)
    
    summary_text = "\n".join(summary_parts)

    return {
        "completed_sections": [summary_text], 
        "extracted_conditions": extracted_conditions,
        "extracted_procedures": extracted_procedures,
        "extracted_health_factors": extracted_health_factors
    }


def comorbidity_worker(state: ConditionWorkerState):
    """Worker: Finds potential comorbidities for extracted conditions."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["### Comorbidity Analysis\nNo medical conditions extracted for comorbidity search."]}

    comorbidity_summary = ["### Comorbidity Analysis"]

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


def icd11_worker(state: ConditionWorkerState):
    """Worker: Retrieves ICD-11 diagnostic codes for conditions."""
    if not state.get("extracted_conditions"):
        return {"completed_sections": ["### ICD-11 Diagnostic Codes\nNo conditions extracted for ICD-11 lookup."]}

    icd_sections = ["### ICD-11 Diagnostic Codes"]

    for condition in state["extracted_conditions"]:
        icd_sections.append(f"\n#### {condition}")
        results = search_icd11(condition, limit=3, filter_blocks=True)

        if not results:
            icd_sections.append("- No matching ICD-11 codes found.")
            continue

        # Get best match
        best_match = results[0]
        icd_sections.append(f"- **Primary:** {best_match['code']} — {best_match['title']}")
        
        if best_match.get("uri"):
            detail = fetch_icd11_details(best_match["uri"])
            if detail and detail.get("definition"):
                icd_sections.append(f"  *{detail['definition'][:200]}...*")
        
        # Add alternative if significantly different
        if len(results) > 1:
            alt = results[1]
            if alt['code'][:3] != best_match['code'][:3]:
                icd_sections.append(f"- **Alternative:** {alt['code']} — {alt['title']}")

    return {"completed_sections": ["\n".join(icd_sections)]}


def cpt_worker(state: ProcedureWorkerState):
    """Worker: Finds CPT/HCPCS codes for extracted procedures and services."""
    if not state.get("extracted_procedures"):
        return {"completed_sections": ["### CPT/HCPCS Procedure Codes\nNo procedures or services extracted for CPT code analysis."]}

    cpt_summary = ["### CPT/HCPCS Procedure Codes"]

    for procedure in state["extracted_procedures"]:
        cpt_summary.append(f"\n#### {procedure}")
        results = search_cpt_codes(procedure, k=5)

        if not results:
            cpt_summary.append("- No related CPT codes found.")
            continue

        top_codes = [
            f"{item['code']} — {item['description']} (similarity: {item['similarity']})"
            for item in results
        ]
        formatted_codes = "\n".join([f"- {c}" for c in top_codes])

        prompt = f"""
You are a clinical coding expert. Analyze these retrieved CPT codes for the procedure: "{procedure}".

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


def z_code_worker(state: HealthFactorWorkerState):
    """Worker: Retrieves ICD-11 Z-codes for health status factors."""
    if not state.get("extracted_health_factors"):
        return {"completed_sections": ["### ICD-11 Z-Codes (Health Status)\nNo health status factors extracted for Z-code analysis."]}

    z_sections = ["### ICD-11 Z-Codes (Health Status Factors)"]

    for factor in state["extracted_health_factors"]:
        z_sections.append(f"\n#### {factor}")
        results = get_z_codes_for_condition(factor, limit=2)

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
# Orchestrator (Parallel Routing)
# ============================
def orchestrator(state: State):
    """
    Routes extracted entities to appropriate workers for parallel execution.
    """
    workers_to_run = []
    
    # Comorbidity + ICD-11 for conditions
    if state.get("extracted_conditions"):
        workers_to_run.extend([
            Send("comorbidity_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": state["extracted_conditions"]
            }),
            Send("icd11_worker", {
                "patient_report": state["patient_report"],
                "extracted_conditions": state["extracted_conditions"]
            })
        ])
    
    # CPT codes for procedures
    if state.get("extracted_procedures"):
        workers_to_run.append(
            Send("cpt_worker", {
                "patient_report": state["patient_report"],
                "extracted_procedures": state["extracted_procedures"]
            })
        )
    
    # Z-codes for health status factors
    if state.get("extracted_health_factors"):
        workers_to_run.append(
            Send("z_code_worker", {
                "patient_report": state["patient_report"],
                "extracted_health_factors": state["extracted_health_factors"]
            })
        )
    
    # If no workers selected, proceed to synthesizer
    if not workers_to_run:
        return [Send("synthesizer", state)]
    
    return workers_to_run


# ============================
# Graph Assembly
# ============================
graph = StateGraph(State)

# Add all nodes
graph.add_node("entity_extractor", entity_extractor_node)
graph.add_node("comorbidity_worker", comorbidity_worker)
graph.add_node("icd11_worker", icd11_worker)
graph.add_node("cpt_worker", cpt_worker)
graph.add_node("z_code_worker", z_code_worker)
graph.add_node("synthesizer", synthesizer)

# Sequential: START -> entity extraction
graph.add_edge(START, "entity_extractor")

# Parallel: entity extractor -> orchestrator -> multiple workers
graph.add_conditional_edges(
    "entity_extractor",
    orchestrator,
    ["comorbidity_worker", "icd11_worker", "cpt_worker", "z_code_worker", "synthesizer"]
)

# All workers converge to synthesizer
graph.add_edge("comorbidity_worker", "synthesizer")
graph.add_edge("icd11_worker", "synthesizer")
graph.add_edge("cpt_worker", "synthesizer")
graph.add_edge("z_code_worker", "synthesizer")

# Final edge
graph.add_edge("synthesizer", END)

medical_coding_pipeline = graph.compile()

# ============================
# Display Graph
# ============================
try:
    graph_image = medical_coding_pipeline.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(graph_image)
    print("Workflow graph saved as workflow.png")
except Exception as e:
    print(f"Could not generate graph image: {e}")

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    input_path = "Med-reports/Sample-filled-in-MR.pdf"

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
        state = medical_coding_pipeline.invoke({"patient_report": chunk})
        
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
