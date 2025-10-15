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

# Import search function from validation.py
from validation import search_comorbidities


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
            for r in results:
                comorbidity_summary.append(
                    f"- {r.metadata.get('condition', 'Unknown')} (ICD-10: {r.metadata.get('icd10', 'N/A')})"
                )

    return {"completed_sections": ["\n".join(comorbidity_summary)]}


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
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_workers, ["condition_extractor_worker"])
graph.add_edge("condition_extractor_worker", "comorbidity_worker")
graph.add_edge("comorbidity_worker", "synthesizer")
graph.add_edge("synthesizer", END)

comorbidity_pipeline = graph.compile()

# ============================
# Display Graph
# ============================
graph_image = comorbidity_pipeline.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(graph_image)
print("Workflow graph saved as workflow.png")

# ============================
# Example Run
# ============================
sample_report = """
Mr Tan Ah Kow has a history of medical conditions. He has had hypertension and
hyperlipidemia since 1990 and suffered several strokes in 2005. He subsequently
developed heart problems (cardiomyopathy), cardiac failure and chronic renal disease
and was treated in ABC Hospital.
"""

state = comorbidity_pipeline.invoke({"patient_report": sample_report})
final_text = "\n\n".join(state["final_report"]) if isinstance(state["final_report"], list) else state["final_report"]
print("\n========== FINAL REPORT ==========\n")
print(final_text)
