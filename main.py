import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Annotated, List
import operator

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
class Comorbidity(BaseModel):
    condition: str = Field(description="Comorbid condition name.")
    relation: str = Field(description="Relation to the primary diagnosis.")

class ComorbidityAnalysis(BaseModel):
    comorbidities: List[Comorbidity] = Field(description = "List of detected comorbidities.")
    highlighted_report: str = Field(description = "Original report text with comorbidities highlighted using **Markdown bold**")

# ============================
# Structured LLM for Comorbidities
# ============================
comorbidity_detector = llm.with_structured_output(ComorbidityAnalysis)

# ============================
# LangGraph Setup
# ============================
from langgraph.types import Send
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ----------------------------
# State Definitions
# ----------------------------
class State(TypedDict):
    patient_report: str
    tasks: list[str]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    task: str
    patient_report: str
    completed_sections: Annotated[list, operator.add]

# ----------------------------
# Nodes
# ----------------------------
def orchestrator(state: State):
    """Orchestrator defines one comorbidity detection task."""
    tasks = ["Detect and highlight comorbidities in the patient report."]
    return {"tasks" : tasks}

def comorbidity_worker(state: WorkerState):
    """Worker detects and highlights comorbidities from a report."""
    result = comorbidity_detector.invoke([
        SystemMessage(content = ("You are a clinical NLP model that extracts comorbidities from medical reports and highlight them using **Markdown bold** in the text.")),
        HumanMessage(content = f"Patient Report:\n\n{state['patient_report']}")
    ])

    # Convert structured data to readable text for synthesizer
    if not result.comorbidities:
        summary_text = "No comorbidities detected."
    else:
        summary_text = "### Detected Comorbidites\n" + "\n".join(
            [f"- **{c.condition}** ({c.relation})" for c in result.comorbidities]
        )
        summary_text += "\n\n### Highlighted report:\n" + result.highlighted_report

    return {"completed_sections" : [summary_text]}

def synthesizer(state: State):
    """Combine worker results into a final report."""
    combined_text = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report" : [combined_text]}

# ----------------------------
# Conditional Edge Function
# ----------------------------
def assign_workers(state: State):
    """Assign a single worker for the comorbidity task."""
    return [Send("comorbidity_worker", {"task": t, "patient_report": state["patient_report"]}) for t in state["tasks"]]

# ----------------------------
# Graph Assembly
# ----------------------------
graph = StateGraph(State)

graph.add_node("orchestrator", orchestrator)
graph.add_node("comorbidity_worker", comorbidity_worker)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", assign_workers, ["comorbidity_worker"])
graph.add_edge("comorbidity_worker", "synthesizer")
graph.add_edge("synthesizer", END)

comorbidity_pipeline = graph.compile()

# ============================
# Display Graph
# ============================
# Save the graph instead of displaying (for terminal users)
graph_image = comorbidity_pipeline.get_graph().draw_mermaid_png()
with open("workflow.png", "wb") as f:
    f.write(graph_image)
print("Workflow graph saved as workflow.png")

# ============================
# Example Run
# ============================
sample_report = """
The patient has type 2 diabetes mellitus for 10 years.
Also suffers from hypertension, obesity, and chronic kidney disease stage 2.
There is a history of mild depression and sleep apnea.
"""

state = comorbidity_pipeline.invoke({"patient_report": sample_report})

final_text = state["final_report"]

# If it's a list (Langgraph merges lists), flatten it to string
if isinstance(final_text, list):
    final_text = "\n\n".join(final_text)

print(final_text)
