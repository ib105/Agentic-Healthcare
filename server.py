import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import shutil
from typing import List
from dotenv import load_dotenv

from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes
from icd11_retriever import search_icd11, get_z_codes_for_condition
from report_loading import extract_text_from_pdf, extract_text_from_image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel as PydanticBaseModel, Field

load_dotenv()

app = FastAPI(title="Medical Coding MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Entity extraction schemas
class Condition(PydanticBaseModel):
    name: str = Field(description="Detected medical condition or diagnosis.")

class Procedure(PydanticBaseModel):
    name: str = Field(description="Medical procedure, service, or treatment performed.")

class HealthStatusFactor(PydanticBaseModel):
    name: str = Field(description="Health status factor (screening, history, risk factor, etc.).")

class MedicalEntityExtraction(PydanticBaseModel):
    conditions: List[Condition] = Field(description="List of diseases, diagnoses, and medical conditions.")
    procedures: List[Procedure] = Field(description="List of procedures, services, tests, and treatments.")
    health_status_factors: List[HealthStatusFactor] = Field(description="List of health status factors.")

entity_extractor = llm.with_structured_output(MedicalEntityExtraction)

class ToolCall(BaseModel):
    name: str
    arguments: dict

TOOLS = [
    {
        "name": "extract_report_text",
        "description": "Extract text from medical report (PDF or image)",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to PDF or image file"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "extract_medical_entities",
        "description": "Extract medical entities (conditions, procedures, health factors) from report text",
        "parameters": {
            "type": "object",
            "properties": {
                "report_text": {"type": "string", "description": "Medical report text"}
            },
            "required": ["report_text"]
        }
    },
    {
        "name": "find_comorbidities",
        "description": "Search for comorbidities related to a medical condition",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Medical condition name"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 3}
            },
            "required": ["condition"]
        }
    },
    {
        "name": "find_icd11_codes",
        "description": "Search ICD-11 diagnostic codes for a condition",
        "parameters": {
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Medical condition name"},
                "limit": {"type": "integer", "description": "Number of results", "default": 3}
            },
            "required": ["condition"]
        }
    },
    {
        "name": "find_cpt_codes",
        "description": "Search CPT/HCPCS codes for a medical procedure",
        "parameters": {
            "type": "object",
            "properties": {
                "procedure": {"type": "string", "description": "Medical procedure name"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 3}
            },
            "required": ["procedure"]
        }
    },
    {
        "name": "find_z_codes",
        "description": "Search ICD-11 Z-codes for health status factors",
        "parameters": {
            "type": "object",
            "properties": {
                "health_factor": {"type": "string", "description": "Health factor"},
                "limit": {"type": "integer", "description": "Number of results", "default": 3}
            },
            "required": ["health_factor"]
        }
    },
    {
        "name": "generate_explanation",
        "description": "Generate explanation for a medical code",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Medical entity name"},
                "code": {"type": "string", "description": "Medical code"},
                "description": {"type": "string", "description": "Code description"},
                "code_type": {"type": "string", "description": "Type: comorbidity, icd11, cpt, or z_code"}
            },
            "required": ["entity", "code", "description", "code_type"]
        }
    },
    {
        "name": "generate_batch_explanations",
        "description": "Generate explanations for multiple medical codes in a single request (optimized for API quota)",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Array of items to explain",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "Medical entity name"},
                            "code": {"type": "string", "description": "Medical code"},
                            "description": {"type": "string", "description": "Code description"},
                            "code_type": {"type": "string", "description": "Type: comorbidity, icd11, cpt, or z_code"}
                        },
                        "required": ["entity", "code", "description", "code_type"]
                    }
                }
            },
            "required": ["items"]
        }
    }
]

@app.get("/")
def root():
    return {"message": "Medical Coding MCP Server", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/tools")
def list_tools():
    return TOOLS

@app.post("/call-tool")
def call_tool(tool_call: ToolCall):
    name = tool_call.name
    args = tool_call.arguments
    
    try:
        if name == "extract_report_text":
            result = extract_report_text(**args)
        elif name == "extract_medical_entities":
            result = extract_medical_entities(**args)
        elif name == "find_comorbidities":
            result = find_comorbidities(**args)
        elif name == "find_icd11_codes":
            result = find_icd11_codes(**args)
        elif name == "find_cpt_codes":
            result = find_cpt_codes(**args)
        elif name == "find_z_codes":
            result = find_z_codes(**args)
        elif name == "generate_explanation":
            result = generate_explanation(**args)
        elif name == "generate_batch_explanations":
            result = generate_batch_explanations(**args)
        else:
            return JSONResponse({"error": f"Unknown tool: {name}"}, status_code=400)
        
        return {"result": result}
    except Exception as e:
        import traceback
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc()}, 
            status_code=500
        )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing"""
    try:
        uploads_dir = "/app/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Using timestamp to avoid conflicts
        import time
        timestamp = int(time.time() * 1000)
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"upload_{timestamp}{file_extension}"
        
        file_path = os.path.join(uploads_dir, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File uploaded successfully: {file_path}")
        return {"file_path": file_path, "filename": safe_filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ===== Tool implementations =====
def extract_report_text(file_path: str) -> str:
    """Extract text from PDF or image file"""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return extract_text_from_image(file_path)
        else:
            return "Error: Unsupported file format. Use PDF, JPG, or PNG."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_medical_entities(report_text: str) -> str:
    """Extract medical entities from report text using LLM"""
    try:
        result = entity_extractor.invoke([
            SystemMessage(content="""You are a clinical NLP model that extracts three types of entities from medical reports:
1. **Conditions**: Diseases, diagnoses, medical conditions (e.g., diabetes, hypertension, fracture)
2. **Procedures**: Medical procedures, services, tests, treatments (e.g., MRI scan, blood test, surgery)
3. **Health Status Factors**: Screening, preventive care, family history, risk factors, follow-up visits (e.g., "screening for cancer", "family history of diabetes")

Extract all relevant entities from the report. Even if the text seems to be a template or form, extract any medical terms, conditions, or procedures mentioned."""),
            HumanMessage(content=f"Patient Report:\n\n{report_text}")
        ])
        
        # Convert to JSON format
        entities = {
            "conditions": [c.name for c in result.conditions] if result.conditions else [],
            "procedures": [p.name for p in result.procedures] if result.procedures else [],
            "health_factors": [h.name for h in result.health_status_factors] if result.health_status_factors else []
        }
        
        return json.dumps(entities)
    except Exception as e:
        error_msg = f"Error extracting entities: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return json.dumps({
            "conditions": [], 
            "procedures": [], 
            "health_factors": [], 
            "error": str(e)
        })

def find_comorbidities(condition: str, top_k: int = 3) -> str:
    """Search for comorbidities"""
    try:
        results = search_comorbidities(condition, k=top_k)
        if not results:
            return f"No comorbidities found for {condition}"
        
        output = f"Comorbidities for {condition}:\n"
        for doc, distance in results:
            name = doc.metadata.get("condition", "Unknown")
            icd10 = doc.metadata.get("icd10", "N/A")
            output += f"- {name} (ICD-10: {icd10}, Distance: {distance:.3f})\n"
        return output
    except Exception as e:
        return f"Error searching comorbidities: {str(e)}"

def find_icd11_codes(condition: str, limit: int = 3) -> str:
    """Search ICD-11 codes"""
    try:
        results = search_icd11(condition, limit=limit, filter_blocks=True)
        if not results:
            return f"No ICD-11 codes found for {condition}"
        
        output = f"ICD-11 codes for {condition}:\n"
        for item in results:
            output += f"- {item['code']}: {item['title']}\n"
        return output
    except Exception as e:
        return f"Error searching ICD-11: {str(e)}"

def find_cpt_codes(procedure: str, top_k: int = 3) -> str:
    """Search CPT codes"""
    try:
        results = search_cpt_codes(procedure, k=top_k)
        if not results:
            return f"No CPT codes found for {procedure}"
        
        output = f"CPT codes for {procedure}:\n"
        for item in results:
            output += f"- {item['code']}: {item['description']}\n"
        return output
    except Exception as e:
        return f"Error searching CPT codes: {str(e)}"

def find_z_codes(health_factor: str, limit: int = 3) -> str:
    """Search Z-codes"""
    try:
        results = get_z_codes_for_condition(health_factor, limit=limit)
        if not results:
            return f"No Z-codes found for {health_factor}"
        
        output = f"Z-codes for {health_factor}:\n"
        for item in results:
            output += f"- {item['code']}: {item['title']}\n"
        return output
    except Exception as e:
        return f"Error searching Z-codes: {str(e)}"

def generate_explanation(entity: str, code: str, description: str, code_type: str) -> str:
    """Generate explanation for a medical code"""
    try:
        prompts = {
            'comorbidity': f"In 2-3 concise lines, explain why {entity} and {description} (ICD-10: {code}) commonly co-occur.",
            'icd11': f"In 2-3 concise lines, explain why ICD-11 code {code} ({description}) is used for {entity}.",
            'cpt': f"In 2-3 concise lines, explain what CPT code {code} ({description}) involves for {entity}.",
            'z_code': f"In 2-3 concise lines, explain what Z-code {code} ({description}) represents for {entity}."
        }
        
        prompt = prompts.get(code_type, f"Explain the relationship between {entity} and code {code}.")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

def generate_batch_explanations(items: List[dict]) -> str:
    """
    Generate explanations for multiple medical codes in a SINGLE Gemini API call.
    This drastically reduces API quota usage from N calls to 1 call.
    """
    try:
        if not items:
            return json.dumps({"explanations": []})
        
        print(f"Generating batch explanations for {len(items)} items")
        
        # Comprehensive prompt for all items
        prompt = """Generate concise medical explanations (2-3 sentences each) for the following codes. 
Return your response as a JSON array with objects containing 'entity', 'code', and 'explanation' fields.

Items to explain:

"""
        
        for i, item in enumerate(items, 1):
            entity = item.get('entity', 'Unknown')
            code = item.get('code', 'N/A')
            description = item.get('description', 'Unknown')
            code_type = item.get('code_type', 'unknown')
            
            if code_type == 'comorbidity':
                prompt += f"{i}. Entity: {entity}\n   Code: {code} (ICD-10)\n   Related Condition: {description}\n   Task: Explain why these conditions commonly co-occur.\n\n"
            elif code_type == 'icd11':
                prompt += f"{i}. Entity: {entity}\n   Code: {code} (ICD-11)\n   Description: {description}\n   Task: Explain why this diagnostic code is used for this condition.\n\n"
            elif code_type == 'cpt':
                prompt += f"{i}. Entity: {entity}\n   Code: {code} (CPT)\n   Procedure: {description}\n   Task: Explain what this procedure involves.\n\n"
            elif code_type == 'z_code':
                prompt += f"{i}. Entity: {entity}\n   Code: {code} (Z-code)\n   Description: {description}\n   Task: Explain what this health status code represents.\n\n"
        
        prompt += """\nProvide explanations in this exact JSON format:
{
  "explanations": [
    {
      "entity": "condition/procedure name",
      "code": "the code",
      "explanation": "2-3 sentence explanation"
    }
  ]
}"""
        
        # Single Gemini API call for all explanations
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()
        
        # Try to extract JSON from response
        try:
            # Removing markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            
            # Validate structure
            if 'explanations' not in result:
                raise ValueError("Response missing 'explanations' field")
            
            # Ensure all items have explanations
            explanations_dict = {f"{exp['entity']}|{exp['code']}": exp for exp in result['explanations']}
            
            # Fill in any missing explanations
            final_explanations = []
            for item in items:
                entity = item.get('entity', 'Unknown')
                code = item.get('code', 'N/A')
                key = f"{entity}|{code}"
                
                if key in explanations_dict:
                    final_explanations.append(explanations_dict[key])
                else:
                    # fallback
                    final_explanations.append({
                        "entity": entity,
                        "code": code,
                        "explanation": f"This code is associated with {entity}."
                    })
            
            result['explanations'] = final_explanations
            print(f"Successfully generated {len(final_explanations)} explanations")
            return json.dumps(result)
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text[:500]}")
            
            # Fallback: Create simple explanations
            fallback_explanations = []
            for item in items:
                fallback_explanations.append({
                    "entity": item.get('entity', 'Unknown'),
                    "code": item.get('code', 'N/A'),
                    "explanation": f"Medical code {item.get('code', 'N/A')} is associated with {item.get('entity', 'this condition')}."
                })
            
            return json.dumps({"explanations": fallback_explanations})
            
    except Exception as e:
        print(f"Error in batch explanation generation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return fallback explanations
        fallback_explanations = []
        for item in items:
            fallback_explanations.append({
                "entity": item.get('entity', 'Unknown'),
                "code": item.get('code', 'N/A'),
                "explanation": f"Explanation unavailable for {item.get('code', 'N/A')}."
            })
        
        return json.dumps({"explanations": fallback_explanations})

if __name__ == "__main__":
    print("Starting Medical Coding MCP Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
