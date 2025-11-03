import streamlit as st
import os
import tempfile
from PIL import Image, ImageDraw
import fitz
import io
import base64
from typing import List, Dict, Tuple
import pytesseract
from pytesseract import Output
import concurrent.futures
from functools import lru_cache
import re
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field

load_dotenv()

from client import MCPGeminiClient

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Initialize MCP client
@st.cache_resource
def get_mcp_client():
    """Initialize MCP client with proper URL for Docker environment"""
    mcp_url = os.getenv("MCP_SERVER_URL", "http://server:8000")
    return MCPGeminiClient(mcp_url=mcp_url)

# ===== Helper function to call MCP tools =====
def call_mcp_tool(tool_name: str, arguments: dict):
    """Call MCP server tool via HTTP"""
    mcp_url = os.getenv("MCP_SERVER_URL", "http://server:8000")
    try:
        response = requests.post(
            f"{mcp_url}/call-tool",
            json={"name": tool_name, "arguments": arguments},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["result"]
    except Exception as e:
        st.error(f"MCP tool call failed: {e}")
        return None

# ===== Entity extraction via MCP =====
def extract_entities_mcp(report_text: str) -> dict:
    """Extract medical entities via MCP server"""
    
    result = call_mcp_tool("extract_medical_entities", {"report_text": report_text})
    
    if not result:
        st.error("MCP tool returned no result")
        return {"conditions": [], "procedures": [], "health_factors": []}
    
    # Parse the response
    try:
        import json
        if isinstance(result, str):
            # Parse JSON string
            data = json.loads(result)
        else:
            data = result
        
        # Check for error in response
        if "error" in data:
            st.error(f"Entity extraction error: {data['error']}")
            
        entities = {
            "conditions": data.get("conditions", []),
            "procedures": data.get("procedures", []),
            "health_factors": data.get("health_factors", [])
        }
        
        st.write(f"Extracted entities: {entities}")
        return entities
        
    except Exception as e:
        st.error(f"Error parsing entity extraction result: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {"conditions": [], "procedures": [], "health_factors": []}

# ===== Extract text using MCP =====
def extract_report_text(file_path: str) -> str:
    """Extract text from PDF or image via MCP server"""
    result = call_mcp_tool("extract_report_text", {"file_path": file_path})
    return result if result else ""

# ===== Search functions using MCP =====
def search_comorbidities_mcp(condition: str, k: int = 3):
    """Search comorbidities via MCP"""
    result = call_mcp_tool("find_comorbidities", {"condition": condition, "top_k": k})
    if not result or "No comorbidities found" in result:
        return []
    
    comorbidities = []
    lines = result.split('\n')[1:]  # Skip header
    for line in lines:
        if line.strip().startswith('-'):
            try:
                match = re.search(r'- (.+?) \(ICD-10: (.+?), Distance: ([\d.]+)\)', line)
                if match:
                    comorbidities.append({
                        'name': match.group(1),
                        'icd10': match.group(2),
                        'distance': float(match.group(3))
                    })
            except:
                pass
    return comorbidities

def search_icd11_mcp(condition: str, limit: int = 3):
    """Search ICD-11 codes via MCP"""
    result = call_mcp_tool("find_icd11_codes", {"condition": condition, "limit": limit})
    if not result or "No ICD-11 codes found" in result:
        return []
    
    codes = []
    lines = result.split('\n')[1:]  # Skip header
    for line in lines:
        if line.strip().startswith('-'):
            try:
                match = re.search(r'- (.+?): (.+)', line)
                if match:
                    codes.append({
                        'code': match.group(1),
                        'title': match.group(2)
                    })
            except:
                pass
    return codes

def search_cpt_mcp(procedure: str, k: int = 3):
    """Search CPT codes via MCP"""
    result = call_mcp_tool("find_cpt_codes", {"procedure": procedure, "top_k": k})
    if not result or "No CPT codes found" in result:
        return []
    
    codes = []
    lines = result.split('\n')[1:]  # Skip header
    for line in lines:
        if line.strip().startswith('-'):
            try:
                match = re.search(r'- (.+?): (.+)', line)
                if match:
                    codes.append({
                        'code': match.group(1),
                        'description': match.group(2)
                    })
            except:
                pass
    return codes

def search_z_codes_mcp(health_factor: str, limit: int = 3):
    """Search Z-codes via MCP"""
    result = call_mcp_tool("find_z_codes", {"health_factor": health_factor, "limit": limit})
    if not result or "No Z-codes found" in result:
        return []
    
    codes = []
    lines = result.split('\n')[1:]  # Skip header
    for line in lines:
        if line.strip().startswith('-'):
            try:
                match = re.search(r'- (.+?): (.+)', line)
                if match:
                    codes.append({
                        'code': match.group(1),
                        'title': match.group(2)
                    })
            except:
                pass
    return codes

# ===== Generate explanations via MCP =====
def generate_explanation_mcp(entity: str, code: str, description: str, code_type: str) -> str:
    """Generate explanation for a single code via MCP"""
    result = call_mcp_tool("generate_explanation", {
        "entity": entity,
        "code": code,
        "description": description,
        "code_type": code_type
    })
    return result if result else f"No explanation available for {code}."

# ===== Cache OCR results =====
@st.cache_data
def run_ocr_on_image(image_bytes: bytes) -> Dict:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')

# ===== Batch explanations =====
@st.cache_data(ttl=3600)
def get_batch_explanations(entities_with_codes: List[Tuple[str, str, str, str]]) -> Dict[str, str]:
    """Get explanations for multiple codes"""
    explanations = {}
    for entity, code, description, code_type in entities_with_codes:
        key = f"{entity}|{code}"
        explanations[key] = generate_explanation_mcp(entity, code, description, code_type)
    return explanations

# ===== Entity normalization =====
def normalize_entity(entity: str) -> List[str]:
    """Split complex entities into simpler searchable terms"""
    variants = [entity]
    
    if '(' in entity:
        base = re.sub(r'\([^)]*\)', '', entity).strip()
        inside = re.findall(r'\(([^)]*)\)', entity)
        variants.append(base)
        variants.extend(inside)
    
    if ' and ' in entity.lower():
        parts = re.split(r'\s+and\s+', entity, flags=re.IGNORECASE)
        variants.extend(parts)
    
    if 'history of' in entity.lower():
        without_history = re.sub(r'history of\s+', '', entity, flags=re.IGNORECASE).strip()
        variants.append(without_history)
        words = entity.split()
        if len(words) > 3:
            variants.append(' '.join(words[-2:]))
    
    if len(entity.split()) == 1:
        variants.append(entity.lower())
        variants.append(entity.capitalize())
    
    variants = [' '.join(v.split()) for v in variants if v.strip()]
    return list(set(variants))

def convert_pdf_to_image(pdf_path: str) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_data))

def create_text_index(ocr_data: Dict) -> Dict[str, List[int]]:
    index = {}
    for i, text in enumerate(ocr_data['text']):
        word = text.lower().strip()
        if word:
            if word not in index:
                index[word] = []
            index[word].append(i)
    return index

def find_entity_positions_optimized(entity: str, text_index: Dict, ocr_data: Dict) -> List[int]:
    """Find entity allowing for surrounding context and punctuation"""
    variants = normalize_entity(entity)
    
    for variant in variants:
        words = variant.lower().split()
        if not words:
            continue
        
        first_word = words[0].strip('.,;:!?')
        first_word_positions = text_index.get(first_word, [])
        
        for pos in first_word_positions:
            match = True
            matched_length = 0
            
            for offset, word in enumerate(words):
                if pos + offset >= len(ocr_data['text']):
                    match = False
                    break
                
                ocr_word = ocr_data['text'][pos + offset].lower().strip('.,;:!?')
                search_word = word.strip('.,;:!?')
                
                if ocr_word == search_word:
                    matched_length += 1
                elif offset == 0:
                    match = False
                    break
            
            if match and matched_length >= len(words):
                return [pos]
    
    return []

def find_and_highlight(image: Image.Image, ocr_data: Dict, conditions: List[str], 
                      procedures: List[str], health_factors: List[str]) -> Tuple[Image.Image, Dict]:
    text_index = create_text_index(ocr_data)
    img_with_highlights = image.copy()
    draw = ImageDraw.Draw(img_with_highlights, 'RGBA')
    
    img_width, img_height = image.size
    positions = {'conditions': {}, 'procedures': {}, 'health_factors': {}}
    
    colors = {
        'conditions': {'fill': (255, 100, 100, 80), 'outline': (220, 20, 60, 255)},
        'procedures': {'fill': (100, 149, 237, 80), 'outline': (65, 105, 225, 255)},
        'health_factors': {'fill': (144, 238, 144, 80), 'outline': (34, 139, 34, 255)}
    }
    
    def highlight_entities(entities: List[str], entity_type: str):
        for entity in entities:
            positions[entity_type][entity] = []
            matched_positions = find_entity_positions_optimized(entity, text_index, ocr_data)
            word_count = len(entity.split())
            
            for start_pos in matched_positions:
                boxes_indices = list(range(start_pos, min(start_pos + word_count, len(ocr_data['text']))))
                xs = [ocr_data['left'][idx] for idx in boxes_indices]
                ys = [ocr_data['top'][idx] for idx in boxes_indices]
                ws = [ocr_data['width'][idx] for idx in boxes_indices]
                hs = [ocr_data['height'][idx] for idx in boxes_indices]
                
                x = min(xs)
                y = min(ys)
                w = max(xs[j] + ws[j] for j in range(len(xs))) - x
                h = max(ys[j] + hs[j] for j in range(len(ys))) - y
                
                draw.rectangle([x, y, x+w, y+h], 
                             fill=colors[entity_type]['fill'], 
                             outline=colors[entity_type]['outline'], 
                             width=2)
                
                x_pct = (x / img_width) * 100
                y_pct = (y / img_height) * 100
                w_pct = (w / img_width) * 100
                h_pct = (h / img_height) * 100
                
                positions[entity_type][entity].append((x_pct, y_pct, w_pct, h_pct))
    
    highlight_entities(conditions, 'conditions')
    highlight_entities(procedures, 'procedures')
    highlight_entities(health_factors, 'health_factors')
    
    return img_with_highlights, positions

# ===== Fetch all codes via MCP in parallel =====
def fetch_all_codes_parallel(conditions: List[str], procedures: List[str], health_factors: List[str]) -> Tuple[Dict, List]:
    entity_data = {'conditions': {}, 'procedures': {}, 'health_factors': {}}
    codes_for_explanation = []
    
    def fetch_condition_data(condition):
        comorbidities = search_comorbidities_mcp(condition, k=3)
        icd11 = search_icd11_mcp(condition, limit=3)
        
        for item in comorbidities:
            codes_for_explanation.append((condition, item['icd10'], item['name'], 'comorbidity'))
        
        for item in icd11:
            codes_for_explanation.append((condition, item['code'], item['title'], 'icd11'))
        
        return condition, {'comorbidities': comorbidities, 'icd11': icd11}
    
    def fetch_procedure_data(procedure):
        cpt = search_cpt_mcp(procedure, k=3)
        
        for item in cpt:
            codes_for_explanation.append((procedure, item['code'], item['description'], 'cpt'))
        
        return procedure, {'cpt': cpt}
    
    def fetch_health_factor_data(factor):
        z_codes = search_z_codes_mcp(factor, limit=3)
        
        for item in z_codes:
            codes_for_explanation.append((factor, item['code'], item['title'], 'z_code'))
        
        return factor, {'z_codes': z_codes}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        condition_futures = [executor.submit(fetch_condition_data, c) for c in conditions]
        procedure_futures = [executor.submit(fetch_procedure_data, p) for p in procedures]
        factor_futures = [executor.submit(fetch_health_factor_data, f) for f in health_factors]
        
        for future in concurrent.futures.as_completed(condition_futures):
            try:
                name, data = future.result()
                entity_data['conditions'][name] = data
            except:
                pass
        
        for future in concurrent.futures.as_completed(procedure_futures):
            try:
                name, data = future.result()
                entity_data['procedures'][name] = data
            except:
                pass
        
        for future in concurrent.futures.as_completed(factor_futures):
            try:
                name, data = future.result()
                entity_data['health_factors'][name] = data
            except:
                pass
    
    return entity_data, codes_for_explanation

def upload_file_to_mcp(file_bytes, filename):
    """Upload file to MCP server and get the server-side path"""
    mcp_url = os.getenv("MCP_SERVER_URL", "http://server:8000")
    try:
        files = {"file": (filename, file_bytes)}
        response = requests.post(f"{mcp_url}/upload", files=files, timeout=30)
        response.raise_for_status()
        return response.json()["file_path"]
    except Exception as e:
        st.error(f"File upload failed: {e}")
        return None

def create_interactive_html(image_b64: str, positions: Dict, entity_data: Dict, explanations: Dict) -> str:
    highlights = ""
    modals = ""
    detail_modals = ""
    
    border_colors = {'conditions': '#DC143C', 'procedures': '#4169E1', 'health_factors': '#228B22'}
    
    for entity_type, entities in positions.items():
        for idx, (entity, boxes) in enumerate(entities.items()):
            for box_idx, (x, y, w, h) in enumerate(boxes):
                mid = f"m{entity_type}{idx}{box_idx}"
                border_color = border_colors[entity_type]
                
                highlights += f'<div class="hl" onclick="show(\'{mid}\')" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;border-color:{border_color};"></div>'
                
                entity_info = entity_data.get(entity_type, {}).get(entity, {})
                modal_tabs = ""
                modal_content = ""
                
                if entity_type == "conditions":
                    comorbidities = entity_info.get('comorbidities', [])[:3]
                    comorbid_buttons = ""
                    
                    if comorbidities:
                        for c_idx, item in enumerate(comorbidities):
                            cid = f"d{entity_type}{idx}{box_idx}{c_idx}"
                            name = item.get('name', 'Unknown')
                            icd10 = item.get('icd10', 'N/A')
                            distance = item.get('distance', 0)
                            
                            comorbid_buttons += f'<button class="code-btn" onclick="show(\'{cid}\')">{name}<br><small>Distance: {distance:.3f}</small></button>'
                            
                            explanation = explanations.get(f"{entity}|{icd10}", f"Related comorbidity for {entity}.")
                            
                            detail_modals += f'''<div id="{cid}" class="smd" onclick="hide('{cid}')">
                                <div class="smc" onclick="event.stopPropagation()">
                                    <span class="x" onclick="hide('{cid}')">&times;</span>
                                    <h3>{name}</h3>
                                    <p>{explanation}</p>
                                </div>
                            </div>'''
                    else:
                        comorbid_buttons = "<p style='color:#999;'>No comorbidities found</p>"
                    
                    icd11_codes = entity_info.get('icd11', [])[:3]
                    icd11_buttons = ""
                    
                    if icd11_codes:
                        for i_idx, icd_item in enumerate(icd11_codes):
                            iid = f"i{entity_type}{idx}{box_idx}{i_idx}"
                            code = icd_item.get('code', 'N/A')
                            title = icd_item.get('title', 'Unknown')
                            
                            icd11_buttons += f'<button class="code-btn" onclick="show(\'{iid}\')">{code}<br><small>{title[:40]}...</small></button>'
                            
                            explanation = explanations.get(f"{entity}|{code}", f"ICD-11 code for {entity}.")
                            
                            detail_modals += f'''<div id="{iid}" class="smd" onclick="hide('{iid}')">
                                <div class="smc" onclick="event.stopPropagation()">
                                    <span class="x" onclick="hide('{iid}')">&times;</span>
                                    <h3>{code}</h3>
                                    <h4>{title}</h4>
                                    <p>{explanation}</p>
                                </div>
                            </div>'''
                    else:
                        icd11_buttons = "<p style='color:#999;'>No ICD-11 codes found</p>"
                    
                    modal_tabs = f'''
                        <button class="tab active" onclick="switchTab(event, 'comorbid{entity_type}{idx}{box_idx}')">Comorbidities</button>
                        <button class="tab" onclick="switchTab(event, 'icd11{entity_type}{idx}{box_idx}')">ICD-11</button>
                    '''
                    
                    modal_content = f'''
                        <div id="comorbid{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>Related Comorbidities</h3>
                            <div class="btn-grid">{comorbid_buttons}</div>
                        </div>
                        <div id="icd11{entity_type}{idx}{box_idx}" class="tab-content">
                            <h3>ICD-11 Diagnostic Codes</h3>
                            <div class="btn-grid">{icd11_buttons}</div>
                        </div>
                    '''
                
                elif entity_type == "procedures":
                    cpt_codes = entity_info.get('cpt', [])[:3]
                    cpt_buttons = ""
                    
                    if cpt_codes:
                        for p_idx, item in enumerate(cpt_codes):
                            pid = f"p{entity_type}{idx}{box_idx}{p_idx}"
                            code = item.get('code', 'N/A')
                            desc = item.get('description', 'Unknown')
                            
                            cpt_buttons += f'<button class="code-btn" onclick="show(\'{pid}\')">{code}<br><small>{desc[:40]}...</small></button>'
                            
                            explanation = explanations.get(f"{entity}|{code}", f"CPT code for {entity}.")
                            
                            detail_modals += f'''<div id="{pid}" class="smd" onclick="hide('{pid}')">
                                <div class="smc" onclick="event.stopPropagation()">
                                    <span class="x" onclick="hide('{pid}')">&times;</span>
                                    <h3>{code}</h3>
                                    <h4>{desc}</h4>
                                    <p>{explanation}</p>
                                </div>
                            </div>'''
                    else:
                        cpt_buttons = "<p style='color:#999;'>No CPT codes found</p>"
                    
                    modal_tabs = f'<button class="tab active" onclick="switchTab(event, \'cpt{entity_type}{idx}{box_idx}\')">CPT Codes</button>'
                    modal_content = f'''
                        <div id="cpt{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>CPT/HCPCS Procedure Codes</h3>
                            <div class="btn-grid">{cpt_buttons}</div>
                        </div>
                    '''
                
                elif entity_type == "health_factors":
                    z_codes = entity_info.get('z_codes', [])[:3]
                    z_buttons = ""
                    
                    if z_codes:
                        for z_idx, item in enumerate(z_codes):
                            zid = f"z{entity_type}{idx}{box_idx}{z_idx}"
                            code = item.get('code', 'N/A')
                            title = item.get('title', 'Unknown')
                            
                            z_buttons += f'<button class="code-btn" onclick="show(\'{zid}\')">{code}<br><small>{title[:40]}...</small></button>'
                            
                            explanation = explanations.get(f"{entity}|{code}", f"Z-code for {entity}.")
                            
                            detail_modals += f'''<div id="{zid}" class="smd" onclick="hide('{zid}')">
                                <div class="smc" onclick="event.stopPropagation()">
                                    <span class="x" onclick="hide('{zid}')">&times;</span>
                                    <h3>{code}</h3>
                                    <h4>{title}</h4>
                                    <p>{explanation}</p>
                                </div>
                            </div>'''
                    else:
                        z_buttons = "<p style='color:#999;'>No Z-codes found</p>"
                    
                    modal_tabs = f'<button class="tab active" onclick="switchTab(event, \'zcodes{entity_type}{idx}{box_idx}\')">Z-Codes</button>'
                    modal_content = f'''
                        <div id="zcodes{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>ICD-11 Z-Codes</h3>
                            <div class="btn-grid">{z_buttons}</div>
                        </div>
                    '''
                
                entity_label = entity_type.replace('_', ' ').title()
                modals += f'''<div id="{mid}" class="md" onclick="hide('{mid}')">
                    <div class="mc" onclick="event.stopPropagation()">
                        <span class="x" onclick="hide('{mid}')">&times;</span>
                        <div class="badge {entity_type}">{entity_label}</div>
                        <h2>{entity}</h2>
                        <div class="tabs">{modal_tabs}</div>
                        {modal_content}
                    </div>
                </div>'''
    
    return f'''<style>
    
.rc{{position:relative;display:inline-block;}}
.ri{{max-width:100%;height:auto;display:block;}}
.hl{{position:absolute;border:3px solid;cursor:pointer;transition:all 0.2s;}}
.hl:hover{{background:rgba(255,255,255,0.2);transform:scale(1.02);}}
.md,.smd{{display:none;position:fixed;z-index:999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.6);}}
.smd{{z-index:1000;}}
.mc{{background:#fff;margin:5% auto;padding:30px;border-radius:12px;width:85%;max-width:800px;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-height:85vh;overflow-y:auto;}}
.smc{{background:#fff;margin:15% auto;padding:25px;border-radius:10px;width:90%;max-width:500px;box-shadow:0 4px 20px rgba(0,0,0,0.3);}}
.x{{float:right;font-size:28px;font-weight:bold;cursor:pointer;color:#aaa;}}
.x:hover{{color:#000;}}
.badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;}}
.badge.conditions{{background:#ffebee;color:#c62828;}}
.badge.procedures{{background:#e3f2fd;color:#1565c0;}}
.badge.health_factors{{background:#e8f5e9;color:#2e7d32;}}
.tabs{{display:flex;gap:5px;margin:20px 0;border-bottom:2px solid #e9ecef;}}
.tab{{padding:10px 16px;background:none;border:none;cursor:pointer;font-size:14px;color:#666;border-bottom:3px solid transparent;transition:all 0.3s;}}
.tab:hover{{color:#007bff;background:#f8f9fa;}}
.tab.active{{color:#007bff;border-bottom-color:#007bff;font-weight:bold;}}
.tab-content{{display:none;margin-top:20px;}}
.tab-content.active{{display:block;}}
.btn-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:15px;}}
.code-btn{{padding:15px;background:#f8f9fa;border:2px solid #dee2e6;border-radius:8px;cursor:pointer;text-align:left;transition:all 0.2s;font-size:14px;line-height:1.5;}}
.code-btn:hover{{background:#e9ecef;border-color:#007bff;transform:translateY(-2px);box-shadow:0 4px 8px rgba(0,0,0,0.1);}}
.smc h3{{margin:0 0 10px 0;color:#333;font-size:20px;}}
.smc h4{{margin:10px 0;color:#666;font-size:16px;font-weight:normal;}}
.smc p{{line-height:1.8;color:#444;margin:15px 0;}}
</style>
<div class="rc"><img src="data:image/png;base64,{image_b64}" class="ri">{highlights}</div>{modals}{detail_modals}
<script>
function show(id){{document.getElementById(id).style.display="block";}}
function hide(id){{document.getElementById(id).style.display="none";}}
function switchTab(evt, tabId){{
    var mc = evt.target.closest('.mc');
    mc.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    mc.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    evt.target.classList.add('active');
}}
</script>'''

st.title("🏥 Medical Report Analyzer")
st.markdown("Upload a medical report (PDF or image) to extract medical entities and retrieve relevant coding information.")

uploaded_file = st.file_uploader("Upload Report", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        with st.spinner("Uploading file to MCP server..."):
            # Upload file to server
            mcp_url = os.getenv("MCP_SERVER_URL", "http://server:8000")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{mcp_url}/upload", files=files, timeout=30)
            response.raise_for_status()
            server_file_path = response.json()["file_path"]
            st.success(f"File uploaded: {server_file_path}")
        
        with st.spinner("Extracting text..."):
            full_text = extract_report_text(server_file_path)
            
            if not full_text or full_text.startswith("Error:"):
                st.error(f"Text extraction failed: {full_text}")
                st.stop()
        
        
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == '.pdf':
            report_image = convert_pdf_to_image(server_file_path)
        else:
            report_image = Image.open(server_file_path)
        
        with st.spinner("Running OCR..."):
            buffered = io.BytesIO()
            report_image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            ocr_data = run_ocr_on_image(image_bytes)
        
        with st.spinner("Extracting entities via MCP..."):
            entities = extract_entities_mcp(full_text)
        
        conditions = entities.get("conditions", [])
        procedures = entities.get("procedures", [])
        health_factors = entities.get("health_factors", [])
        
        total_entities = len(conditions) + len(procedures) + len(health_factors)
        
        if total_entities == 0:
            st.warning("No medical entities detected.")
        else:
            st.success(f"Detected {total_entities} entities: {len(conditions)} condition(s), {len(procedures)} procedure(s), {len(health_factors)} health factor(s)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if conditions:
                    st.markdown("**🔴 Conditions:**")
                    for c in conditions:
                        st.markdown(f"- {c}")
            with col2:
                if procedures:
                    st.markdown("**🔵 Procedures:**")
                    for p in procedures:
                        st.markdown(f"- {p}")
            with col3:
                if health_factors:
                    st.markdown("**🟢 Health Factors:**")
                    for h in health_factors:
                        st.markdown(f"- {h}")
            
            with st.spinner("Retrieving medical codes via MCP..."):
                entity_data, codes_for_explanation = fetch_all_codes_parallel(conditions, procedures, health_factors)
            
            with st.spinner("Generating explanations..."):
                explanations = get_batch_explanations(codes_for_explanation)
            
            with st.spinner("Highlighting entities..."):
                highlighted_img, positions = find_and_highlight(
                    report_image, ocr_data, conditions, procedures, health_factors
                )
            
            buffered = io.BytesIO()
            highlighted_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            html = create_interactive_html(img_b64, positions, entity_data, explanations)
            
            st.markdown("### Interactive Medical Report")
            st.markdown("**Click highlighted entities** to view medical codes. Click code buttons for details.")
            st.components.v1.html(html, height=1200, scrolling=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("📄 Upload a medical report to get started")
