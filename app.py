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

from main import medical_coding_pipeline, llm
from report_loading import extract_text_from_pdf, extract_text_from_image
from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes
from icd11_retriever import search_icd11, fetch_icd11_details, get_z_codes_for_condition
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Cache OCR results to avoid re-processing
@st.cache_data
def run_ocr_on_image(image_bytes: bytes) -> Dict:
    """Cache OCR results to avoid re-processing the same image"""
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')

def convert_pdf_to_image(pdf_path: str) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_data))

def create_text_index(ocr_data: Dict) -> Dict[str, List[int]]:
    """Create an index of word positions for O(1) lookups"""
    index = {}
    for i, text in enumerate(ocr_data['text']):
        word = text.lower().strip()
        if word:
            if word not in index:
                index[word] = []
            index[word].append(i)
    return index

def find_entity_positions_optimized(entity: str, text_index: Dict, ocr_data: Dict) -> List[int]:
    """Fast entity matching using pre-built index"""
    words = entity.lower().split()
    if not words:
        return []
    
    # Find positions of first word
    first_word_positions = text_index.get(words[0], [])
    
    matched_positions = []
    for pos in first_word_positions:
        # Check if subsequent words follow
        match = True
        for offset, word in enumerate(words[1:], 1):
            if pos + offset >= len(ocr_data['text']):
                match = False
                break
            if ocr_data['text'][pos + offset].lower().strip() != word:
                match = False
                break
        
        if match:
            matched_positions.append(pos)
    
    return matched_positions

def find_and_highlight(image: Image.Image, ocr_data: Dict, conditions: List[str], 
                      procedures: List[str], health_factors: List[str]) -> Tuple[Image.Image, Dict]:
    """Optimized highlighting with pre-computed OCR data and text index"""
    
    # Create text index for fast lookups
    text_index = create_text_index(ocr_data)
    
    img_with_highlights = image.copy()
    draw = ImageDraw.Draw(img_with_highlights, 'RGBA')
    
    img_width, img_height = image.size
    positions = {
        'conditions': {},
        'procedures': {},
        'health_factors': {}
    }
    
    # Color schemes for different entity types
    colors = {
        'conditions': {'fill': (255, 100, 100, 80), 'outline': (220, 20, 60, 255)},  # Red
        'procedures': {'fill': (100, 149, 237, 80), 'outline': (65, 105, 225, 255)},  # Blue
        'health_factors': {'fill': (144, 238, 144, 80), 'outline': (34, 139, 34, 255)}  # Green
    }
    
    def highlight_entities(entities: List[str], entity_type: str):
        for entity in entities:
            positions[entity_type][entity] = []
            
            # Find all matching positions using optimized search
            matched_positions = find_entity_positions_optimized(entity, text_index, ocr_data)
            
            # Get word count for bounding box calculation
            word_count = len(entity.split())
            
            for start_pos in matched_positions:
                # Calculate bounding box for the matched phrase
                boxes_indices = list(range(start_pos, start_pos + word_count))
                
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

def get_llm_explanation(entity: str, entity_type: str, code_type: str, codes_data: List) -> str:
    """Generate meaningful explanation for medical codes using LLM."""
    
    if not codes_data:
        return "<p style='color:#999;'>No codes found for this entity.</p>"
    
    if code_type == "comorbidity":
        codes_text = "\n".join([
            f"- {doc.metadata.get('condition', 'Unknown')} (ICD-10: {doc.metadata.get('icd10', 'N/A')})"
            for doc, _ in codes_data
        ])
        prompt = f"""
You are a medical expert. Explain the relationship between "{entity}" and these comorbidities:

{codes_text}

Provide:
1. A brief overview (2-3 sentences) of why these conditions commonly co-occur
2. Clinical significance of monitoring these comorbidities

Keep it concise and clinical. Use medical terminology but remain accessible.
"""
    
    elif code_type == "cpt":
        codes_text = "\n".join([
            f"- {item['code']}: {item['description']}"
            for item in codes_data
        ])
        
        entity_context = "this procedure" if entity_type == "procedures" else "this condition"
        
        prompt = f"""
You are a medical coding expert. Explain the CPT/HCPCS codes for "{entity}" ({entity_context}):

Retrieved codes:
{codes_text}

Provide:
1. Which code(s) best match {entity_context} and why (2-3 sentences)
2. What these procedures/services involve
3. When they are typically performed

Be specific and clinical. Format with clear paragraphs.
"""
    
    elif code_type == "icd11":
        codes_text = "\n".join([
            f"- {item.get('code', 'N/A')}: {item.get('title', 'Unknown')}"
            for item in codes_data
        ])
        prompt = f"""
You are a medical coding expert. Explain the ICD-11 diagnostic codes for "{entity}":

Retrieved codes:
{codes_text}

Provide:
1. Which code is the primary/best match and why (2-3 sentences)
2. What distinguishes it from alternative codes (if multiple)
3. Clinical context for using this code

Be specific about the diagnostic criteria. Format with clear paragraphs.
"""
    
    elif code_type == "z_code":
        codes_text = "\n".join([
            f"- {item.get('code', 'N/A')}: {item.get('title', 'Unknown')}"
            for item in codes_data
        ])
        
        entity_context = "this health status factor" if entity_type == "health_factors" else "this condition"
        
        prompt = f"""
You are a medical coding expert. Explain the ICD-11 Z-codes for "{entity}" ({entity_context}):

Retrieved codes:
{codes_text}

Provide:
1. What these Z-codes represent in clinical context (2-3 sentences)
2. When these codes are used (screening, history, prevention, etc.)
3. Why they're relevant to {entity_context}

Be clear about the non-disease nature of Z-codes. Format with clear paragraphs.
"""
    
    try:
        llm_response = llm.invoke([HumanMessage(content=prompt)])
        return llm_response.content.strip().replace('\n', '<br>')
    except Exception as e:
        return f"<p style='color:#dc3545;'>Error generating explanation: {str(e)}</p>"

def fetch_all_codes_parallel(conditions: List[str], procedures: List[str], health_factors: List[str]) -> Dict:
    """Fetch all medical codes in parallel for better performance"""
    
    entity_data = {'conditions': {}, 'procedures': {}, 'health_factors': {}}
    
    def fetch_condition_data(condition):
        return condition, {
            'comorbidities': search_comorbidities(condition, k=3),
            'icd11': search_icd11(condition, limit=3, filter_blocks=True),
            'cpt': search_cpt_codes(condition, k=5),
            'z_codes': get_z_codes_for_condition(condition, limit=2)
        }
    
    def fetch_procedure_data(procedure):
        return procedure, {'cpt': search_cpt_codes(procedure, k=5)}
    
    def fetch_health_factor_data(factor):
        return factor, {'z_codes': get_z_codes_for_condition(factor, limit=2)}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        condition_futures = [executor.submit(fetch_condition_data, c) for c in conditions]
        procedure_futures = [executor.submit(fetch_procedure_data, p) for p in procedures]
        factor_futures = [executor.submit(fetch_health_factor_data, f) for f in health_factors]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(condition_futures):
            try:
                name, data = future.result()
                entity_data['conditions'][name] = data
            except Exception as e:
                st.warning(f"Error fetching data for condition: {e}")
        
        for future in concurrent.futures.as_completed(procedure_futures):
            try:
                name, data = future.result()
                entity_data['procedures'][name] = data
            except Exception as e:
                st.warning(f"Error fetching data for procedure: {e}")
        
        for future in concurrent.futures.as_completed(factor_futures):
            try:
                name, data = future.result()
                entity_data['health_factors'][name] = data
            except Exception as e:
                st.warning(f"Error fetching data for health factor: {e}")
    
    return entity_data

def create_interactive_html(image_b64: str, positions: Dict, entity_data: Dict) -> str:
    """
    entity_data structure:
    {
        'conditions': {
            'condition_name': {
                'comorbidities': [...],
                'icd11': [...],
                'cpt': [...],
                'z_codes': [...]
            }
        },
        'procedures': {
            'procedure_name': {
                'cpt': [...]
            }
        },
        'health_factors': {
            'factor_name': {
                'z_codes': [...]
            }
        }
    }
    """
    
    highlights = ""
    modals = ""
    
    # Define colors for each entity type
    border_colors = {
        'conditions': '#DC143C',      # Crimson red
        'procedures': '#4169E1',      # Royal blue
        'health_factors': '#228B22'   # Forest green
    }
    
    for entity_type, entities in positions.items():
        for idx, (entity, boxes) in enumerate(entities.items()):
            for box_idx, (x, y, w, h) in enumerate(boxes):
                mid = f"m{entity_type}{idx}{box_idx}"
                border_color = border_colors[entity_type]
                
                highlights += f'<div class="hl" onclick="show(\'{mid}\')" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;border-color:{border_color};"></div>'
                
                # Get entity-specific data
                entity_info = entity_data.get(entity_type, {}).get(entity, {})
                
                # Build modal content based on entity type
                modal_tabs = ""
                modal_content = ""
                
                if entity_type == "conditions":
                    # Comorbidities tab
                    comorbidities = entity_info.get('comorbidities', [])
                    comorbid_html = ""
                    
                    if comorbidities:
                        # Get LLM explanation
                        explanation = get_llm_explanation(entity, entity_type, "comorbidity", comorbidities)
                        comorbid_html = f'<div class="explanation">{explanation}</div><hr style="margin:20px 0;">'
                        
                        # Individual comorbidity cards with details
                        for c_idx, (doc, dist) in enumerate(comorbidities):
                            cid = f"c{entity_type}{idx}{box_idx}{c_idx}"
                            comorbid_name = doc.metadata.get("condition", "Unknown")
                            icd10 = doc.metadata.get("icd10", "N/A")
                            
                            content = doc.page_content
                            problem = ""
                            symptoms = ""
                            
                            if "Problem:" in content:
                                problem = content.split("Problem:")[1].split("Symptoms:")[0].strip() if "Symptoms:" in content else content.split("Problem:")[1].strip()
                            if "Symptoms:" in content:
                                symptoms = content.split("Symptoms:")[1].strip()
                            
                            comorbid_html += f'''<div class="ci" onclick="show('{cid}')">
                                <b>{comorbid_name}</b> <span style="color:#007bff;cursor:pointer;">ⓘ Details</span><br>
                                <small>ICD-10: {icd10} | Distance: {dist:.4f}</small>
                            </div>
                            <div id="{cid}" class="md" onclick="hide('{cid}')">
                                <div class="mc" onclick="event.stopPropagation()">
                                    <span class="x" onclick="hide('{cid}')">&times;</span>
                                    <h2>{comorbid_name}</h2>
                                    <div style="color:#666;margin-bottom:20px;">ICD-10: {icd10}</div>
                                    {f'<div style="margin-bottom:15px;"><b>Problem:</b><p style="margin:8px 0;line-height:1.6;">{problem}</p></div>' if problem else ''}
                                    {f'<div><b>Symptoms:</b><p style="margin:8px 0;line-height:1.6;">{symptoms}</p></div>' if symptoms else ''}
                                </div>
                            </div>'''
                    else:
                        comorbid_html = "<p style='color:#999;'>No comorbidities found.</p>"
                    
                    # ICD-11 tab
                    icd11_codes = entity_info.get('icd11', [])
                    if icd11_codes:
                        explanation = get_llm_explanation(entity, entity_type, "icd11", icd11_codes)
                        icd11_html = f'<div class="explanation">{explanation}</div><hr style="margin:20px 0;">'
                        
                        for icd_item in icd11_codes[:3]:
                            code = icd_item.get('code', 'N/A')
                            title = icd_item.get('title', 'Unknown')
                            uri = icd_item.get('uri', '')
                            
                            definition = ""
                            if uri:
                                detail = fetch_icd11_details(uri)
                                if detail and detail.get('definition'):
                                    definition = f"<p style='margin:8px 0;color:#666;line-height:1.6;font-style:italic;'>{detail['definition']}</p>"
                            
                            icd11_html += f'''<div class="ci" style="cursor:default;border-left-color:#007bff;">
                                <b style="color:#007bff;">{code}</b> — {title}
                                {definition}
                            </div>'''
                    else:
                        icd11_html = "<p style='color:#999;'>No ICD-11 codes found.</p>"
                    
                    # CPT tab (optional for conditions)
                    cpt_codes = entity_info.get('cpt', [])
                    if cpt_codes:
                        explanation = get_llm_explanation(entity, entity_type, "cpt", cpt_codes)
                        cpt_html = f'<div class="explanation">{explanation}</div>'
                    else:
                        cpt_html = "<p style='color:#999;'>No CPT codes found for this condition.</p>"
                    
                    # Z-codes tab (optional for conditions)
                    z_codes = entity_info.get('z_codes', [])
                    if z_codes:
                        explanation = get_llm_explanation(entity, entity_type, "z_code", z_codes)
                        z_html = f'<div class="explanation">{explanation}</div>'
                    else:
                        z_html = "<p style='color:#999;'>No Z-codes found for this condition.</p>"
                    
                    modal_tabs = f'''
                        <button class="tab active" onclick="switchTab(event, 'comorbid{entity_type}{idx}{box_idx}')">Comorbidities</button>
                        <button class="tab" onclick="switchTab(event, 'icd11{entity_type}{idx}{box_idx}')">ICD-11</button>
                        <button class="tab" onclick="switchTab(event, 'cpt{entity_type}{idx}{box_idx}')">CPT Codes</button>
                        <button class="tab" onclick="switchTab(event, 'zcodes{entity_type}{idx}{box_idx}')">Z-Codes</button>
                    '''
                    
                    modal_content = f'''
                        <div id="comorbid{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>Comorbidities</h3>
                            {comorbid_html}
                        </div>
                        <div id="icd11{entity_type}{idx}{box_idx}" class="tab-content">
                            <h3>ICD-11 Diagnostic Codes</h3>
                            {icd11_html}
                        </div>
                        <div id="cpt{entity_type}{idx}{box_idx}" class="tab-content">
                            <h3>CPT/HCPCS Codes</h3>
                            {cpt_html}
                        </div>
                        <div id="zcodes{entity_type}{idx}{box_idx}" class="tab-content">
                            <h3>Z-Codes (Health Status)</h3>
                            {z_html}
                        </div>
                    '''
                
                elif entity_type == "procedures":
                    # CPT codes for procedures
                    cpt_codes = entity_info.get('cpt', [])
                    if cpt_codes:
                        explanation = get_llm_explanation(entity, entity_type, "cpt", cpt_codes)
                        cpt_html = f'<div class="explanation">{explanation}</div>'
                    else:
                        cpt_html = "<p style='color:#999;'>No CPT codes found.</p>"
                    
                    modal_tabs = f'<button class="tab active" onclick="switchTab(event, \'cpt{entity_type}{idx}{box_idx}\')">CPT/HCPCS Codes</button>'
                    modal_content = f'''
                        <div id="cpt{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>CPT/HCPCS Procedure Codes</h3>
                            {cpt_html}
                        </div>
                    '''
                
                elif entity_type == "health_factors":
                    # Z-codes for health factors
                    z_codes = entity_info.get('z_codes', [])
                    if z_codes:
                        explanation = get_llm_explanation(entity, entity_type, "z_code", z_codes)
                        z_html = f'<div class="explanation">{explanation}</div>'
                    else:
                        z_html = "<p style='color:#999;'>No Z-codes found.</p>"
                    
                    modal_tabs = f'<button class="tab active" onclick="switchTab(event, \'zcodes{entity_type}{idx}{box_idx}\')">Z-Codes</button>'
                    modal_content = f'''
                        <div id="zcodes{entity_type}{idx}{box_idx}" class="tab-content active">
                            <h3>ICD-11 Z-Codes (Health Status)</h3>
                            {z_html}
                        </div>
                    '''
                
                # Create modal
                entity_label = entity_type.replace('_', ' ').title()
                modals += f'''<div id="{mid}" class="md" onclick="hide('{mid}')">
                    <div class="mc" onclick="event.stopPropagation()">
                        <span class="x" onclick="hide('{mid}')">&times;</span>
                        <div class="entity-badge {entity_type}">{entity_label}</div>
                        <h2>{entity}</h2>
                        <div class="tabs">
                            {modal_tabs}
                        </div>
                        {modal_content}
                    </div>
                </div>'''
    
    return f'''<style>
.rc{{position:relative;display:inline-block;}}
.ri{{max-width:100%;height:auto;display:block;}}
.hl{{position:absolute;border:3px solid;cursor:pointer;transition:all 0.2s;}}
.hl:hover{{background:rgba(255,255,255,0.2);transform:scale(1.02);}}
.md{{display:none;position:fixed;z-index:999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.6);}}
.mc{{background:#fff;margin:5% auto;padding:30px;border-radius:12px;width:85%;max-width:800px;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-height:85vh;overflow-y:auto;}}
.x{{float:right;font-size:28px;font-weight:bold;cursor:pointer;color:#aaa;}}
.x:hover{{color:#000;}}
.entity-badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px;}}
.entity-badge.conditions{{background:#ffebee;color:#c62828;}}
.entity-badge.procedures{{background:#e3f2fd;color:#1565c0;}}
.entity-badge.health_factors{{background:#e8f5e9;color:#2e7d32;}}
.ci{{padding:12px;margin:8px 0;background:#f8f9fa;border-left:4px solid #007bff;border-radius:4px;transition:all 0.2s;cursor:pointer;}}
.ci:hover{{background:#e9ecef;transform:translateX(5px);}}
.explanation{{background:#f0f7ff;padding:15px;border-radius:8px;margin-bottom:15px;line-height:1.8;border-left:4px solid #007bff;}}
.tabs{{display:flex;gap:5px;margin:20px 0;border-bottom:2px solid #e9ecef;flex-wrap:wrap;}}
.tab{{padding:10px 16px;background:none;border:none;cursor:pointer;font-size:14px;color:#666;border-bottom:3px solid transparent;transition:all 0.3s;}}
.tab:hover{{color:#007bff;background:#f8f9fa;}}
.tab.active{{color:#007bff;border-bottom-color:#007bff;font-weight:bold;}}
.tab-content{{display:none;margin-top:20px;}}
.tab-content.active{{display:block;}}
</style>
<div class="rc"><img src="data:image/png;base64,{image_b64}" class="ri">{highlights}</div>{modals}
<script>
function show(id){{document.getElementById(id).style.display="block";}}
function hide(id){{document.getElementById(id).style.display="none";}}
function switchTab(evt, tabId){{
    var tabContent = evt.target.closest('.mc').querySelectorAll('.tab-content');
    tabContent.forEach(tc => tc.classList.remove('active'));
    var tabs = evt.target.closest('.mc').querySelectorAll('.tab');
    tabs.forEach(t => t.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    evt.target.classList.add('active');
}}
</script>'''

st.title("Medical Report Analyzer")
st.markdown("Upload a medical report (PDF or image) to extract medical entities and retrieve relevant coding information.")

uploaded_file = st.file_uploader("Upload Report", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing report..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == '.pdf':
                full_text = extract_text_from_pdf(tmp_path)
                report_image = convert_pdf_to_image(tmp_path)
            else:
                full_text = extract_text_from_image(tmp_path)
                report_image = Image.open(tmp_path)
            
            # Run OCR once and cache it
            with st.spinner("Running OCR on document..."):
                buffered = io.BytesIO()
                report_image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                ocr_data = run_ocr_on_image(image_bytes)
            
            # Run the enhanced pipeline
            with st.spinner("Extracting medical entities..."):
                state = medical_coding_pipeline.invoke({"patient_report": full_text})
            
            conditions = state.get("extracted_conditions", [])
            procedures = state.get("extracted_procedures", [])
            health_factors = state.get("extracted_health_factors", [])
            
            total_entities = len(conditions) + len(procedures) + len(health_factors)
            
            if total_entities == 0:
                st.warning("No medical entities detected in the report.")
            else:
                st.success(f"Detected {total_entities} entities: {len(conditions)} condition(s), {len(procedures)} procedure(s), {len(health_factors)} health factor(s)")
                
                # Show extracted entities
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
                
                # Retrieve all medical coding information in parallel
                with st.spinner("Retrieving medical codes in parallel..."):
                    entity_data = fetch_all_codes_parallel(conditions, procedures, health_factors)
                
                # Highlight all entities in image using cached OCR
                with st.spinner("Highlighting entities in report..."):
                    highlighted_img, positions = find_and_highlight(
                        report_image, ocr_data, conditions, procedures, health_factors
                    )
                
                # Convert to base64
                buffered = io.BytesIO()
                highlighted_img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Generate interactive HTML with LLM explanations
                with st.spinner("Generating AI explanations..."):
                    html = create_interactive_html(img_b64, positions, entity_data)
                
                st.markdown("### Interactive Medical Report")
                st.markdown("**Click on any highlighted entity** to view detailed medical coding information with AI-generated explanations.")
                st.components.v1.html(html, height=1200, scrolling=True)
        
        except Exception as e:
            st.error(f"Error processing report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
else:
    st.info("Upload a medical report to get started")
