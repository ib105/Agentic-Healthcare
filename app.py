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

load_dotenv()

from main import medical_coding_pipeline, llm
from report_loading import extract_text_from_pdf, extract_text_from_image
from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes
from icd11_retriever import search_icd11, fetch_icd11_details, get_z_codes_for_condition
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# ===== Cache OCR results =====
@st.cache_data
def run_ocr_on_image(image_bytes: bytes) -> Dict:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')

# ===== Cache ICD-11 details =====
@lru_cache(maxsize=100)
def cached_fetch_icd11_details(uri: str) -> Dict:
    """Cached version of fetch_icd11_details to avoid redundant API calls"""
    return fetch_icd11_details(uri)

# ===== Batch LLM explanations instead of one-by-one =====
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_batch_explanations(entities_with_codes: List[Tuple[str, str, str, str]]) -> Dict[str, str]:
    """
    Get explanations for multiple codes in a single LLM call
    entities_with_codes: [(entity, code, description, code_type), ...]
    Returns: {f"{entity}|{code}": explanation}
    """
    if not entities_with_codes:
        return {}
    
    # Build batch prompt
    prompts = []
    for i, (entity, code, description, code_type) in enumerate(entities_with_codes, 1):
        if code_type == 'comorbidity':
            prompts.append(f"{i}. Why do {entity} and {description} (ICD-10: {code}) commonly co-occur?")
        elif code_type == 'icd11':
            prompts.append(f"{i}. Why is ICD-11 code {code} ({description}) used for {entity}?")
        elif code_type == 'cpt':
            prompts.append(f"{i}. What does CPT code {code} ({description}) involve for {entity}?")
        elif code_type == 'z_code':
            prompts.append(f"{i}. What does Z-code {code} ({description}) represent for {entity}?")
    
    batch_prompt = f"""Answer each question in EXACTLY 2-3 concise lines. Number your responses (1., 2., etc.):

{chr(10).join(prompts)}"""
    
    try:
        response = llm.invoke([HumanMessage(content=batch_prompt)])
        content = response.content.strip()
        
        # Parse numbered responses
        explanations = {}
        lines = content.split('\n')
        current_num = 0
        current_text = []
        
        for line in lines:
            match = re.match(r'^(\d+)\.\s*(.+)', line)
            if match:
                # Save previous explanation
                if current_num > 0 and current_text:
                    entity, code, desc, _ = entities_with_codes[current_num - 1]
                    key = f"{entity}|{code}"
                    explanations[key] = ' '.join(current_text).strip()
                
                # Start new explanation
                current_num = int(match.group(1))
                current_text = [match.group(2)]
            elif current_num > 0:
                current_text.append(line.strip())
        
        # Save last explanation
        if current_num > 0 and current_text:
            entity, code, desc, _ = entities_with_codes[current_num - 1]
            key = f"{entity}|{code}"
            explanations[key] = ' '.join(current_text).strip()
        
        return explanations
    except Exception as e:
        print(f"Batch explanation error: {e}")
        return {}

# ===== Normalize entities for better matching =====
def normalize_entity(entity: str) -> List[str]:
    """
    Split complex entities into simpler searchable terms
    Returns list of variants to try matching
    """
    variants = [entity]
    
    # Remove parentheses content but keep both versions
    if '(' in entity:
        base = re.sub(r'\([^)]*\)', '', entity).strip()
        inside = re.findall(r'\(([^)]*)\)', entity)
        variants.append(base)
        variants.extend(inside)
    
    # Split on "and" for compound procedures
    if ' and ' in entity.lower():
        parts = re.split(r'\s+and\s+', entity, flags=re.IGNORECASE)
        variants.extend(parts)
    
    # Handle "history of X" patterns
    if 'history of' in entity.lower():
        # Try without "history of"
        without_history = re.sub(r'history of\s+', '', entity, flags=re.IGNORECASE).strip()
        variants.append(without_history)
        # Try just the last significant words
        words = entity.split()
        if len(words) > 3:
            variants.append(' '.join(words[-2:]))  # last 2 words
    
    # For single-word medical terms, add lowercase variant
    if len(entity.split()) == 1:
        variants.append(entity.lower())
        variants.append(entity.capitalize())
    
    # Remove extra whitespace and duplicates
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
        
        pattern_words = [re.escape(w) for w in words]
        
        # Try exact match first
        first_word = words[0].strip('.,;:!?')
        first_word_positions = text_index.get(first_word, [])
        
        for pos in first_word_positions:
            match = True
            matched_length = 0
            
            # Check if subsequent words match (allowing punctuation)
            for offset, word in enumerate(words):
                if pos + offset >= len(ocr_data['text']):
                    match = False
                    break
                
                ocr_word = ocr_data['text'][pos + offset].lower().strip('.,;:!?')
                search_word = word.strip('.,;:!?')
                
                if ocr_word == search_word:
                    matched_length += 1
                elif offset == 0:  # First word must match
                    match = False
                    break
            
            # Accept if we matched at least the core phrase
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
            
            # Try matching original entity first
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

# ===== Collect all codes first, then batch explain =====
def fetch_all_codes_parallel(conditions: List[str], procedures: List[str], health_factors: List[str]) -> Tuple[Dict, List]:
    entity_data = {'conditions': {}, 'procedures': {}, 'health_factors': {}}
    codes_for_explanation = []  # Collect all codes needing explanation
    
    def fetch_condition_data(condition):
        comorbidities = search_comorbidities(condition, k=3)
        icd11 = search_icd11(condition, limit=3, filter_blocks=True)
        
        # Collect codes for batch explanation
        for doc, dist in comorbidities:
            name = doc.metadata.get("condition", "Unknown")
            icd10 = doc.metadata.get("icd10", "N/A")
            codes_for_explanation.append((condition, icd10, name, 'comorbidity'))
        
        for item in icd11:
            codes_for_explanation.append((condition, item['code'], item['title'], 'icd11'))
        
        return condition, {'comorbidities': comorbidities, 'icd11': icd11}
    
    def fetch_procedure_data(procedure):
        cpt = search_cpt_codes(procedure, k=3)
        
        for item in cpt:
            codes_for_explanation.append((procedure, item['code'], item['description'], 'cpt'))
        
        return procedure, {'cpt': cpt}
    
    def fetch_health_factor_data(factor):
        z_codes = get_z_codes_for_condition(factor, limit=3)
        
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
                        for c_idx, (doc, dist) in enumerate(comorbidities):
                            cid = f"d{entity_type}{idx}{box_idx}{c_idx}"
                            name = doc.metadata.get("condition", "Unknown")
                            icd10 = doc.metadata.get("icd10", "N/A")
                            
                            comorbid_buttons += f'<button class="code-btn" onclick="show(\'{cid}\')">{name}<br><small>Distance: {dist:.3f}</small></button>'
                            
                            # Get explanation from batch
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
            
            with st.spinner("Running OCR..."):
                buffered = io.BytesIO()
                report_image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                ocr_data = run_ocr_on_image(image_bytes)
            
            with st.spinner("Extracting entities..."):
                state = medical_coding_pipeline.invoke({"patient_report": full_text})
            
            conditions = state.get("extracted_conditions", [])
            procedures = state.get("extracted_procedures", [])
            health_factors = state.get("extracted_health_factors", [])
            
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
                
                with st.spinner("Retrieving medical codes..."):
                    # Fetch all codes AND collect what needs explanation
                    entity_data, codes_for_explanation = fetch_all_codes_parallel(conditions, procedures, health_factors)
                
                # ===== Single batch LLM call for ALL explanations =====
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
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
else:
    st.info("Upload a medical report to get started")
