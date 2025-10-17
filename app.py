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

from main import comorbidity_pipeline
from report_loading import extract_text_from_pdf, extract_text_from_image
from comorbidity_retriever import search_comorbidities
from cpt_retriever import search_cpt_codes

st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

def convert_pdf_to_image(pdf_path: str) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(img_data))

def find_and_highlight(image: Image.Image, conditions: List[str]) -> Tuple[Image.Image, Dict]:
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')
    img_with_highlights = image.copy()
    draw = ImageDraw.Draw(img_with_highlights, 'RGBA')
    
    img_width, img_height = image.size
    positions = {}
    
    for condition in conditions:
        words = condition.lower().split()
        positions[condition] = []
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            match_found = True
            boxes = []
            
            for j, word in enumerate(words):
                if i + j >= n_boxes:
                    match_found = False
                    break
                if word not in ocr_data['text'][i + j].lower():
                    match_found = False
                    break
                boxes.append(i + j)
            
            if match_found and boxes:
                xs = [ocr_data['left'][idx] for idx in boxes]
                ys = [ocr_data['top'][idx] for idx in boxes]
                ws = [ocr_data['width'][idx] for idx in boxes]
                hs = [ocr_data['height'][idx] for idx in boxes]
                
                x = min(xs)
                y = min(ys)
                w = max(xs[j] + ws[j] for j in range(len(xs))) - x
                h = max(ys[j] + hs[j] for j in range(len(ys))) - y
                
                draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 0, 80), outline=(255, 200, 0, 255), width=2)
                
                x_pct = (x / img_width) * 100
                y_pct = (y / img_height) * 100
                w_pct = (w / img_width) * 100
                h_pct = (h / img_height) * 100
                
                positions[condition].append((x_pct, y_pct, w_pct, h_pct))
    
    return img_with_highlights, positions

def create_interactive_html(image_b64: str, positions: Dict, comorbidities: Dict, cpt_codes: Dict) -> str:
    from langchain_core.messages import HumanMessage
    from main import llm
    
    highlights = ""
    modals = ""
    
    for idx, (condition, boxes) in enumerate(positions.items()):
        for box_idx, (x, y, w, h) in enumerate(boxes):
            hid = f"h{idx}{box_idx}"
            mid = f"m{idx}{box_idx}"
            
            highlights += f'<div class="hl" onclick="show(\'{mid}\')" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;"></div>'
            
            # Comorbidities content
            comorbid_html = ""
            for c_idx, (doc, dist) in enumerate(comorbidities.get(condition, [])):
                cid = f"c{idx}{box_idx}{c_idx}"
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
                    <b>{comorbid_name}</b> <span style="color:#007bff;cursor:pointer;">ⓘ</span><br>
                    <small>ICD-10: {icd10}</small>
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
            
            # CPT codes content with LLM analysis
            cpt_results = cpt_codes.get(condition, [])
            
            if cpt_results:
                top_codes = [
                    f"{item['code']} — {item['description']} (similarity: {item['similarity']:.2f})"
                    for item in cpt_results
                ]
                formatted_codes = "\n".join([f"- {c}" for c in top_codes])
                
                # Get LLM analysis
                prompt = f"""
                You are a clinical coding assistant. Based on the following retrieved CPT/HCPCS codes and their descriptions, explain what procedures might relate to the condition: "{condition}".
                If there is no single direct CPT code, say that explicitly and summarize the likely related categories.
                
                Retrieved codes:
                {formatted_codes}

                Respond in a concise section with:
                - A short overview paragraph.
                - A bulleted list of relevant codes and their use (use * for bullets).
                Keep it brief and clinical.
                """
                
                llm_response = llm.invoke([HumanMessage(content=prompt)])
                analysis = llm_response.content.strip().replace('\n', '<br>')
                
                cpt_html = f'<div style="line-height:1.8;">{analysis}</div>'
            else:
                cpt_html = "<p>None found</p>"
            
            # Modal with tabs
            modals += f'''<div id="{mid}" class="md" onclick="hide('{mid}')">
                <div class="mc" onclick="event.stopPropagation()">
                    <span class="x" onclick="hide('{mid}')">&times;</span>
                    <h2>{condition}</h2>
                    <div class="tabs">
                        <button class="tab active" onclick="switchTab(event, 'comorbid{idx}{box_idx}')">Comorbidities</button>
                        <button class="tab" onclick="switchTab(event, 'cpt{idx}{box_idx}')">CPT Codes</button>
                    </div>
                    <div id="comorbid{idx}{box_idx}" class="tab-content active">
                        <h3>Comorbidities</h3>
                        {comorbid_html or "<p>None found</p>"}
                    </div>
                    <div id="cpt{idx}{box_idx}" class="tab-content">
                        <h3>CPT/HCPCS Codes</h3>
                        {cpt_html}
                    </div>
                </div>
            </div>'''
    
    return f'''<style>
.rc{{position:relative;display:inline-block;}}
.ri{{max-width:100%;height:auto;display:block;}}
.hl{{position:absolute;border:2px solid #FFD700;cursor:pointer;}}
.hl:hover{{background:rgba(255,215,0,0.3);}}
.md{{display:none;position:fixed;z-index:999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.5);}}
.mc{{background:#fff;margin:10% auto;padding:30px;border-radius:12px;width:80%;max-width:700px;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-height:80vh;overflow-y:auto;}}
.x{{float:right;font-size:28px;font-weight:bold;cursor:pointer;color:#aaa;}}
.x:hover{{color:#000;}}
.ci{{padding:12px;margin:8px 0;background:#f8f9fa;border-left:4px solid #007bff;border-radius:4px;transition:all 0.2s;}}
.ci:hover{{background:#e9ecef;}}
.tabs{{display:flex;gap:10px;margin:20px 0;border-bottom:2px solid #e9ecef;}}
.tab{{padding:10px 20px;background:none;border:none;cursor:pointer;font-size:16px;color:#666;border-bottom:3px solid transparent;transition:all 0.3s;}}
.tab:hover{{color:#007bff;}}
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

uploaded_file = st.file_uploader("Upload Report", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    with st.spinner("Processing..."):
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
            
            state = comorbidity_pipeline.invoke({"patient_report": full_text})
            conditions = state.get("extracted_conditions", [])
            
            comorbidities_map = {}
            cpt_codes_map = {}
            for condition in conditions:
                comorbidities_map[condition] = search_comorbidities(condition, k=3)
                cpt_codes_map[condition] = search_cpt_codes(condition, k=5)
            
            highlighted_img, positions = find_and_highlight(report_image, conditions)
            
            buffered = io.BytesIO()
            highlighted_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            html = create_interactive_html(img_b64, positions, comorbidities_map, cpt_codes_map)
            st.components.v1.html(html, height=1000, scrolling=True)
        
        finally:
            os.unlink(tmp_path)
