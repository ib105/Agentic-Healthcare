import streamlit as st
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
import fitz
import io
import base64
from typing import List, Dict, Tuple
import pytesseract
from pytesseract import Output

from main import comorbidity_pipeline
from report_loading import extract_text_from_pdf, extract_text_from_image
from validation import search_comorbidities

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

def create_interactive_html(image_b64: str, positions: Dict, comorbidities: Dict) -> str:
    highlights = ""
    modals = ""
    
    for idx, (condition, boxes) in enumerate(positions.items()):
        for box_idx, (x, y, w, h) in enumerate(boxes):
            hid = f"h{idx}{box_idx}"
            mid = f"m{idx}{box_idx}"
            
            highlights += f'<div class="hl" onclick="show(\'{mid}\')" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;"></div>'
            
            comorbid_html = ""
            for c_idx, (doc, dist) in enumerate(comorbidities.get(condition, [])):
                cid = f"c{idx}{box_idx}{c_idx}"
                comorbid_name = doc.metadata.get("condition", "Unknown")
                icd10 = doc.metadata.get("icd10", "N/A")
                
                # Extract only Problem and Symptoms
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
            
            modals += f'<div id="{mid}" class="md" onclick="hide(\'{mid}\')"><div class="mc" onclick="event.stopPropagation()"><span class="x" onclick="hide(\'{mid}\')">&times;</span><h2>{condition}</h2><h3>Comorbidities</h3>{comorbid_html or "<p>None found</p>"}</div></div>'
    
    return f'''<style>
.rc{{position:relative;display:inline-block;}}
.ri{{max-width:100%;height:auto;display:block;}}
.hl{{position:absolute;border:2px solid #FFD700;cursor:pointer;}}
.hl:hover{{background:rgba(255,215,0,0.3);}}
.md{{display:none;position:fixed;z-index:999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.5);}}
.mc{{background:#fff;margin:10% auto;padding:30px;border-radius:12px;width:80%;max-width:600px;box-shadow:0 4px 20px rgba(0,0,0,0.3);max-height:80vh;overflow-y:auto;}}
.x{{float:right;font-size:28px;font-weight:bold;cursor:pointer;color:#aaa;}}
.x:hover{{color:#000;}}
.ci{{padding:12px;margin:8px 0;background:#f8f9fa;border-left:4px solid #007bff;border-radius:4px;cursor:pointer;transition:all 0.2s;}}
.ci:hover{{background:#e9ecef;transform:translateX(5px);}}
</style>
<div class="rc"><img src="data:image/png;base64,{image_b64}" class="ri">{highlights}</div>{modals}
<script>
function show(id){{document.getElementById(id).style.display="block";}}
function hide(id){{document.getElementById(id).style.display="none";}}
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
            for condition in conditions:
                comorbidities_map[condition] = search_comorbidities(condition, k=3)
            
            highlighted_img, positions = find_and_highlight(report_image, conditions)
            
            buffered = io.BytesIO()
            highlighted_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            html = create_interactive_html(img_b64, positions, comorbidities_map)
            st.components.v1.html(html, height=1000, scrolling=True)
        
        finally:
            os.unlink(tmp_path)
