# app.py

import sys, types, importlib
import base64
import json
import os
import re
import time
import uuid
import subprocess
from typing import List, Dict

import fitz
import numpy as np
from PIL import Image

import openai
import ragflow
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_sock import Sock
from ragflow import deepdoc
from pathlib import Path

# ------------------- CONFIG ---------------------------------------------------

openai.api_key = "your_api_key"
UPLOAD_ROOT = Path(__file__).parent / "uploads"
UPLOAD_ROOT.mkdir(exist_ok=True, parents=True)

app = Flask(__name__)
sock = Sock(app)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_ROOT)

# ===================== Reference text filter (for claims) =======================

def is_reference_text(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    if re.match(r'^\d+\.', text.strip()) and ',' in text and '.' in text:
        return True
    if re.search(r'et al\.|CMAJ|N Engl J Med|doi:|PMID|\d{4};', text):
        return True
    if " " not in text and sum(1 for c in text if c.isupper()) > 3:
        return True
    if text.count(',') > 4 and text.count('.') < 2:
        return True
    return False

def detokenize_claim(text: str) -> str:
    if not text or " " in text or len(text) < 12:
        return text
    fixed = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return fixed[0].upper() + fixed[1:].lower()

def filter_and_fix_claims(claims: List[Dict]) -> List[Dict]:
    out = []
    for c in claims:
        t = c.get("claim_sentence", "")
        if is_reference_text(t):
            continue
        c["claim_sentence"] = detokenize_claim(t)
        out.append(c)
    return out

def ensure_unique_columns(df):
    new_cols, seen = [], {}
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

# ===================== File & JSON helpers =======================

def get_paper_folder(name: str) -> str:
    return str(UPLOAD_ROOT / name)

def create_paper_folder(filename: str) -> (str, str):
    base = Path(filename).stem
    folder_name = f"{base}_{int(time.time())}"
    folder = UPLOAD_ROOT / folder_name
    folder.mkdir(exist_ok=True)
    return folder_name, str(folder)

def extract_json_from_response(txt: str) -> str:
    m = re.search(r'```json(.*?)```', txt, re.DOTALL)
    if m:
        return m.group(1).strip()
    s, e = txt.find('['), txt.rfind(']')
    return txt[s:e+1] if s != -1 and e != -1 else txt

def clean_html_table(html_table: str) -> str:
    if not html_table:
        return ""
    html_table = re.sub(r"^```html?\s*\n", "", html_table.strip(), flags=re.IGNORECASE)
    html_table = re.sub(r"\n?```$", "", html_table.strip())
    table_open_pattern = re.compile(
        r"<table[^>]*\bborder\s*=\s*([\"']?)1\1[^>]*>",
        flags=re.IGNORECASE,
    )
    return table_open_pattern.sub(
        '<table class="table table-bordered table-striped">', html_table
    )

def json_to_html_table(table_json):
    if isinstance(table_json, str) and table_json.lstrip().startswith("<table"):
        return table_json
    try:
        data = json.loads(table_json)
        if not data:
            return "<p>No data available</p>"
        headers = data[0].keys()
        html = ['<table class="table table-bordered table-striped"><thead><tr>']
        html += [f"<th>{h}</th>" for h in headers]
        html.append("</tr></thead><tbody>")
        for row in data:
            html.append("<tr>" + "".join(f"<td>{row[h]}</td>" for h in headers) + "</tr>")
        html.append("</tbody></table>")
        return "".join(html)
    except Exception as e:
        return f"<p>Error converting table: {e}</p>"

# ==================== DeepDoc integration ============================

def _make_ocr():
    try:
        from ragflow.deepdoc.vision.ocr import PaddleOCRRunner
        return PaddleOCRRunner()
    except ImportError:
        from ragflow.deepdoc.vision.ocr import OCR
        return OCR()

def run_deepdoc(pdf_path: Path, out_dir: Path):
    ocr = _make_ocr()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        print(f"[DeepDoc] OCR page {i}/{doc.page_count}")
        pix = page.get_pixmap(dpi=200)
        img_path = out_dir / f"{pdf_path.stem}_page-{i:03d}.png"
        img_path.write_bytes(pix.tobytes("png"))
        pil_img = Image.open(img_path)
        img_arr = np.array(pil_img)
        res = ocr(img_arr)
        txt_path = img_path.with_suffix(".png.txt")
        text_lines = []
        for r in res:
            if isinstance(r, dict) and "text" in r:
                text_lines.append(r["text"])
            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                if isinstance(r[1], str):
                    text_lines.append(r[1])
                elif isinstance(r[1], (list, tuple)) and isinstance(r[1][0], str):
                    text_lines.append(r[1][0])
            elif isinstance(r, str):
                text_lines.append(r)
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")

# ==================== GPT-based metadata & tables =========================

def gpt_extract_metadata(img_path: Path, ocr_text: str) -> Dict[str, str]:
    """
    Extracts the paper title & abstract via GPT, with JSON code-block wrapper.
    Falls back to regex if parsing fails.
    """
    with open(img_path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()

    prompt = (
        "I have scanned the first page of a scientific paper.  Extract the **title** and **abstract**.  "
        "Return *only* a JSON object, wrapped in a ```json …``` code block, with exactly these keys:\n\n"
        "```json\n"
        "{\n"
        "  \"title\": \"…\",\n"
        "  \"abstract\": \"…\"\n"
        "}\n"
        "```\n\n"
        "OCR TEXT (first 2000 chars):\n" + ocr_text[:2000]
    )

    msg = [
        {"role": "system", "content": "You are a paper metadata extractor."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ],
        }
    ]

    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=msg,
        temperature=0.0,
        max_tokens=1024,
    )

    raw = resp.choices[0].message.content
    print("[gpt_extract_metadata] raw:", raw)
    snippet = extract_json_from_response(raw)
    try:
        doc = json.loads(snippet)
        return {
            "title": doc.get("title", "").strip(),
            "abstract": doc.get("abstract", "").strip()
        }
    except Exception as e:
        print("[gpt_extract_metadata] parse failed:", e)
    # fallback
    lines = ocr_text.splitlines()
    title = next((l.strip() for l in lines if l.strip()), "")
    m = re.search(r'Abstract[:\s]*(.+?)(?:\n\s*\d|\n\n|\Z)', ocr_text, re.S)
    abstract = m.group(1).strip() if m else ""
    return {"title": title, "abstract": abstract}

def gpt_extract_tables(img_path: Path, ocr_text: str) -> List[Dict]:
    """
    Extract *all* tables from a page via GPT, asking for a JSON array of
    { "caption": "...", "html": "<table>…</table>" } objects.
    """
    with open(img_path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode()

    prompt = (
        "You are given a scanned page image + its OCR text.  Identify *all* tables on the page, "
        "including their full caption text and the raw HTML for each table.  "
        "Return *only* a JSON array of objects with exactly these keys:\n\n"
        "```json\n"
        "[\n"
        "  { \"caption\": \"…\", \"html\": \"<table>…</table>\" },\n"
        "  …\n"
        "]\n"
        "```\n\n"
        "OCR TEXT (first 2000 chars):\n" + ocr_text[:2000]
    )

    msg = [
        {"role": "system", "content": "You are a table extractor."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ],
        }
    ]

    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=msg,
        temperature=0.0,
        max_tokens=2048,
    )

    raw = resp.choices[0].message.content
    json_str = extract_json_from_response(raw)
    try:
        return json.loads(json_str)
    except Exception as e:
        print("[gpt_extract_tables] parse failed:", e)
        return []

# ==================== Parsing routines ============================

def parse_tables_deepdoc(deepdoc_dir: Path) -> List[Dict]:
    out = []
    for img in sorted(deepdoc_dir.glob("*.png")):
        page_no = int(re.search(r'page-(\d+)', img.stem).group(1))
        print(f"Parsing all tables in Page {page_no}")
        txt_path = img.with_suffix(img.suffix + ".txt")
        if not txt_path.exists():
            continue
        ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        tables = gpt_extract_tables(img, ocr_text)
        for tbl in tables:
            out.append({
                "page": page_no,
                "caption": tbl.get("caption", "").strip(),
                "table_json": "",
                "table_html": tbl.get("html", "").strip(),
            })
    return out

def preprocess_pdf_deepdoc(deepdoc_dir: Path) -> List[Dict]:
    pages = []
    for txt in sorted(deepdoc_dir.glob("*.png.txt")):
        page_no = int(re.search(r'page-(\d+)', txt.stem).group(1))
        pages.append({
            "page": page_no,
            "text": txt.read_text(encoding="utf-8", errors="ignore")
        })
    return pages

# ===================== Claim extraction & helpers =========================

def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    chunk_size = max_tokens * 4
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def extract_claims(text_data: List[Dict]) -> List[Dict]:
    combined_text = "\n".join(f"(Page {t['page']}): {t['text']}" for t in text_data)
    text_chunks = chunk_text(combined_text, max_tokens=4000)
    all_claims = []
    for idx, chunk in enumerate(text_chunks, start=1):
        prompt = f"""
You are an AI that extracts scientific or clinically relevant claims related to Tables from the text below.
Return a JSON list of objects with:
  - "claim_sentence": the relevant snippet
  - "page": page number if possible
  - "table_reference": e.g. "Table 1"
This is chunk {idx} of {len(text_chunks)}:
{chunk}
"""
        try:
            resp = openai.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a scientific claim extractor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = resp.choices[0].message.content
            try:
                claims = json.loads(content)
            except json.JSONDecodeError:
                claims = []
            if isinstance(claims, list):
                all_claims.extend(claims)
        except Exception as e:
            print(f"Error extracting claims for chunk {idx}: {e}")
    return all_claims

def compute_relevance(input_claim: str, evidence_list: List[Dict], evidence_type: str = "claim") -> List[Dict]:
    prompt = f"""
You are an expert at determining the relevance between a scientific query and a list of {evidence_type}s.
Given an input query and a list of {evidence_type}s, calculate a relevance score (0–100) for each item.
Return a JSON list of objects with keys:
- "evidence_index": index in the provided list
- "relevance_score": number between 0–100
- "explanation": brief explanation

Input Query: {input_claim}

{evidence_type.capitalize()}s:
{json.dumps(
    [e['claim_sentence'] if evidence_type=="claim" else (e['table_json'] or e.get('table_html', '')) 
     for e in evidence_list], indent=2
)}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"You are a scientific {evidence_type} relevance analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        match = re.search(r"\[.*\]", resp.choices[0].message.content, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except Exception as e:
        print(f"Relevance scoring error: {e}")
        return []

def fact_check_against_evidence(input_claim: str, evidence: Dict, evidence_type: str) -> Dict:
    if evidence_type == "claim":
        prompt = f"""
You are a scientific fact-checker. Verify the following query claim using the provided evidence claim.
Return valid JSON with keys:
- "input_claim"
- "evidence_claim"
- "status" (Supported, Refuted, Partially Supported, Unverified, Not Relevant)
- "details" (rationale)

Input claim: {input_claim}
Evidence claim: {evidence.get('claim_sentence')}
"""
    else:
        prompt = f"""
You are a scientific fact-checker. Verify the following query claim using the provided table.
Return valid JSON with keys:
- "input_claim"
- "table_reference"
- "status" (Supported, Refuted, Partially Supported, Unverified, Not Relevant)
- "details" (rationale referring to table rows/columns)

Input claim: {input_claim}
Table reference: {evidence.get('caption')} (page {evidence.get('page')})
TABLE HTML: {evidence.get('table_html')}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a scientific fact-checker."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        match = re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL)
        return json.loads(match.group(0)) if match else {"status": "Unknown", "details": "Could not parse GPT output."}
    except Exception as e:
        return {"status": "Unknown", "details": f"Error: {e}"}

# ---------------------------- ROUTES ----------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file selected"}), 400

    folder_name, folder_path = create_paper_folder(file.filename)
    pdf_path = Path(folder_path) / file.filename
    file.save(pdf_path)

    try:
        # 1) OCR via DeepDoc
        print("[1/4] Running DeepDoc …")
        deepdoc_dir = Path(folder_path) / "deepdoc"
        run_deepdoc(pdf_path, deepdoc_dir)

        # 2) parse all tables
        print("[2/4] Parsing tables with GPT …")
        table_data = parse_tables_deepdoc(deepdoc_dir)

        # 3) extract full page text
        print("[3/4] Extracting full page text …")
        text_data = preprocess_pdf_deepdoc(deepdoc_dir)

        # 4) extract title & abstract
        print("[4/4] GPT-extracting title & abstract …")
        first_img = sorted(deepdoc_dir.glob("*.png"))[0]
        first_txt = first_img.with_suffix(first_img.suffix + ".txt")
        meta = {"title": "", "abstract": ""}
        if first_txt.exists():
            meta = gpt_extract_metadata(
                first_img,
                first_txt.read_text(encoding="utf-8", errors="ignore")
            )

        # build result.json
        combined = []
        for tbl in table_data:
             page_text = next((p['text'] for p in text_data if p['page'] == tbl['page']), "")
             cap = tbl["caption"]
             parts = page_text.split(cap, 1)
             if len(parts) == 2:
                 before, after = parts[0].strip(), parts[1].strip()
                 context = f"{before}\n\n{after}"
             else:
                 # fallback to full page if caption not found
                 context = page_text
 
             combined.append({
                 "title":     meta.get("title", ""),
                 "abstract":  meta.get("abstract", ""),
                 "context":   context,
                 "caption":   cap,
                 "table_html": tbl["table_html"]
            })

        with open(Path(folder_path) / "result.json", "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        return jsonify({
            "message": "PDF parsed",
            "paper_folder": folder_name,
            "result": combined
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/extract_claims', methods=['POST'])
def extract_claims_route():
    data = request.get_json() or {}
    pf = data.get('paper_folder')
    if not pf:
        return jsonify({'error': 'paper_folder not provided'}), 400
    folder = get_paper_folder(pf)
    text_path = os.path.join(folder, "text.json")
    try:
        text_data = json.load(open(text_path))
        claims = extract_claims(text_data)
        with open(os.path.join(folder, "claims.json"), "w") as f:
            json.dump(claims, f)
        return jsonify({'message': 'Claims extracted', 'claims': claims})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_manual_claims', methods=['POST'])
def save_manual_claims():
    data = request.get_json() or {}
    pf = data.get('paper_folder')
    rev = data.get('revised_claims')
    if not pf or rev is None:
        return jsonify({'error': 'Missing required data'}), 400
    folder = get_paper_folder(pf)
    try:
        with open(os.path.join(folder, "claims.json"), "w") as f:
            json.dump(rev, f)
        return jsonify({'message': 'Claims updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_claim', methods=['POST'])
def check_claim_route():
    data = request.get_json() or {}
    pf = data.get('paper_folder')
    ic = data.get('input_claim')
    thr = data.get('relevance_threshold', 70)
    if not pf or not ic:
        return jsonify({'error': 'Missing required data'}), 400

    folder = get_paper_folder(pf)
    try:
        claims = json.load(open(os.path.join(folder, "claims.json")))
        tables = json.load(open(os.path.join(folder, "result.json")))

        claims = filter_and_fix_claims(claims)
        cr = compute_relevance(ic, claims, "claim")
        relevant_claims = [
            {**claims[i["evidence_index"]],
             "relevance_score": i["relevance_score"],
             "relevance_explanation": i["explanation"]}
            for i in cr if i["relevance_score"] >= thr
        ]

        tr = compute_relevance(ic, tables, "table")
        relevant_tables = [
            {**tables[i["evidence_index"]],
             "relevance_score": i["relevance_score"],
             "relevance_explanation": i["explanation"]}
            for i in tr if i["relevance_score"] >= thr
        ]

        results = []
        for rt in relevant_tables:
            fc = fact_check_against_evidence(ic, rt, "table")
            rt["status"], rt["details"] = fc.get("status", ""), fc.get("details", "")
            results.append({
                "evidence_type": "Table",
                "input_claim": ic,
                "table_reference": rt["caption"],
                "table_html": clean_html_table(rt["table_html"]),
                "relevance_score": rt["relevance_score"],
                "evidence": "",
                "label": rt["status"],
                "rationale": rt["details"]
            })
        for rc in relevant_claims:
            fc = fact_check_against_evidence(ic, rc, "claim")
            rc["status"], rc["details"] = fc.get("status", ""), fc.get("details", "")
            results.append({
                "evidence_type": "Text",
                "input_claim": ic,
                "evidence": rc["claim_sentence"],
                "relevance_score": rc["relevance_score"],
                "label": rc["status"],
                "rationale": rc["details"],
                "table_reference": "",
                "table_html": ""
            })

        results = sorted([r for r in results if r["evidence_type"]=="Table"],
                         key=lambda x: -x["relevance_score"]) + \
                  sorted([r for r in results if r["evidence_type"]=="Text"],
                         key=lambda x: -x["relevance_score"])

        return jsonify({
            'message': 'Claim checking completed',
            'input_claim': ic,
            'relevance_threshold': thr,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/uploads/<path:fn>")
def uploads_static(fn):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fn)

# -------------------------- RUN --------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=5001)
