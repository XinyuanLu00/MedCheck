import argparse, os, pickle, textwrap, json
from pathlib import Path
from typing import List

from flask import Flask, request, render_template, url_for, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import openai

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_api_key"

# -----------------------------------------------------------------------------
# JSON TABLE LOADING
# -----------------------------------------------------------------------------
def load_tables_from_json(json_path: Path):
    """Load tables from a JSON file and return a list of table documents."""
    with open(json_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    table_docs = []
    for i, table in enumerate(tables):
        # Combine all relevant information into a single document
        combined_content = f"""
Title: {table.get('title', '')}
Abstract: {table.get('abstract', '')}
Caption: {table.get('caption', '')}
Context: {table.get('context', '')}
Table Content: {table.get('table_html', '')}
"""
        
        # Create a document-like object
        from langchain.schema import Document
        doc = Document(
            page_content=combined_content.strip(),
            metadata={
                "source": json_path.name,
                "title": table.get('title', ''),
                "caption": table.get('caption', ''),
                "table_index": i,
                "table_html": table.get('table_html', ''),
                "paper_title": table.get('title', '')
            }
        )
        table_docs.append(doc)
    
    print("number of tables for ", json_path.name, ": ", len(table_docs))
    return table_docs

# -----------------------------------------------------------------------------
# INDEXING UTILITIES
# -----------------------------------------------------------------------------
def load_papers(json_dir: Path) -> List:
    """Load every table from JSON files in *json_dir* and return LangChain Documents."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    print(f"[Indexer] Loading tables from {json_dir}")
    
    for json_path in json_dir.glob("*.json"):
        if json_path.name == ".DS_Store":
            continue
        print(f"[Indexer] Processing {json_path.name}")
        table_docs = load_tables_from_json(json_path)
        docs.extend(table_docs)
    
    print(f"[Indexer] Loaded {len(docs)} tables total")
    # return splitter.split_documents(docs)
    return docs  # Don't split documents - each table should remain as one unit

def build_vector_store(docs, embeddings_model, index_path: Path):
    """Create FAISS index and save to *index_path*."""
    print("[Indexer] Building FAISS index … this may take a while…")
    vs = FAISS.from_documents(docs, embeddings_model)
    vs.save_local(str(index_path))
    print(f"[Indexer] Saved index to {index_path}")
    return vs

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def create_app(json_dir: str, k: int, rebuild: bool):
    load_dotenv()
    json_dir = Path(json_dir)
    idx_dir = json_dir / "faiss_index"

    embeddings = OpenAIEmbeddings()

    if rebuild or not idx_dir.exists():
        docs = load_papers(json_dir)
        vector_store = build_vector_store(docs, embeddings, idx_dir)
    else:
        vector_store = FAISS.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
        print(f"[Indexer] Loaded existing index from {idx_dir}")

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def home():
        answer = docs = claim = None
        if request.method == "POST":
            claim = request.form["claim"].strip()
            if claim:
                # Direct retrieval without LLM processing
                retrieved_docs = retriever.get_relevant_documents(claim)
                docs = []
                for d in retrieved_docs:
                    meta = d.metadata
                    docs.append({
                        "title": meta.get("paper_title", ""),
                        "source": meta.get("source", ""),
                        "caption": meta.get("caption", ""),
                        "table_index": meta.get("table_index", 0),
                        "table_html": meta.get("table_html", ""),
                        "page_content": d.page_content
                    })
        return render_template("index.html", answer=answer, docs=docs, k=k, claim=claim)

    def fact_check_with_gpt41(claim, table):
        prompt = f'''
You are a scientific fact-checker. Given a user claim and a scientific table (with its paper title, abstract, caption, context, and table content), determine if the table supports or refutes the claim. 

Return a JSON object with:
- "label": one of ["Supported", "Refuted", "Partially_Supported", "Partially_Refuted", "Unverified", "Not_Relevant"]
- "explanation": a concise explanation for your decision

User Claim: {claim}

Paper Title: {table.get('title','')}
Abstract: {table.get('abstract','')}
Caption: {table.get('caption','')}
Context: {table.get('context','')}
Table Content (HTML): {table.get('table_html','')}
'''
        try:
            response = openai.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a scientific fact-checker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            # Try to extract JSON
            import re, json
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {"label": "Unverified", "explanation": content}
        except Exception as e:
            return {"label": "Unverified", "explanation": f"Error: {str(e)}"}

    @app.route('/fact_check', methods=['POST'])
    def fact_check():
        data = request.get_json() or {}
        claim = data.get('claim')
        table = data.get('table')
        if not claim or not table:
            return jsonify({'error': 'Missing claim or table'}), 400
        result = fact_check_with_gpt41(claim, table)
        return jsonify(result)

    return app

# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientific Tables RAG Flask Demo")
    parser.add_argument("--json_dir", default="./papers", help="Folder with JSON files containing tables")
    parser.add_argument("--k", type=int, default=3, help="Top‑K tables to retrieve")
    parser.add_argument("--reindex", action="store_true", default=False, help="Force rebuilding index")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.json_dir, args.k, args.reindex)
    app.run(host=args.host, port=args.port, debug=True)