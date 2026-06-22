"""
Clinical NER & Entity Linking — Flask Web App
app.py — main application file
"""

import os
import json
import pickle
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer

# ── APP SETUP ────────────────────────────────────────────────────
app = Flask(__name__)

# ── LOAD CONFIG ──────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path) as f:
    config = json.load(f)

# ── LABEL MAP ────────────────────────────────────────────────────
ID2LABEL = {
    0: 'O',
    1: 'B-Chemical',
    2: 'I-Chemical',
    3: 'B-Disease',
    4: 'I-Disease'
}

# ── LOAD ALL MODELS AT STARTUP ───────────────────────────────────
# We load once when Flask starts — not on every request
# This keeps response time fast

print("Loading NER tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config['model_path'],
    local_files_only=True
)

print("Loading NER model...")
ner_model = AutoModelForTokenClassification.from_pretrained(
    config['model_path'],
    num_labels=5,
    ignore_mismatched_sizes=True,
    local_files_only=True
)
ner_model.eval()
device = torch.device('cpu')
ner_model = ner_model.to(device)

print("Loading sentence transformer...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading MESH embeddings...")
with open(config['mesh_embed_path'], 'rb') as f:
    embed_data = pickle.load(f)
mesh_ids        = embed_data['mesh_ids']
mesh_names      = embed_data['mesh_names']
mesh_embeddings = embed_data['mesh_embeddings']

print("All models loaded — Flask is ready!\n")


# ── LINKER FUNCTION ───────────────────────────────────────────────
def link_to_mesh(span_text, top_k=3):
    """Find top_k MESH concepts matching the span text."""
    span_emb  = embedder.encode([span_text], convert_to_numpy=True)
    span_norm = span_emb  / np.linalg.norm(span_emb,  axis=1, keepdims=True)
    mesh_norm = mesh_embeddings / np.linalg.norm(mesh_embeddings, axis=1, keepdims=True)
    sims      = np.dot(mesh_norm, span_norm.T).flatten()
    top_idx   = np.argsort(sims)[::-1][:top_k]
    return [
        {
            'mesh_id'  : mesh_ids[i],
            'mesh_name': mesh_names[i],
            'score'    : round(float(sims[i]), 4)
        }
        for i in top_idx
    ]


# ── PIPELINE FUNCTION ─────────────────────────────────────────────
def run_pipeline(text):
    """
    Full NER + linking pipeline.
    Input : raw clinical text string
    Output: list of entity dicts with span, type, mesh_id, score
    """
    # Tokenize
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].tolist()
    tokens         = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # NER predictions
    with torch.no_grad():
        outputs     = ner_model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits[0], dim=-1).tolist()

    # Reconstruct word-level tokens (merge ## subwords)
    word_tokens = []
    i = 0
    while i < len(tokens):
        token      = tokens[i]
        pred       = predictions[i]
        char_start = offset_mapping[i][0]
        char_end   = offset_mapping[i][1]

        if char_start == char_end:   # special token
            i += 1
            continue

        if token.startswith('##'):   # subword — extend previous word
            if word_tokens:
                word_tokens[-1]['char_end'] = char_end
            i += 1
            continue

        # First subword of a real word — merge ahead
        j = i + 1
        while j < len(tokens) and tokens[j].startswith('##'):
            char_end = offset_mapping[j][1]
            j += 1

        word_tokens.append({
            'label'     : ID2LABEL[pred],
            'char_start': char_start,
            'char_end'  : char_end
        })
        i = j

    # Group B/I labels into spans
    spans         = []
    current_start = None
    current_end   = None
    current_type  = None

    for wt in word_tokens:
        label = wt['label']
        if label.startswith('B-'):
            if current_start is not None:
                spans.append({
                    'text': text[current_start:current_end].strip(),
                    'type': current_type,
                    'char_start': current_start,
                    'char_end'  : current_end
                })
            current_start = wt['char_start']
            current_end   = wt['char_end']
            current_type  = label[2:]
        elif label.startswith('I-') and current_start is not None:
            current_end = wt['char_end']
        else:
            if current_start is not None:
                spans.append({
                    'text': text[current_start:current_end].strip(),
                    'type': current_type,
                    'char_start': current_start,
                    'char_end'  : current_end
                })
            current_start = None
            current_end   = None
            current_type  = None

    if current_start is not None:
        spans.append({
            'text': text[current_start:current_end].strip(),
            'type': current_type,
            'char_start': current_start,
            'char_end'  : current_end
        })

    # Link each span to MESH
    results = []
    for span in spans:
        if not span['text']:
            continue
        top_match = link_to_mesh(span['text'], top_k=1)[0]
        results.append({
            'span'      : span['text'],
            'type'      : span['type'],
            'char_start': span['char_start'],
            'char_end'  : span['char_end'],
            'mesh_id'   : top_match['mesh_id'],
            'mesh_name' : top_match['mesh_name'],
            'score'     : top_match['score']
        })

    return results


# ── ROUTES ───────────────────────────────────────────────────────
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    """
    Receive clinical note text from the frontend,
    run the pipeline, return results as JSON.
    """
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        results = run_pipeline(text)
        return jsonify({
            'entities': results,
            'count'   : len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── RUN ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
