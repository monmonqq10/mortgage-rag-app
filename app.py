import os
import pickle
import gdown
import gradio as gr
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import CrossEncoder

POLICY_FILE_ID = "16XMoIRCdC9swJ4jLzXdsP3tsnmaOR0i6"
EMBED_FILE_ID = "1TkqaOF2v2K4hb8rSSfIQWs9f4re7yurf"

def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

download_from_drive(POLICY_FILE_ID, "policy_chunks.csv")
download_from_drive(EMBED_FILE_ID, "chunk_embeddings.pkl")

policy_chunks_df = pd.read_csv("policy_chunks.csv")

with open("chunk_embeddings.pkl", "rb") as f:
    chunk_embeddings = pickle.load(f)

chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

HF_TOKEN = os.environ.get("HF_TOKEN")

llm_client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

embed_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HF_TOKEN
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def cosine_scores(query_vec, matrix):
    query_vec = np.array(query_vec, dtype=np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(matrix_norm, query_vec)

def get_query_embedding(query_text):
    result = embed_client.feature_extraction(query_text)
    return np.array(result, dtype=np.float32)

def retrieve_top_k(query_text, k=5, candidate_k=30):
    query_embedding = get_query_embedding(query_text)
    scores = cosine_scores(query_embedding, chunk_embeddings)

    candidate_indices = scores.argsort()[::-1][:candidate_k]

    candidates = []
    for idx in candidate_indices:
        candidates.append({
            "source": policy_chunks_df.iloc[idx]["source"],
            "text": policy_chunks_df.iloc[idx]["text"],
            "cosine_score": float(scores[idx])
        })

    pairs = [(query_text, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    ranked_idx = np.argsort(rerank_scores)[::-1][:k]

    results = []
    for idx in ranked_idx:
        item = candidates[idx].copy()
        item["score"] = float(rerank_scores[idx])
        results.append(item)

    return results

def build_rag_prompt(case_text, retrieved_chunks):
    evidence = "\n\n".join([
        f"[Policy Excerpt {i+1}]\n{chunk['text'][:900]}"
        for i, chunk in enumerate(retrieved_chunks[:5])
    ])

    return f"""
Mortgage case and question:
{case_text}

Retrieved policy evidence:
{evidence}

Write one concise answer in 5 to 7 sentences.

Rules:
1. Answer the specific question directly.
2. Use only the retrieved evidence and the case facts.
3. Do not invent thresholds, approval reasons, risk levels, or compliance results.
4. If the evidence only lists review factors, say that it identifies factors to consider but does not confirm compliance.
5. If the evidence is insufficient, say that a definitive compliance determination cannot be confirmed from the retrieved evidence.
6. Do not use headings, bullet points, numbering, or markdown in the answer.
7. Do not repeat these rules.
"""

def generate_response(case_text):
    retrieved = retrieve_top_k(case_text, k=5, candidate_k=30)
    prompt = build_rag_prompt(case_text, retrieved)

    response = llm_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a careful mortgage policy-grounded explanation assistant. Answer only the user question. Do not repeat the prompt or rules."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.0
    )

    response = response.choices[0].message.content

    evidence_text = ""
    for i, r in enumerate(retrieved, 1):
        evidence_text += (
            f"\n\nPolicy Excerpt {i}\n"
            f"Source: {r['source']}\n"
            f"Rerank Score: {r['score']:.4f}\n"
            f"Cosine Score: {r['cosine_score']:.4f}\n"
            f"{r['text'][:800]}"
        )

    return response.strip(), evidence_text

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=10, label="Mortgage Case and Question"),
    outputs=[
        gr.Textbox(lines=8, label="Generated Policy-Grounded Explanation"),
        gr.Textbox(lines=15, label="Retrieved Policy Evidence")
    ],
    title="Policy-Grounded Mortgage Explanation RAG System"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
