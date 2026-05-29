import os
import pickle
import gdown
import gradio as gr
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient

POLICY_FILE_ID = "16XMoIRCdC9swJ4jLzXdsP3tsnmaOR0i6"
EMBED_FILE_ID = "1TkqaOF2v2K4hb8rSSfIQWs9f4re7yurf"

def download_from_drive(file_id, output):
    if not os.path.exists(output):
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

def cosine_scores(query_vec, matrix):
    query_vec = np.array(query_vec, dtype=np.float32)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(matrix_norm, query_vec)

def get_query_embedding(query_text):
    result = embed_client.feature_extraction(query_text)
    return np.array(result, dtype=np.float32)

def keyword_score(text, query):
    keywords = [
        "ability-to-repay", "ability to repay", "1026.43",
        "debt-to-income", "debt to income", "income", "assets",
        "monthly payment", "credit history", "employment status",
        "mortgage-related obligations", "mortgage related obligations",
        "simultaneous loans", "qualified mortgage",
        "loan-to-value", "combined loan-to-value",
        "principal residence", "first lien", "subordinate lien",
        "refinancing", "home purchase"
    ]

    penalty_keywords = [
        "high-cost mortgage", "hoepa", "average prime offer rate",
        "annual percentage rate", "percentage points",
        "advertising", "disclosure", "escrow"
    ]

    text_lower = str(text).lower()
    query_lower = str(query).lower()

    score = 0

    for kw in keywords:
        if kw in text_lower:
            score += 1
        if kw in query_lower and kw in text_lower:
            score += 2

    # Penalize unrelated Regulation Z sections for ATR questions
    if "ability" in query_lower or "repayment" in query_lower or "debt-to-income" in query_lower:
        for kw in penalty_keywords:
            if kw in text_lower:
                score -= 2

    return score

def retrieve_top_k(query_text, k=5, candidate_k=30):
    query_embedding = get_query_embedding(query_text)
    scores = cosine_scores(query_embedding, chunk_embeddings)

    candidate_indices = scores.argsort()[::-1][:candidate_k]

    reranked = []
    for idx in candidate_indices:
        text = policy_chunks_df.iloc[idx]["text"]
        source = policy_chunks_df.iloc[idx]["source"]

        cosine = float(scores[idx])
        keyword = keyword_score(text, query_text)

        final_score = (0.75 * cosine) + (0.25 * keyword)

        reranked.append({
            "source": source,
            "text": text,
            "score": final_score,
            "cosine_score": cosine,
            "keyword_score": keyword
        })

    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    return reranked[:k]

def build_rag_prompt(case_text, retrieved_chunks):
    evidence = "\n\n".join([
        f"[Policy Excerpt {i+1}]\n{chunk['text'][:900]}"
        for i, chunk in enumerate(retrieved_chunks[:5])
    ])

    return f"""
You are answering a mortgage policy-grounded explanation question.

Mortgage Case and Question:
{case_text}

Retrieved Policy Evidence:
{evidence}

Task:
Write one concise answer in 5 to 7 sentences.

Start by directly answering the specific question.
Use only the retrieved policy evidence and the most relevant case facts.
Explain how the retrieved evidence supports or limits the answer.
If the retrieved evidence does not provide a clear rule, threshold, or compliance result, state that a definitive compliance determination cannot be confirmed from the provided policy evidence.

Important:
- Focus on the exact question asked.
- Do not repeat all case details.
- Do not give general mortgage background.
- Copy numeric values exactly as shown in the Mortgage Case.
- Do not invent thresholds, rules, risk levels, approval logic, or lender decision reasons.
- Do not say the loan meets or violates a requirement unless the retrieved evidence clearly supports it.
- If retrieved evidence is about HOEPA, high-cost mortgage thresholds, APR trigger thresholds, advertising, escrow, or disclosure rules, do not use it to answer an Ability-to-Repay repayment-capacity question unless the question specifically asks about those topics.
- Ignore unrelated regulatory thresholds or background percentages.
- Do not use headings, bullet points, numbering, labels, or markdown.
- Do not start with "Answer:"
"""

def generate_response(case_text):
    retrieved = retrieve_top_k(case_text, k=5, candidate_k=30)
    prompt = build_rag_prompt(case_text, retrieved)

    response = llm_client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
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
            f"Final Score: {r['score']:.4f}\n"
            f"Cosine Score: {r['cosine_score']:.4f}\n"
            f"Keyword Score: {r['keyword_score']}\n"
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
