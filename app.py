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
        "1026.43",
        "ability-to-repay",
        "ability to repay",
        "repayment ability",
        "income",
        "assets",
        "employment",
        "debt-to-income",
        "debt to income",
        "monthly payment",
        "simultaneous loans",
        "mortgage-related obligations",
        "credit history",
        "qualified mortgage"
    ]

    penalty_keywords = [
        "hmda",
        "regulation c",
        "community reinvestment act",
        "cra",
        "reporting",
        "disclosure",
        "high-cost mortgage",
        "hoepa",
        "average prime offer rate",
        "annual percentage rate",
        "fannie mae",
        "selling guide"
    ]

    text_lower = str(text).lower()
    query_lower = str(query).lower()

    score = 0

    for kw in keywords:
        if kw in text_lower:
            score += 1
        if kw in query_lower and kw in text_lower:
            score += 2

    for kw in penalty_keywords:
        if kw in text_lower:
            score -= 10

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
You are a mortgage policy explanation assistant.

Mortgage Case and Question:
{case_text}

Retrieved Policy Evidence:
{evidence}

Task:
Answer the user's question in 5 to 7 sentences.

Instructions:
- Answer the specific question directly.
- Write in plain English.
- Explain the meaning of the mortgage metrics in the case.
- Explain why those metrics matter for repayment-capacity review.
- Use the retrieved policy evidence as support.
- Focus on helping a non-technical reader understand the result.
- Summarize policy concepts in simple language.
- Do not quote regulation text.
- Do not mention CFR numbers, section numbers, appendix numbers, paragraph numbers, legal citations, or policy document references.
- Do not copy policy wording directly.
- Do not say "According to the policy" or "The regulation states".
- Do not mention retrieved excerpts.
- Do not use bullet points or headings.
- Keep the explanation factual and grounded in the provided evidence.
- If the evidence does not provide enough information for a definitive conclusion, clearly state that additional information would be needed.

The audience is a mortgage applicant with no legal or regulatory background.
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
