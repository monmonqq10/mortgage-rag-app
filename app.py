import os
import pickle
import gdown
import gradio as gr
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient

POLICY_FILE_ID = "1PbVcl42w32sfVyE5eqhwR331OuDR1TDb"
EMBED_FILE_ID = "1r1ohBm1dpurAw9dt0-elBaqfi_Dl12rd"

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
    model="mistralai/Mistral-Nemo-Instruct-2407",
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

def retrieve_top_k(query_text, k=5):
    query_embedding = get_query_embedding(query_text)
    scores = cosine_scores(query_embedding, chunk_embeddings)
    top_indices = scores.argsort()[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "source": policy_chunks_df.iloc[idx]["source"],
            "text": policy_chunks_df.iloc[idx]["text"],
            "score": float(scores[idx])
        })
    return results

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
- Ignore unrelated regulatory thresholds or background percentages.
- Do not use headings, bullet points, numbering, labels, or markdown.
- Do not start with "Answer:"
"""

def generate_response(case_text):
    retrieved = retrieve_top_k(case_text, k=5)
    prompt = build_rag_prompt(case_text, retrieved)

    response = llm_client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.0
    )
    
    response = response.choices[0].message.content

    evidence_text = ""
    for i, r in enumerate(retrieved, 1):
        evidence_text += f"\n\nPolicy Excerpt {i}\nSource: {r['source']}\nScore: {r['score']:.4f}\n{r['text'][:800]}"

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
