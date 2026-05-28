import os
import gdown

POLICY_FILE_ID = "1E2A323P2awpY-Bcwq2EA_n8w4ecHJBkv"
EMBED_FILE_ID = "1GKRPVQE4yMN3HO9FhYFHxImax6zlImbU"

def download_from_drive(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

download_from_drive(POLICY_FILE_ID, "policy_chunks.csv")
download_from_drive(EMBED_FILE_ID, "chunk_embeddings.pkl")

import os
import pickle
import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

policy_chunks_df = pd.read_csv("policy_chunks.csv")

with open("chunk_embeddings.pkl", "rb") as f:
    chunk_embeddings = pickle.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

def retrieve_top_k(query_text, k=5):
    query_embedding = embed_model.encode([query_text], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
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

    response = client.text_generation(
        prompt,
        max_new_tokens=600,
        temperature=0.0,
        do_sample=False
    )

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
