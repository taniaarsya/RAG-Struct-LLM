# === Import and Setup ===
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# === RAG Backend & LLM ===

## Load SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

## FAISS Cosine Similarity Index
def build_faiss_index_cosine(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

## Retrieval Logic
def retrieve(query, index, df, top_k=None):
    return df  # simple return, or implement similarity search if needed

## Generate answer from OpenAI API
def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_message = "You are an intelligent assistant that answers questions based on the provided data."
    user_message = f"""
    Question: {query}

    Relevant data:
    {context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0]['message']["content"].strip()

## Combine selected columns into one text field
def transform_data(df, selected_columns):
    df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    return df

# === Streamlit UI ===

## Title
st.title("📊 Credit Risk Analysis Assistant")

## Sidebar
st.sidebar.markdown(
    "<h2 style='text-align: center;'>🛠️ Configuration</h2>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("📎 Upload CSV File", type='csv')
input_api_key = st.sidebar.text_input("🔐 Enter OpenAI API Key", type='password')
button_api = st.sidebar.button("Activate API Key")

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("✅ API Key Activated")

## Main Functionality
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("✅ Select Columns")
    selected_columns = st.multiselect(
        "Choose columns to include in the analysis:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )

    if not selected_columns:
        st.warning("⚠️ Please select at least one column.")
        st.stop()

    st.dataframe(df[selected_columns].head(50))

    query = st.text_input("💬 Ask a question about the dataset:")
    run_query = st.button("🔍 Get Answer")

    if run_query and st.session_state.api_key:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index_cosine(df['text'].tolist())

            with st.spinner("🔎 Searching for relevant data..."):
                results = retrieve(query, index, df)
                context = "\n".join(results["text"].tolist())

            with st.spinner("🧠 Generating answer..."):
                answer = generate_answer(query, context, st.session_state.api_key)

            st.subheader("📢 Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    elif run_query and not st.session_state.api_key:
        st.warning("🔐 Please activate your API Key first.")
    else:
        st.info("📂 Please upload a CSV file to start.")