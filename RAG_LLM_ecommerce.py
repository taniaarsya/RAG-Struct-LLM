# RAG LLM for E-Commerce Product Analysis
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# ---------------------- SETUP ----------------------
st.set_page_config(page_title="RAG Product Assistant", layout="wide")
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4A90E2;'>🛍️ E-Commerce Product Intelligence (RAG LLM)</h1>
        <p>Ask anything about your product sales data — Powered by AI + FAISS</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------- MODEL SETUP ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def build_faiss_index(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve(query, index, df, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding.astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    result_df = df.iloc[indices[0]].copy()
    result_df['similarity_score'] = scores[0]
    return result_df

def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_msg = "You are an intelligent assistant that answers based on the provided product data."
    user_msg = f"""
    Question: {query}

    Relevant Data:
    {context}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message["content"]

def prepare_text_column(df, selected_cols):
    df["text"] = df[selected_cols].astype(str).agg(" | ".join, axis=1)
    return df

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("🔧 Configuration")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type='csv')
input_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
activate_key = st.sidebar.button("Activate API Key")

if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if activate_key and input_api_key:
    st.session_state.api_key = input_api_key
    st.sidebar.success("API Key is active ✅")

# ---------------------- MAIN AREA ----------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip()

        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head())

        st.markdown("### 🧩 Select Columns to Analyze")
        selected_columns = st.multiselect(
            "Choose columns for semantic embedding:",
            options=df.columns.tolist(),
            default=['Product Name', 'Category', 'Sub-Category', 'Sales', 'Discount', 'Profit']
        )

        if not selected_columns:
            st.warning("⚠️ Please select at least one column.")
            st.stop()

        st.markdown("#### 💬 Ask a Question")
        query = st.text_input("What would you like to know about the products?", key="user_query")
        run_button = st.button("🔍 Get Answer", use_container_width=True)

        if run_button and st.session_state.api_key:
            with st.spinner("Encoding data and searching..."):
                df = prepare_text_column(df, selected_columns)
                index, _ = build_faiss_index(df['text'].tolist())
                results = retrieve(query, index, df)
                context = "\n".join(results['text'].tolist())

            with st.spinner("Generating answer from GPT..."):
                answer = generate_answer(query, context, st.session_state.api_key)

            st.markdown("### ✅ Top Relevant Entries")
            st.dataframe(results.drop(columns='similarity_score'))

            st.markdown("### 💡 Answer")
            st.markdown(
                "<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>",
                unsafe_allow_html=True
            )
            st.success(answer)
            st.markdown("</div>", unsafe_allow_html=True)

        elif run_button and not st.session_state.api_key:
            st.error("🔐 Please activate your OpenAI API key first.")
    else:
        st.info("📥 Upload a CSV file from the sidebar to begin.")


# # Streamlit Chat App with Document Upload (Public Ready)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import openai

# # ---------------------- CONFIG ----------------------
# st.set_page_config(page_title="AI Document Chat Assistant", layout="wide")
# st.markdown("""
#     <div style='text-align: center;'>
#         <h1 style='color: #4A90E2;'>🧠 AI Document Chat Assistant</h1>
#         <p>Interact with your dataset in natural language — built on Retrieval-Augmented Generation (RAG)</p>
# <p style='font-size: 14px; color: gray;'>Built with Sentence-Transformers · FAISS · OpenAI GPT-4</p>
#     </div>
# """, unsafe_allow_html=True)

# # ---------------------- MODEL SETUP ----------------------
# @st.cache_resource
# def load_model():
#     return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# model = load_model()

# def build_faiss_index(texts):
#     embeddings = model.encode(texts, convert_to_numpy=True)
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     embeddings = embeddings.astype('float32')
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index, embeddings

# def retrieve(query, index, df, top_k=3):
#     query_embedding = model.encode([query], convert_to_numpy=True)
#     query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
#     query_embedding = query_embedding.astype("float32")
#     scores, indices = index.search(query_embedding, top_k)
#     result_df = df.iloc[indices[0]].copy()
#     result_df['similarity_score'] = scores[0]
#     return result_df

# def generate_answer(query, context, api_key):
#     openai.api_key = api_key
#     system_msg = "You are an intelligent assistant that answers based on the provided product data."
#     user_msg = f"""
#     Question: {query}

#     Relevant Data:
#     {context}
#     """
#     response = openai.ChatCompletion.create(
#         model="gpt-4-1106-preview",
#         messages=[
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": user_msg}
#         ],
#         temperature=0.5,
#         max_tokens=1000
#     )
#     return response.choices[0].message["content"]

# def prepare_text_column(df, selected_cols):
#     df["text"] = df[selected_cols].astype(str).agg(" | ".join, axis=1)
#     return df

# # ---------------------- SIDEBAR ----------------------
# st.sidebar.header("🔧 Configuration")
# uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type='csv')
# input_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
# activate_key = st.sidebar.button("Activate API Key")

# if 'api_key' not in st.session_state:
#     st.session_state.api_key = None
# if activate_key and input_api_key:
#     st.session_state.api_key = input_api_key
#     st.sidebar.success("API Key is active ✅")

# # ---------------------- MAIN AREA ----------------------
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
#         df.columns = df.columns.str.strip()

#         st.markdown("### 📊 Data Preview")
#         st.dataframe(df.head())

#         st.markdown("### 🧩 Select Columns to Analyze")
#         selected_columns = st.multiselect(
#             "Choose columns for semantic embedding:",
#             options=df.columns.tolist(),
#             default=['Product Name', 'Category', 'Sub-Category', 'Sales', 'Discount', 'Profit']
#         )

#         if not selected_columns:
#             st.warning("⚠️ Please select at least one column.")
#             st.stop()

#         st.markdown("#### 💬 Ask a Question")
#         query = st.text_input("What would you like to know about the data?", key="user_query")
#         run_button = st.button("🔍 Get Answer", use_container_width=True)

#         if run_button and st.session_state.api_key:
#             with st.spinner("Encoding data and searching..."):
#                 df = prepare_text_column(df, selected_columns)
#                 index, _ = build_faiss_index(df['text'].tolist())
#                 results = retrieve(query, index, df)
#                 context = "\n".join(results['text'].tolist())

#             with st.spinner("Generating answer from GPT..."):
#                 answer = generate_answer(query, context, st.session_state.api_key)

#             st.markdown("### ✅ Top Relevant Entries")
#             st.dataframe(results.drop(columns='similarity_score'))

#             st.markdown("### 💡 Answer")
#             st.markdown(
#                 "<div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>",
#                 unsafe_allow_html=True
#             )
#             st.success(answer)
#             st.markdown("</div>", unsafe_allow_html=True)

#         elif run_button and not st.session_state.api_key:
#             st.error("🔐 Please activate your OpenAI API key first.")
#     else:
#         st.info("📥 Upload a CSV file from the sidebar to begin.")