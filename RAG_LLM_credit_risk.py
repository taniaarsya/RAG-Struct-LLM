# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import openai

# # ====== SETUP ======
# @st.cache_resource
# def load_model():
#     return SentenceTransformer("paraphrase-MiniLM-L6-v2")

# model = load_model()

# def build_faiss_index(texts):
#     embeddings = model.encode(texts)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings).astype("float32"))
#     return index, embeddings

# def retrieve(query, index, df, embeddings, top_k=5):
#     query_vec = model.encode([query])
#     D, I = index.search(np.array(query_vec).astype("float32"), top_k)
#     return df.iloc[I[0]]

# def generate_answer(query, context, api_key):
#     try:
#         openai.api_key = api_key
#         system_message = "Kamu adalah asisten cerdas yang menjelaskan penyebab risiko gagal bayar berdasarkan data yang diberikan."
#         user_message = f"""
#         Pertanyaan: {query}

#         Data yang relevan:
#         {context}
#         """
#         response = openai.ChatCompletion.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.7,
#             max_tokens=1000
#         )
#         return response.choices[0].message["content"].strip()
#     except Exception as e:
#         return f"⚠️ Terjadi error saat memanggil API: {str(e)}"

# # ====== STREAMLIT UI ======
# st.title("💳 RAG Credit Risk Analyzer")

# # ====== SIDEBAR ======
# st.sidebar.header("⚙️ Pengaturan")
# uploaded_file = st.sidebar.file_uploader("📂 Upload file CSV", type="csv")
# input_api_key = st.sidebar.text_input("🔑 Masukkan OpenAI API Key", type="password")
# activate_api = st.sidebar.button("🔒 Aktifkan API Key")

# if "api_key" not in st.session_state:
#     st.session_state.api_key = None
# if "history" not in st.session_state:
#     st.session_state.history = []

# if activate_api and input_api_key:
#     st.session_state.api_key = input_api_key
#     st.sidebar.success("✅ API Key aktif!")

# if st.sidebar.button("🗑️ Hapus Riwayat"):
#     st.session_state.history = []
#     st.sidebar.success("Riwayat berhasil dihapus!")

# # ====== MAIN ======
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # Tambahkan kolom turunan jika belum ada
#     if "AGE" not in df.columns and "DAYS_BIRTH" in df.columns:
#         df["AGE"] = (-df["DAYS_BIRTH"]) // 365
#     if "YEARS_EMPLOYED" not in df.columns and "DAYS_EMPLOYED" in df.columns:
#         df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"]) // 365

#     st.subheader("🧩 Pilih Kolom untuk Membuat Narasi Kredit")
#     selected_columns = st.multiselect(
#         "Kolom yang akan digunakan untuk membuat narasi:",
#         options=df.columns.tolist(),
#         default=[
#             "SK_ID_CURR", "TARGET", "CODE_GENDER", "CNT_CHILDREN", "NAME_FAMILY_STATUS",
#             "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT",
#             "AMT_ANNUITY", "AMT_GOODS_PRICE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
#             "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2",
#             "EXT_SOURCE_3", "AGE", "YEARS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE",
#             "BUREAU_AMT_CREDIT_SUM", "BUREAU_AMT_CREDIT_SUM_DEBT", 
#             "PREV_AMT_CREDIT_MAX", "PREV_AMT_ANNUITY_MAX", "INS_AMT_PAYMENT_SUM", "CC_AMT_BALANCE"
#         ]
#     )

#     if not selected_columns:
#         st.warning("⚠️ Pilih minimal satu kolom.")
#         st.stop()

#     st.write("📄 Pratinjau Data Terpilih")
#     st.dataframe(df[selected_columns].head())

#     def transform_to_narrative(df, selected_columns):
#         df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
#         return df

#     query = st.text_input("❓ Ajukan pertanyaan terkait pola risiko")
#     run_query = st.button("🚀 Analisis Pertanyaan")

#     if run_query and st.session_state.api_key:
#         try:
#             df = transform_to_narrative(df, selected_columns)
#             index, embeddings = build_faiss_index(df["text"].tolist())

#             with st.spinner("🔍 Mengambil konteks dari narasi..."):
#                 results = retrieve(query, index, df, embeddings, top_k=5)
#                 context = "\n".join(results["text"].tolist())

#             with st.spinner("🤖 Menghasilkan jawaban..."):
#                 answer = generate_answer(query, context, st.session_state.api_key)

#             st.subheader("💬 Jawaban:")
#             st.success(answer)
#             st.session_state.history.append((query, answer))

#         except Exception as e:
#             st.error(f"Terjadi error saat proses analisis: {str(e)}")

#     elif run_query and not st.session_state.api_key:
#         st.warning("🔐 Silakan aktifkan API Key terlebih dahulu.")

# # ====== HISTORY ======
# if st.session_state.history:
#     st.subheader("📚 Riwayat Pertanyaan & Jawaban")
#     for i, (q, a) in enumerate(reversed(st.session_state.history[-5:]), 1):
#         with st.expander(f"❓ #{i}: {q}"):
#             st.markdown(f"💬 **Jawaban:** {a}")

# Credit Risk RAG Streamlit App

# === Import dan Setup ===
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# === Backend RAG & LLM ===

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def build_faiss_index_cosine(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve(query, index, df, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding.astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    result_df = df.iloc[indices[0]].copy()
    result_df["similarity_score"] = scores[0]
    return result_df

def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_message = "You are an intelligent assistant answering questions based on provided credit risk data."
    user_message = f"""Question: {query}

Relevant Data:
{context}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0]['message']["content"].strip()

def transform_data(df, selected_columns):
    df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    return df

# === UI ===
st.set_page_config(page_title="Credit Risk RAG", layout="wide")
st.title("💳 Credit Risk RAG Assistant")

# === Sidebar ===
st.sidebar.markdown("<h2 style='text-align: center;'>Configuration</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("📄 Upload CSV File", type='csv')
input_api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type='password')
button_api = st.sidebar.button("Activate API Key")

if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("✅ API Key Activated")

# === Main Content ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Select Columns for Analysis")
    selected_columns = st.multiselect("Select columns to include in text embedding:", options=df.columns.tolist(), default=df.columns.tolist())
    
    if not selected_columns:
        st.warning("⚠️ Please select at least one column.")
        st.stop()
    
    st.dataframe(df[selected_columns])

    query = st.text_input("🗨️ Enter your question:")
    run_query = st.button("🔍 Generate Answer")

    if run_query and st.session_state.api_key:
        try:
            df = transform_data(df, selected_columns)
            index, _ = build_faiss_index_cosine(df["text"].tolist())
            with st.spinner("🔎 Retrieving relevant records..."):
                results = retrieve(query, index, df)
                context = "\n".join(results["text"].tolist())
            with st.spinner("✍️ Generating answer..."):
                answer = generate_answer(query, context, st.session_state.api_key)
            st.subheader("💬 Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    elif run_query and not st.session_state.api_key:
        st.warning("🔐 Please activate your API key first.")
else:
    st.info("📂 Please upload a CSV file to begin.")
