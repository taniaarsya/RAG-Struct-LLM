# 🧠 RAG-Based AI Solutions: Domain-Specific Question Answering with LLMs

### Turning internal business data into natural language answers using LLMs + vector search

This portfolio presents two real-world use cases where I implemented **RAG-based AI solutions** to solve business problems in different domains:

- 📦 **Product Insight Chatbot (E-Commerce)**  
- 🧾 **Credit Risk Assistant Chatbot (Banking)**  

Each project is built using **LLMs (GPT-4)**, **FAISS vector search**, **Sentence Embeddings**, and an interactive **Streamlit UI** to enable question-answering from internal datasets.

---

## 🚀 Overview

**Retrieval-Augmented Generation (RAG)** is a powerful technique that combines:

- 🔎 **Retrieval** → Search relevant data using vector similarity  
- ✍️ **Generation** → Generate context-aware answers using LLMs  

By integrating both, the model avoids hallucination and ensures accurate, grounded responses using real business data.

---

## 📊 Use Case 1: Product Insight Chatbot (E-Commerce)

**Business Question Examples:**
- “What are customers saying about the new feature?”
- “Show me recent feedback on delivery speed.”
- “Which product categories are seeing lower satisfaction?”

🎥 [Demo Video - Product Insight Chatbot](https://drive.google.com/file/d/1j5xA_ewWbDlBO5gKWNzxMdP_MPedlmbL/view?usp=sharing)

<details>
<summary>🧠 Technical Stack</summary>

- Sentence Embedding: `paraphrase-MiniLM-L6-v2`  
- Vector Store: `FAISS`  
- LLM: `OpenAI GPT-4`  
- UI: `Streamlit`  
</details>

---

## 🧾 Use Case 2: Credit Risk Assistant Chatbot (Banking)

**Business Question Examples:**
- “What does a high-risk customer profile usually look like?”
- “How does income affect repayment behavior?”
- “Give an example of a borrower who defaulted despite a high credit score.”

🎥 [Demo Video - Credit Risk Assistant](https://drive.google.com/file/d/1a7ePYVpWQtAvHrWui9XdKmcyDnUmi0u3/view?usp=sharing)

<details>
<summary>🧠 Technical Stack</summary>

- Sentence Embedding: `paraphrase-MiniLM-L6-v2`  
- Vector Store: `FAISS`  
- LLM: `OpenAI GPT-4`  
- UI: `Streamlit`  
</details>

---

## 🧱 Architecture

```text
User Query
   │
   ▼
Vectorization (Embedding)
   │
   ▼
Similarity Search (FAISS)
   │
   ▼
Top-k Context Pass to GPT-4
   │
   ▼
Generated Answer Displayed in Streamlit
```
---
### 📌 Notes
Ensure you have an OpenAI API key set as an environment variable:

- OPENAI_API_KEY=your_api_key
- Works best with tabular and textual datasets that need contextual interpretation.
- This project is for demonstration purposes and can be extended to production-level use cases.

---
### 🚀 Why This Matters
These tools help teams interact with complex business data using natural language, making insights more accessible without needing SQL, dashboards, or technical skills.
They showcase the potential of combining LLMs with internal business intelligence in real-world, secure, and explainable ways.

---

### 👩🏻‍💻 About Me
- 🔗 [LinkedIn](https://www.linkedin.com/in/taniaarsya/)  
- 📫 Email: taniaarsyaa@gmail.com  

