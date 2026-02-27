🧴 Cosmetic Chemical Intelligence

    Cosmetic Chemical Intelligence is a Retrieval-Augmented Generation (RAG) based AI system designed to analyze cosmetic ingredient datasets and provide grounded, explainable responses through an interactive Streamlit interface.
    The application converts structured cosmetic data into vector embeddings, stores them in a persistent Chroma vector database, and retrieves relevant records based on semantic similarity. Retrieved metadata is validated before being incorporated into a hybrid LLM prompt to reduce hallucinations and ensure factual grounding.

The assistant supports:

    1)Chemical-to-product mapping
    2)Ingredient safety explanations
    3)Brand-level queries
    4)Context-aware follow-up questions

    This project demonstrates practical RAG architecture, prompt engineering strategies, metadata validation, and explainable AI response generation in a real-world domain.

✨ Features

    🔍 Retrieve products containing specific chemicals
    🧠 Hybrid RAG architecture (Chroma + OpenAI)
    🧴 Safety and usage explanations for ingredients
    📊 Metadata-aware retrieval logic
    🚫 Reduced hallucination through grounding checks
    💬 Context-aware follow-up support
    
  🏗 Tech Stack

    Python
    Streamlit
    OpenAI (Embeddings + LLM)
    Chroma Vector Database
    Pandas

🧠 Architecture Overview

<img width="1577" height="456" alt="image" src="https://github.com/user-attachments/assets/372ae493-9b89-47ea-9f51-f17cec06dbff" />


📂 Project Structure

    cosmetic-ai-project/
    ├── app.py
    ├── rag_pipeline.py
    |── requirements.txt
    ├── data/
    │   └── chemicals.csv
    └── README.md

📌 Highlights

    Implements end-to-end RAG workflow
    Hybrid reasoning (database + general knowledge fallback)
    Safety-question detection logic
    Structured and explainable AI responses

📄 License

    MIT License
