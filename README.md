# A PDF Q&A Chatbot Using Gemini

Chat Vista is an interactive web application built using Streamlit that allows users to upload PDF files, extract text from them, and then ask questions based on the content using Google’s Generative AI models. The application uses LangChain and FAISS for text processing and retrieval.

---

## Features

- **PDF Upload**: Upload one or more PDF files for content extraction and analysis.
- **Text Chunking**: Efficiently splits large text into manageable chunks using LangChain.
- **Vector Store**: Uses FAISS to store and search through embeddings generated by Google's Generative AI.
- **Conversational Q&A**: Ask questions based on the provided PDF content and get detailed answers.

---

## Prerequisites

1. **Python**: Ensure you have Python 3.7+ installed.
2. **Google Generative AI API Key**: Obtain an API key from Google to use their Generative AI models.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dinah-kezia/gemini-chatbot-using-google-genai.git
