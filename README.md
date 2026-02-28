📄 AI-Powered PDF Summarizer (Abstractive)
A professional desktop application built with Python and PyQt6 that transforms lengthy, complex PDF documents into concise, human-like summaries. By leveraging the BART model and a custom Map-Reduce chunking strategy, this tool handles documents of any length without losing context.

🚀 Key Features
High-Fidelity Extraction: Uses pdfplumber to accurately parse text from complex layouts that usually trip up standard libraries.

Abstractive Summarization: Powered by the facebook/bart-large-cnn model, generating coherent, rewritten summaries rather than just extracting existing sentences.

Smart Chunking: Implements a Map-Reduce logic to process large PDFs that exceed the standard 1024-token limit of LLMs.

Modern Desktop UI: A sleek, responsive interface built with PyQt6 for seamless file handling.

🛠️ Technical Stack
Language: Python 3.x

GUI Framework: PyQt6

PDF Parsing: pdfplumber

NLP Engine: Hugging Face Transformers (BART Model)

Deep Learning: PyTorch

🧠 How It Works
Ingestion: The file is loaded via the PyQt6 interface and parsed using pdfplumber to ensure character metadata and layout are respected.

The "Token" Problem: Most LLMs have a strict input limit. To solve this, I implemented a Map-Reduce strategy:

Map: The text is split into logical chunks.

Summarize: Each chunk is summarized independently by the BART model.

Reduce: The chunk-summaries are combined and processed one final time to create a cohesive global summary.

Output: The final summary is rendered instantly within the application.

🔧 Installation & Setup
Follow these steps to get the environment ready and run the application on your local machine.

📋 Prerequisites
Python 3.8+

RAM: 8GB recommended (for loading the BART model)

🛠️ Step-by-Step Guide
Clone the Repository

Bash
git clone https://github.com/sparshdubey2004/pdf-summarizer.git
cd pdf-summarizer
Create a Virtual Environment (Recommended)

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

#Install Core Dependencies

Bash
pip install PyQt6 pdfplumber transformers torch
Launch the Application

Bash
python pdfsum.py

