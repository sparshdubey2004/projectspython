# --- !! This fix MUST be at the very top !! ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- End fix ---

import sys
import pdfplumber
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication

# Import Hugging Face transformers
from transformers import pipeline, AutoTokenizer

# --- Worker Thread for SUMMARIZATION ---
# This is upgraded to handle long text via chunking
class SummarizeWorker(QThread):
    """
    Worker thread to run the summarization task
    to prevent the GUI from freezing.
    """
    # Signal to emit when summarization is done
    finished = pyqtSignal(str)

    def __init__(self, text, summarizer, tokenizer):
        super().__init__()
        self.text = text
        self.summarizer = summarizer
        self.tokenizer = tokenizer
        # Get the model's max token limit (e.g., 1024 for BART)
        self.model_max_length = self.tokenizer.model_max_length
        # We leave a "buffer" for safety
        self.chunk_size = self.model_max_length - 50 

    def run(self):
        try:
            # --- 1. Split text into chunks ---
            tokens = self.tokenizer.tokenize(self.text)
            if len(tokens) <= self.model_max_length:
                # Text is short enough, summarize normally
                print("Text is short, using single-pass summarization.")
                summary = self.summarize_chunk(self.text)
                self.finished.emit(summary)
                return

            print(f"Text is long ({len(tokens)} tokens). Starting chunked summarization.")
            
            # Re-create chunks based on *token* length
            token_chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                # Convert tokens back to a string
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                token_chunks.append(chunk_text)

            print(f"Split text into {len(token_chunks)} chunks.")

            # --- 2. "Map" Step: Summarize each chunk ---
            chunk_summaries = []
            for i, chunk in enumerate(token_chunks):
                print(f"Summarizing chunk {i+1} / {len(token_chunks)}...")
                chunk_summary = self.summarize_chunk(chunk)
                chunk_summaries.append(chunk_summary)

            # --- 3. "Reduce" Step: Combine summaries and summarize again ---
            combined_summary_text = " ".join(chunk_summaries)
            
            # Check if the *combined summaries* are also too long
            combined_tokens = self.tokenizer.tokenize(combined_summary_text)
            if len(combined_tokens) <= self.model_max_length:
                print("Summarizing the combined summaries (final pass)...")
                final_summary = self.summarize_chunk(combined_summary_text)
            else:
                # If the combined summaries are *still* too long, just join them.
                print("Combined summaries are still too long. Joining them.")
                final_summary = combined_summary_text

            self.finished.emit(final_summary)

        except Exception as e:
            self.finished.emit(f"Error during summarization: {e}")

    def summarize_chunk(self, text_chunk):
        """Helper function to run the summarizer pipeline on one chunk."""
        # Use min_length=10 to avoid "out of range" on short chunks
        summary = self.summarizer(
            text_chunk, 
            max_length=400,  # Max length of the *output* summary
            min_length=100,   # Min length of the *output* summary
            do_sample=False,
            truncation=True  # Truncation is still needed *on each chunk*
        )
        return summary[0]['summary_text']


# --- Main Application Window ---
class SummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- MODEL LOADING ---
       # --- MODEL LOADING ---
        self.model_name = "sshleifer/distilbart-cnn-12-6" 
        
        print(f"Loading summarization pipeline: {self.model_name}")
        try:
            self.summarizer = pipeline("summarization", model=self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(-1)
        # --- END MODEL LOADING ---

        self.full_text = ""
        self.worker_thread = None
        self.init_ui()

    def init_ui(self):
        """Creates the GUI layout and widgets."""
        self.setWindowTitle("PDF Summarizer (PyQt6 + Hugging Face)")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        self.btn_browse = QPushButton("Browse for PDF")
        self.btn_browse.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.btn_browse)

        self.lbl_filepath = QLabel("No file selected.")
        layout.addWidget(self.lbl_filepath)

        layout.addWidget(QLabel("Original Text (from PDF):"))
        self.txt_original = QTextEdit()
        self.txt_original.setReadOnly(True)
        layout.addWidget(self.txt_original)

        self.btn_summarize = QPushButton("Summarize Text")
        self.btn_summarize.clicked.connect(self.run_summarization_task)
        self.btn_summarize.setEnabled(False)
        layout.addWidget(self.btn_summarize)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate mode
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Summary:"))
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        layout.addWidget(self.txt_summary)

        self.setLayout(layout)
        self.show()

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )
        
        if file_name:
            self.lbl_filepath.setText(file_name)
            self.txt_original.setText("Extracting text from PDF...")
            QCoreApplication.processEvents() # Force GUI to update
            
            try:
                all_text = ""
                with pdfplumber.open(file_name) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            all_text += page_text + "\n"
                
                self.full_text = all_text
                self.txt_original.setText(self.full_text)
                self.btn_summarize.setEnabled(True)
                self.txt_summary.clear()

            except Exception as e:
                self.txt_original.setText(f"Error reading PDF: {e}")
                self.btn_summarize.setEnabled(False)

    def run_summarization_task(self):
        if not self.full_text:
            self.txt_summary.setText("No text to summarize.")
            return

        self.btn_summarize.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.txt_summary.setText("Summarizing... This may take a moment.")
        
        self.worker_thread = SummarizeWorker(self.full_text, self.summarizer, self.tokenizer)
        self.worker_thread.finished.connect(self.on_summarize_finished)
        self.worker_thread.start()

    def on_summarize_finished(self, summary_text):
        self.txt_summary.setText(summary_text)
        self.btn_summarize.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.progress_bar.setVisible(False)


# --- Run the Application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SummarizerApp()
    sys.exit(app.exec())