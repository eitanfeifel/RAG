# RAG
Retrieval-augmented generation (RAG) pipeline, using Bert-Sentence embeddings, faiss vectorstore and Gemini  LLM for Question-Answering based on uploaded documents. 

# Contextual QA System with Semantic Search and Generative AI

This project is a Contextual Question-Answering (QA) system that utilizes semantic search and a generative AI model to answer user queries based on relevant contextual information. The system supports `.txt`, `.docx`, and `.pdf` file formats and allows users to upload documents. It extracts, preprocesses, and chunks the text, creates embeddings, and stores them in a vector database for efficient similarity search. Using the generative model, the system then provides an informative response based on the relevant document chunks.

## Features

- **File Upload and Text Extraction**: Supports `.txt`, `.docx`, and `.pdf` file formats and extracts text using `Tkinter` file dialog.
- **Text Preprocessing and Chunking**: Preprocesses text by removing special characters and whitespace. Offers both preliminary and semantic chunking for optimized storage and search.
- **Vector Database for Semantic Search**: Uses FAISS to store document embeddings and efficiently retrieve relevant chunks based on similarity to user queries.
- **Generative AI-Based Answering**: Integrates with the Gemini API to generate personalized responses based on retrieved document context and user queries.
- **Customizable Parameters**: Users can adjust settings for chunk size, similarity threshold, and generative prompts to tailor the system to specific applications.

## Project Structure

- **`upload_embeddings.py`**: Script to preprocess text files, generate embeddings, and store them in the vector database.
- **`embeddings.py`**: Provides functions to generate embeddings using the Sentence-BERT model.
- **`vector_store.py`**: Implements the vector database using FAISS for efficient similarity search.
- **`query.py`**: Main script where users input a question, and the system retrieves relevant context and generates an answer using the Gemini API.

## Installation

### 1. Clone the Repository

git clone https://github.com/eitanfeifel/RAG.git

### 2. Set Up Virtual Environment and Install Dependencies
python -m venv venv
source venv/bin/activate (MacOS/Linux)
.venv\Scripts\activate` (Windows)
pip install -r requirements.txt

### 3. Set Up API-KEY
setx API_KEY "YOUR_API_KEY" (windows) *Use set instead of setx for temporary use, with setx be sure do close and restart terminal*

export API_KEY="YOUR_API_KEY" (MacOS/linux)

### 4. Run Scripts
python upload_embeddings.py -- (Upload and Process Documents: Run upload_embeddings.py to select files, preprocess, chunk, and store embeddings.)

python query.py -- (Query the System: Run query.py to enter a question, retrieve relevant context, and get an AI-generated answer.)

## Customization Guide

1. **Modify Preprocessing (`upload_embeddings.py`)**  
   The `preprocess_text` function allows you to adjust text cleanup steps, such as character removal and whitespace handling. Adjust `chunk_size` to control the initial chunk size before semantic chunking.

2. **Adjust Semantic Chunking Parameters**  
   In `semantic_chunk_text`, modify `max_chunk_size` and `similarity_threshold` to control how text chunks are grouped based on semantic similarity. Increasing `similarity_threshold` makes chunk grouping more stringent.

3. **Add Custom Metadata to Chunks**  
   In `generate_chunk_embeddings`, you can add additional metadata to each chunk to provide context or track attributes such as `source_file`, `page_number`, or any other relevant identifiers. Customize the `combined_metadata` dictionary to include these attributes, making it easier to trace answers back to their original context.

4. **Customize Generative Model Prompts (`query.py`)**  
   In `get_answer_from_gemini`, modify the `prompt` variable to tailor the response style of the generative model. You can add context-specific guidelines or constraints to fine-tune the output.

5. **Adjust Number of Relevant Chunks**  
   In `retrieve_relevant_chunks`, change `top_k` to retrieve more or fewer chunks as context for the generative model. Larger values of `top_k` can improve response quality by providing more context but may increase processing time.

---

## Example Usage

- **Upload Files**: Use `upload_embeddings.py` to upload documents.
- **Run Query**: Use `query.py` to input a question and get an answer based on document content.
- **Customize**: Modify parameters, add metadata, and adjust prompt settings as needed for different applications.

---

## Dependencies

- `sentence-transformers`
- `transformers`
- `faiss`
- `pymupdf`
- `scikit-learn`
- `tkinter` (pre-installed with Python)
- `google-generativeai` (for Gemini API)



 
