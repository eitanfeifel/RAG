# upload_embeddings.py

from sklearn.metrics.pairwise import cosine_similarity
from embeddings import generate_sentence_embeddings
from vector_store import VectorDatabase
from tkinter import filedialog
from docx import Document
import tkinter as tk
import numpy as np
import fitz  
import re
import os


##Function to extract tect from a .txt file
def read_txt( file_path ):
    with open( file_path, "r", encoding="utf-8" ) as file:
        return file.readlines()
    
##Function to extract tect from a .docx file
def read_docx( file_path ):
    doc = Document( file_path )
    return [ para.text for para in doc.paragraphs ]

##Function to extract tect from a .pdf file
def read_pdf( file_path ):
    text = ""
    pdf_document = fitz.open( file_path )
    for page_num in range( pdf_document.page_count ):
        page = pdf_document[ page_num ]
        text += page.get_text()
    pdf_document.close()
    return text.splitlines()

##Function to extract text from the uploaded file
def get_text_from_file( file_path ):
    ext = os.path.splitext( file_path )[ 1 ].lower()
    if ext == ".txt":
        return read_txt( file_path )
    elif ext == ".docx":
        return read_docx( file_path )
    elif ext == ".pdf":
        return read_pdf( file_path )
    else:
        raise ValueError( f"Unsupported file format: { ext }" )



##MODIFY THIS FUNCTION to do any text pre-processing, clean-up, or pre-chunking.  
def preprocess_text( text ):
    """ This is a generic preprocessing function. Adjust to your preference, all present code is optional"""
    text = " ".join( text ) ##Combine text to single line
    text = re.sub( r"[^a-zA-Z0-9\s]", "", text ) ##Remove unwanted Charecters
    text = re.sub( r"\s+", " ", text ).strip() ##Remove extra whitespace
    
    ##OPTIONAL Preliminary Chunking before semantic-chunking
    chunk_size = 700
    chunks = [{"text": text[i:i + chunk_size]} for i in range(0, len(text), chunk_size)]

    return chunks

##Function to chunk text, semantically
def semantic_chunk_text( chunks, max_chunk_size=250, similarity_threshold=0.8 ): ##Adjust default chunk-size and similarity threshold to preference
    grouped_chunks = []
    current_group = []
    current_group_embedding = None

    for chunk in chunks:
        text = chunk[ "text" ]
        if not text.strip():  # Skip empty chunks
            continue
        
        ##Get embeddings of text
        embedding = generate_sentence_embeddings( text )
        
        ##Group text by semantic similarity
        if current_group and len(" ".join( [ c[ "text" ] for c in current_group ] ).split()) > max_chunk_size:
            grouped_chunks.append( current_group )
            current_group = [ chunk ]
            current_group_embedding = embedding
        else:
            if current_group_embedding is None:
                current_group_embedding = embedding
            else:
                similarity = cosine_similarity( [ current_group_embedding ], [ embedding ] )[ 0 ][ 0 ]
                if similarity < similarity_threshold:
                    grouped_chunks.append( current_group )
                    current_group = [ chunk ]
                    current_group_embedding = embedding
                else:
                    current_group.append( chunk )
                    current_group_embedding = np.mean( [ current_group_embedding, embedding ], axis=0 )
    
    if current_group:
        grouped_chunks.append( current_group )
    
    return grouped_chunks

##Create embeddings for each chunk, store any relevant metadata
def generate_chunk_embeddings( grouped_chunks, source_file ):
    embeddings = []
    metadata = []

    for group in grouped_chunks:
        if not group:  # Skip empty groups
            continue

        combined_text = " ".join( [ chunk[ "text" ] for chunk in group ] )
        # print("Combined text for group:", combined_text)  # Debugging statement
        embedding = generate_sentence_embeddings( combined_text )
        embeddings.append( embedding )
        
        # Combine metadata ADD ANY METADATA ACCORDING TO PREFERENCE. 
        combined_metadata = {
            "source_file": source_file,
            "text": combined_text  # Ensure text is added to metadata
        }
        metadata.append( combined_metadata )
    
    return np.array( embeddings ), metadata



def main():
    vector_db = VectorDatabase( embedding_dim= 384 )

    # # Using a specific file path for testing; adjust as needed
    # file_paths = [ "Path/to/your/file" ]

    #.\venv\Scripts\activate


    # Open file dialog to select multiple files
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select Files",
        filetypes=[ ( "Text files", "*.txt" ), ( "Word files", "*.docx" ), ( "PDF files", "*.pdf" ) ]
    )

    for file_path in file_paths:
        print( f"Processing file: { file_path }" )
        try:
            text = get_text_from_file( file_path )
            print('here1')
            chunks = preprocess_text( text ) 
            print('here2')
            semantic_chunks = semantic_chunk_text( chunks, max_chunk_size=250, similarity_threshold=0.8 ) ##Adjust Chunk-Size/Semantic Similarity to preference
            print('here3')
            embeddings, metadata = generate_chunk_embeddings( semantic_chunks, file_path )
            print('here4')
            vector_db.add_embeddings( embeddings, metadata )
            print('here5')
            # print(f"Chunks with embeddings added to the vector store from {file_path}.")
        except ValueError as e:
            print( e )

if __name__ == "__main__":
    main()
