# query.py

from embeddings import generate_sentence_embeddings
from vector_store import VectorDatabase
import google.generativeai as genai
import numpy as np
import os

# Configure the Gemini API with the API key from the environment variable
# Be sure to upload your api_key to your os
genai.configure( api_key=os.environ[ "API_KEY" ] )

#Function to generate answers to user's query 
def get_answer_from_gemini( context, question ):
    """Generate an answer from the Gemini API."""
    model = genai.GenerativeModel( "gemini-1.5-flash" )

    ## ADJUST PROMPT BASED ON SPECIFIC PREFERENCES
    prompt = (
        f"You are tasked with answering a user's query based on relevant information "
        f"based on the retrieved information, adress the user's query and provide an informative answer "
        f"'Do not share the context directly, but instead provide an informed and personalized answer.\n\n "
        f"Context: { context }\n"
        f"Question: { question }\n"
    )
    response = model.generate_content( prompt )
    return response.text


#Function to retrieve context relevant to user's query, ADJUST DEFAULT K AS NEEDED
def retrieve_relevant_chunks( vector_db, query_embedding, top_k=5 ):
    """Retrieve the top-k most relevant chunks from the vector store."""
    results, distances = vector_db.search( query_embedding, k=top_k )
    relevant_chunks = []

    for idx, chunk_metadata in enumerate( results ):
        chunk_text = chunk_metadata.get( "text", "No content available" )
        ##Retrieve any other relevant metadata here

        # # Print each retrieved chunk for developer's reference *DEBUGGING/SANITY-CHECK*
        # print(f"Retrieved chunk { idx }: speaker={ speaker }, text={ chunk_text }" )

        #Format any added metadata to chunk before appending
        relevant_chunks.append( chunk_text )

    return "\n".join( relevant_chunks )

def main():
    # Initialize vector database
    vector_db = VectorDatabase( embedding_dim = 384 )

    # Prompt the user for a question
    question = input( "Please enter your question:" )

    # Generate an embedding for the user's question
    query_embedding = generate_sentence_embeddings( question ).reshape( 1, -1 )

    # Retrieve relevant chunks from the vector database
    relevant_context = retrieve_relevant_chunks( vector_db, query_embedding, top_k=10 ) #Adjust k based on preference

    # print("\nRelevant Context Retrieved (for developer reference):")
    # print(relevant_context)  # Display context for developer to verify relevance *Debugging/Sanity-Check*

    # Get answer from Gemini API
    try:
        answer = get_answer_from_gemini( relevant_context, question )
        print( "\nAnswer:" )
        print( answer )
    except Exception as e:
        print( "Failed to retrieve an answer from the Gemini API:", e )

if __name__ == "__main__":
    main()
