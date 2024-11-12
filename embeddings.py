from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import numpy as np
import torch

#Initialize Language Model for Generating Embeddings
tokenizer = BertTokenizer.from_pretrained( 'bert-base-uncased' )    
sbert_model = SentenceTransformer( 'paraphrase-MiniLM-L6-v2' )   ##Use sbert_model to generate sentence_embeddings 
#bert_model = BertModel.from_pretrained( 'bert-base-uncased' )   ##Use bert_model if you want to generate individual word embeddings


##Function to generate embeddings at the word-level. Can be used individually, or in combination with sentence embeddings.
# def generate_word_embeddings( text ):                 
#     inputs = tokenizer( text, return_tensors = 'pt', padding = True, truncation = True, add_special_tokens = True )
#     with torch.no_grad():
#         outputs = bert_model( **inputs )
#     word_embeddings = outputs.last_hidden_state.mean( dim = 1 ).numpy()
#     return word_embeddings 

##Function to generate embeddings at the sentence-level.
def generate_sentence_embeddings( text ):
    sentence_embedding = sbert_model.encode( text )
    return sentence_embedding 

##Function to combine both word and sentence embeddings, if you choose to use both
# def combine_embeddings( word_embedding, sentence_embedding ):
#     combined = np.concatenate( [ word_embedding, sentence_embedding ], axis = None )
#     return combined


if __name__ == "__main__":
    text = input( "Input: " )
    sentence_embedding = generate_sentence_embeddings( text ) #Generate Sentence Embedding
    # word_embedding = generate_word_embeddings( text )  #Generate Word Embeddings
    # combined_embedding = combine_embeddings( word_embedding, sentence_embedding ) #Combine word + sentence embeddings

    print( "Sentence Embedded: ", sentence_embedding )
    # print( "Word Embedded: ", word_embedding )
    # print( "Combined: ", combined_embedding )