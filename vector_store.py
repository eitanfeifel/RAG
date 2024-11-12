import numpy as np 
import pickle 
import faiss
import os 

class VectorDatabase:
    def __init__( self, embedding_dim, index_file = "index.faiss", metadata_file = "metadata.pkl" ):
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.metadata_file = metadata_file  
        self.embeddings = []
        self.metadata = []

        if os.path.exists( index_file ) and os.path.exists( metadata_file ):
            self.load()
        else:
            self.index = faiss.IndexFlatL2( embedding_dim )

    def add_embeddings( self, embeddings, metadata ):
        self.index.add( embeddings )
        self.embeddings.extend( embeddings )
        self.metadata.extend( metadata )
        self.save()
    
    def search( self, query_embedding, k = 10 ):
        distances, indices = self.index.search( query_embedding, k )
        results = [ self.metadata[ i ] for i in indices[ 0 ] ]
        return results, distances[ 0 ]
    
    def save( self ):
        faiss.write_index( self.index, self.index_file )
        with open( self.metadata_file, "wb" ) as f:
            pickle.dump(self.metadata, f )

    def load( self ):
        self.index = faiss.read_index( self.index_file )
        with open( self.metadata_file, "rb" ) as f:
            self.metadata = pickle.load( f )
    