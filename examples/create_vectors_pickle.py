import pickle
import os
import logging

# Define the directory and filenames for loading vectors
vector_dir = "data/nq/index/cagra"
vector_files = {
    10000: "vectors_10000.pickle",
    1000000: "vectors_1000000.pickle",
    5000000: "vectors_5000000.pickle"
}

def generate_or_load_vectors():
    index_filepath = os.path.join(vector_dir, "vectors.pickle")
    vectors = pickle.load(open(index_filepath, 'rb'))

    return vectors  # A list or array of vectors

def load_vectors(num_vectors):
    if num_vectors not in vector_files:
        raise ValueError(f"Number of vectors {num_vectors} not available. Choose from {list(vector_files.keys())}.")
    
    filepath = os.path.join(vector_dir, vector_files[num_vectors])
    with open(filepath, 'rb') as file:
        vectors = pickle.load(file)
    
    logging.info("LOADED VECTORS")
    logging.info("Nb vectors: %d", len(vectors))
    return vectors

# Load or generate the full set of vectors
vectors = generate_or_load_vectors()

# Save vectors to separate files
for num_vectors, filename in vector_files.items():
    filepath = os.path.join(vector_dir, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(vectors[:num_vectors], file)
    logging.info("Saved %d vectors to %s", num_vectors, filepath)

#num_vectors_to_load = 10000  # Change this to 1000000 or 5000000 as needed
#vectors = load_vectors(num_vectors_to_load)
