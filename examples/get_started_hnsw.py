import torch
from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner
from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, DPRReader, DPRReaderTokenizer
import faiss
import cupy as cp
import numpy as np

import subprocess
import re


from cuvs.neighbors import cagra as cuvs_cagra


from pylibraft.common import DeviceResources
from pylibraft.neighbors import cagra as pylibraft_cagra
from pylibraft.common import device_ndarray
from pylibraft.test.ann_utils import calc_recall, generate_data


import os
import time
import shutil

import gc

import sys
import pkgutil
import get_started_finegrain
import importlib
import instruct_qa
import traceback

import pickle

import logging

#nb_vectors_build = 10000000
nb_vectors_build = 21015324#10000000
# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join("data/nq/index/hnsw", "output_hnsw_build_"+str(nb_vectors_build)+".log")), logging.StreamHandler()])

def _get_gpu_stats(gpu_id):
    """Run nvidia-smi to get the gpu stats without continuous monitoring."""
    gpu_query = ",".join(["utilization.gpu", "memory.used", "memory.total"])
    result = subprocess.run(
        [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f"Error running nvidia-smi: {result.stderr}")
        return []

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.

    stats = result.stdout.strip().split(os.linesep)
    stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
    return stats

def dump_vectors_in_batches(index, batch_size, total_size, file_path):
    start = 0
    with open(file_path, 'wb') as f:
        while start < total_size:
            end = min(start + batch_size, total_size)
            vectors = index.get_embeddings(start, end)
            pickle.dump(vectors, f)
            print(f"Dumped vectors from {start} to {end}")
            start += batch_size

def load_vectors_in_batches(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                batch = pickle.load(f)
                vectors.extend(batch)  # Process or handle batch
                print(f"Loaded batch of size {len(batch)}")
        except EOFError:
            pass
    return vectors



#collection = load_collection("dpr_wiki_collection")
#index = load_index("dpr-nq-multi-hnsw")

#batch_size = 1000 # Adjust batch size based on your system's memory capacity
#total_vectors_count = 21015324

#vectors = index.get_embeddings(0, total_vectors_count)
#print("Len collection ", 21015324)


#index = load_index("dpr-nq-multi-hnsw")
#print(index)
#print(help(index))
#vectors = index.get_embeddings(0, total_vectors_count)

# Load vectors from pickle file
index_filepath = os.path.join("data/nq/index/cagra", "vectors.pickle")
vectors = pickle.load(open(index_filepath, 'rb'))[:nb_vectors_build]
logging.info("LOADED VECTORS")

logging.info("Nb vectors: %d", len(vectors))

index_filepath = os.path.join("data/nq/index/hnsw", "hnsw_"+str(nb_vectors_build)+".bin")
start = time.time()

# TODO
# Create the HNSW index
d = vectors.shape[1]  # dimension of the vectors
index = faiss.IndexHNSWFlat(d, 32)  # 32 is the number of neighbors in the HNSW graph
index.hnsw.efConstruction = 200  # Controls accuracy and construction time
index.hnsw.efSearch = 128
index.hnsw.max_level = 2

# Add vectors to the index
index.add(vectors)


#print(index)
#print(help(index))
#print(index.is_trained)
#print(index.metric_arg)
#print(index.metric_type)
#print(index.ntotal)
#
#print()
#print(index.hnsw.assign_probas)
#print(index.hnsw.check_relative_distance)
#print(index.hnsw.cum_nneighbor_per_level)
#print(index.hnsw.efConstruction)
#print(index.hnsw.efSearch)
#print(index.hnsw.entry_point)
#print(index.hnsw.levels)
#print(index.hnsw.max_level)
#print(index.hnsw.neighbors)
#print(index.hnsw.offsets)
#print(index.hnsw.rng)
#print(index.hnsw.search_bounded_queue)
#print(index.hnsw.upper_beam)
#print(help(index.hnsw))

end = time.time()
logging.info("Seconds: %f", end - start)

faiss.write_index(index, index_filepath)

#loaded_index = faiss.read_index(index_filepath)

#k = 1  # Number of nearest neighbors to search for
#distances, neighbors = loaded_index.search(vectors, k)
#print(distances)
#print(neighbors)
