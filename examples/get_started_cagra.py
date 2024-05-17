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

cp.get_default_memory_pool().free_all_blocks()
gc.collect()

#collection = load_collection("dpr_wiki_collection")
#index = load_index("dpr-nq-multi-hnsw")

#batch_size = 1000 # Adjust batch size based on your system's memory capacity
#total_vectors_count = 21015324
#total_vectors_count = 100
#index_filepath = os.path.join("data/nq/index/cagra", "vectors.pickle")
#dump_vectors_in_batches(index, batch_size, total_vectors_count, index_filepath)
#vectors = load_vectors_in_batches(index_filepath)

#vectors = index.get_embeddings(0, total_vectors_count)
#print("Len collection ", 21015324)
#vectors = index.get_embeddings(0, 21015324)



#index = load_index("dpr-nq-multi-hnsw")
#total_vectors_count = 21015324
#vectors = index.get_embeddings(0, total_vectors_count)
#index_filepath = os.path.join("data/nq/index/cagra", "vectors.pickle")
#pickle.dump(vectors, open(index_filepath, 'wb'))
#print("DUMPED")
#vectors = pickle.load(open(index_filepath, 'rb'))
#print("LOADED VECTORS")

#print("Nb vectors: ", len(vectors))
#vectors_gpu = cp.asarray(vectors)

#print(f"GPU Utilization before training: {_get_gpu_stats(0)[0][1]}")

#resources = DeviceResources()

#build_params = pylibraft_cagra.IndexParams(
#    metric="euclidean",
#    #intermediate_graph_degree=intermediate_graph_degree,
#    #graph_degree=graph_degree,
#    #build_algo=build_algo,
#)
    
#start = time.time()

#index = pylibraft_cagra.build_index(build_params, vectors_gpu)
#index = pylibraft_cagra.build(build_params, vectors_gpu, handle=resources)
#resources.sync()
#end = time.time()
#print("Seconds: ", end - start)

#print(f"GPU Utilization after training: {_get_gpu_stats(0)[0][1]}")



# Load vectors from pickle file
nb_vectors_build = 10000
index_filepath = os.path.join("data/nq/index/cagra", "vectors.pickle")
vectors = pickle.load(open(index_filepath, 'rb'))
#logging.info("LOADED VECTORS")

#logging.info("Nb vectors: %d", len(vectors))
vectors_gpu = cp.asarray(vectors[:nb_vectors_build])

index_filepath = os.path.join("data/nq/index/cagra", "cagra_"+str(nb_vectors_build)+".bin")
#pylibraft_cagra.save(index_filepath, index) 
loaded_index = pylibraft_cagra.load(index_filepath)
#resources.sync()

print(help(pylibraft_cagra.search))
distances, neighbors = pylibraft_cagra.search(pylibraft_cagra.SearchParams(),
                                 loaded_index, vectors_gpu,
                                 1)
                                 #1, handle=resources)
#resources.sync()
distances = cp.asarray(distances)
neighbors = cp.asarray(neighbors)
print(distances, neighbors)
