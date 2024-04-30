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


from cuvs.neighbors import cagra


from pylibraft.common import DeviceResources
from pylibraft.neighbors import cagra as pylibraft_cagra
from pylibraft.common import device_ndarray
from pylibraft.test.ann_utils import calc_recall, generate_data

import pytest


import os
import time
import shutil

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


def loading():
    collection = load_collection("dpr_wiki_collection")
    index = load_index("dpr-nq-multi-hnsw")
    retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)

    #path_to_llama = "/home/dpetresc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
    #model = load_model("meta-llama/Llama-2-7b-chat-hf", weights_path=path_to_llama)

    model = load_model("flan-t5-xxl")
    prompt_template = load_template("qa")

    return collection, index, retriever, model, prompt_template

def running(loading_cached):
    #dataset = generate_data((10000, 10), np.float32)
    #dataset_device = device_ndarray(dataset)
    #build_params = pylibraft_cagra.IndexParams(
    #    metric="sqeuclidean",
    #    intermediate_graph_degree=128,
    #    graph_degree=64,
    #    build_algo="ivf_pq",
    #)
    #index = pylibraft_cagra.build(build_params, dataset_device)

    #dataset_1 = dataset[: 10000 // 2, :]
    #dataset_2 = dataset[10000 // 2 :, :]
    #indices_1 = np.arange(10000 // 2, dtype=np.uint32)
    #indices_2 = np.arange(10000 // 2, 10000, dtype=np.uint32)

    #dataset_1_device = device_ndarray(dataset_1)
    #dataset_2_device = device_ndarray(dataset_2)
    #indices_1_device = device_ndarray(indices_1)
    #indices_2_device = device_ndarray(indices_2)
    #index = pylibraft_cagra.extend(index, dataset_1_device, indices_1_device)
    #index = pylibraft_cagra.extend(index, dataset_2_device, indices_2_device)


    collection, index, retriever, model, prompt_template = loading_cached

    vectors = index.get_embeddings(0, int(len(collection.passages)/200))

    # 21015324 => 64,643,137,024 bytes
    # 700510 => 8 minutes
    # 350255 => 3.45 minutes
    # 105076 => 1 minute, 27266MiB (pas bcp plus pour build l'index)

    #print(help(cagra))
    #print(help(pylibraft_cagra))

    print("Nb vectors: ", len(vectors))

    vectors_gpu = cp.asarray(vectors)

    print(f"GPU Utilization before training: {_get_gpu_stats(0)[0][1]}")

    resources = DeviceResources()

    build_params = cagra.IndexParams(
        metric="euclidean",
        #intermediate_graph_degree=intermediate_graph_degree,
        #graph_degree=graph_degree,
        #build_algo=build_algo,
    )
    
    start = time.time()

    index = cagra.build_index(build_params, vectors_gpu)
    resources.sync()
    end = time.time()
    print("Seconds: ", end - start)

    print(f"GPU Utilization after training: {_get_gpu_stats(0)[0][1]}")

    #index_filepath = os.path.join("data/nq/index/cagra", "cagra.bin")
    #cagra.save(index_filepath, index) 
    #loaded_index = cagra.load(index_filepath)
    resources.sync()


    """ queries = ["what is haleys comet"]

    runner = ResponseRunner(
        model=model,
        retriever=retriever,
        document_collection=collection,
        prompt_template=prompt_template,
        queries=queries,
    )

    responses = runner()
    print(responses[0]["response"]) """
    """
    Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley..."""