import argparse
import logging
import logging.config
from math import inf
import os

import torch

import pickle

from instruct_qa.prompt.utils import load_template
from instruct_qa.retrieval import RetrieverFromFile
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.response_runner import ResponseRunner
from instruct_qa.collections.utils import load_collection
from instruct_qa.generation.utils import load_model
from instruct_qa.dataset.utils import load_dataset
from instruct_qa.experiment_utils import log_commandline_args, generate_experiment_id

import time

from sklearn.neighbors import NearestNeighbors
import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Evaluates a model against a QA dataset.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "../examples/data")),
)
parser.add_argument(
    "--model_name",
    action="store",
    type=str,
    default="opt-125m",
    help="The model to evaluate.",
)
parser.add_argument(
    "--weights_path",
    action="store",
    type=str,
    default=None,
    help="The path to the directory for local weights of the models e.g., llama, alpaca, etc.",
)
parser.add_argument(
    "--dataset_name",
    action="store",
    type=str,
    default="hotpot_qa",
    help="The dataset to evaluate against.",
)
parser.add_argument(
    "--prompt_type",
    action="store",
    type=str,
    default="qa",
    choices=["qa", "llama_chat_qa", "conv_qa", "llama_chat_conv_qa", "qa_unanswerable", "llama_chat_qa_unanswerable", "conv_qa_unanswerable", "llama_chat_conv_qa_unanswerable"],
    help="Specify the prompt used to be used by instruction-following models",
)
parser.add_argument(
    "--dataset_config_name",
    action="store",
    type=str,
    default=None,
    help="The specific dataset configuration to use.",
)
parser.add_argument(
    "--dataset_split",
    action="store",
    type=str,
    default="validation",
    help="The split of the dataset to use.",
)
parser.add_argument(
    "--dataset_file_path",
    action="store",
    type=str,
    default=None,
    help="The path to the dataset file.",
)
parser.add_argument(
    "--temperature",
    action="store",
    type=float,
    default=0.95,
    help="The temperature to use during generation.",
)
parser.add_argument(
    "--top_p",
    action="store",
    type=float,
    default=0.95,
    help="The Nucleus Sampling parameter to use during generation.",
)
parser.add_argument(
    "--min_new_tokens",
    action="store",
    type=int,
    default=1,
    help="The minimum number of tokens to generate.",
)
parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=inf,
    help="The maximum number of tokens to generate.",
)
parser.add_argument(
    "--api_key",
    action="store",
    type=str,
    default=None,
    help="API key if generating from OpenAI model.",
)
parser.add_argument(
    "--document_collection_name",
    action="store",
    type=str,
    default="dpr_wiki_collection",
    help="Document collection to retrieve from.",
)
parser.add_argument(
    "--document_cache_dir",
    action="store",
    type=str,
    default=None,
    help="Directory that document collection is cached in.",
)
parser.add_argument(
    "--document_file_name",
    action="store",
    type=str,
    default=None,
    help="Basename of the path to the file containing the document collection.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="Batch size to use for generation.",
)
parser.add_argument(
    "--logging_interval",
    action="store",
    type=int,
    default=10,
    help="Step frequency to write results to disk.",
)
parser.add_argument(
    "--k",
    action="store",
    type=int,
    default=10,
    help="Number of passages to retrieve.",
)
parser.add_argument(
    "--retriever_name",
    action="store",
    type=str,
    default="all-MiniLM-L6-v2",
    help="Name of the retriever to load.",
)
parser.add_argument(
    "--index_name",
    action="store",
    type=str,
    default=None,
    help="Name of the index to use for retrieval.",
)
parser.add_argument(
    "--index_path",
    action="store",
    type=str,
    default=None,
    help="Path to the index to use for retrieval.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)

parser.add_argument(
    "--use_hosted_retriever",
    choices=["true", "false"],
    default="false",
    help="Whether to use the hosted retriever.",
)

parser.add_argument(
    "--hosted_retriever_url",
    action="store",
    type=str,
    default="http://10.140.16.91:42010/search",
    help="URL of the hosted retriever, if use_hosted_retriever is true.",
)

parser.add_argument(
    "--retriever_cached_results_fp",
    action="store",
    type=str,
    default=None,
    help="Path to the file containing cached retriever results.",
)

parser.add_argument(
    "--post_process_response",
    action="store_true",
    default=False,
    help="Whether to post-process the results. ",
)

if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name=args.dataset_name,
        split=args.dataset_split,
        collection_name=args.document_collection_name,
        model_name=args.model_name.replace("/", "_"),
        retriever_name=args.retriever_name,
        prompt_type=args.prompt_type,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed,
        index_name=args.index_name,
    )

    # Define output path for response and logs
    output_dir = os.path.join(args.persistent_dir, "results", args.dataset_name, "response")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment_id}_index-{args.index_name}_bs-{args.batch_size}.jsonl" if args.index_name else f"{experiment_id}.jsonl")
    log_file = os.path.join(output_dir, f"{experiment_id}_index-{args.index_name}_bs-{args.batch_size}.log" if args.index_name else f"{experiment_id}.log")

    # Create a logging configuration dictionary
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "file": {
                "level": "INFO",
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "standard",
            },
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["file", "console"],
            "level": "INFO",
        },
    }

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger()

    logger.info("Evaluating model:")
    log_commandline_args(args, logger.info)
    
    logger.info(f"Experiment ID: {experiment_id}")

#    dataset = load_dataset(
#        args.dataset_name,
#        split=args.dataset_split,
#        name=args.dataset_config_name,
#        file_path=args.dataset_file_path,
#        nb_loaded=2
#    )

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        name=args.dataset_config_name,
        file_path=args.dataset_file_path,
    )
    print("DATASET: ", dataset.get_queries(dataset))
    logger.info(f"Output response file: {output_file}")
    logger.info(f"Length of dataset: {len(dataset)}")

    logger.info("Loading document collection...")
    kwargs = {}
    if args.document_cache_dir is not None:
        kwargs['cachedir'] = args.document_cache_dir
    else:
        kwargs['cachedir'] = os.path.join(args.persistent_dir, 'nq/collection')
    if args.document_file_name is not None:
        kwargs['file_name'] = args.document_file_name
    document_collection = load_collection(args.document_collection_name, **kwargs)


    retriever = None
    logger.info("Loading retriever...")
    retriever = load_retriever(
        args.retriever_name,
        None,
        retriever_cached_results_fp=args.retriever_cached_results_fp,
    )

    prompt_template = load_template(args.prompt_type)

    os.makedirs(
        f"{args.persistent_dir}/response_brute_force", exist_ok=True
    )

    #runner()
    queries = retriever.encode_queries(dataset.get_queries(dataset))
    if not isinstance(queries, np.ndarray):
        queries = np.array(queries)

    aux_dim = np.zeros(len(queries), dtype="float32")
    query_vectors = np.hstack((queries, aux_dim.reshape(-1, 1)))

    #print("QUERY VECTORS: ", query_vectors)

    index_filepath = os.path.join(os.path.join(args.persistent_dir, "nq/index/cagra"), "vectors.pickle")
    #X = pickle.load(open(index_filepath, 'rb'))[:nb_vectors_build]
    X = pickle.load(open(index_filepath, 'rb'))[:]
    #print("X: ", X)
    #print(len(X))

    start = time.time()

    metric = 'euclidean'  # You can use any valid metric here
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric=metric)
    nbrs.fit(X) 

    # Find the 10 nearest neighbors for each query vector
    distances, indices = nbrs.kneighbors(query_vectors)


    # Store results in a DataFrame
    results_df = pd.DataFrame({
        'query_id': np.arange(len(query_vectors)),
        'query_vector': [query_vectors[i].tolist() for i in range(len(query_vectors))],
        'indices': [indices[i].tolist() for i in range(len(query_vectors))],
        'distances': [distances[i].tolist() for i in range(len(query_vectors))]
    })

    print(results_df.head())

    # Save to CSV
    results_df.to_csv(os.path.join(f"{args.persistent_dir}/response_brute_force", 'nearest_neighbors.csv'), index=False)

    end = time.time()
    logging.info("Total execution time: %f", end - start)
