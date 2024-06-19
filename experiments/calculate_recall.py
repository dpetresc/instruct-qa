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

def knn_threshold(data, count, epsilon=1e-3):
    return data[count - 1] + epsilon

def epsilon_threshold(data, count, epsilon=1e-3):
    return data[count - 1] * (1 + epsilon)

def get_recall_values(dataset_distances, run_distances, count, threshold, epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = threshold(dataset_distances[i], count, epsilon)
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual / float(count)  # Normalize recall
    return (np.mean(recalls), np.std(recalls), recalls)

def knn_recall(dataset_distances, run_distances, count, epsilon=1e-3):
    mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, knn_threshold, epsilon)
    return {'mean': mean, 'std': std, 'recalls': recalls}

def epsilon_recall(dataset_distances, run_distances, count, epsilon=0.01):
    mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, epsilon_threshold, epsilon)
    return {'mean': mean, 'std': std, 'recalls': recalls}

def calculate_recall_with_index_matching(ground_truth_indices, retrieved_indices, k):
    recalls = np.zeros(len(retrieved_indices))
    for i in range(len(retrieved_indices)):
        ground_truth_set = set(ground_truth_indices[i][:k])
        retrieved_set = set(retrieved_indices[i][:k])
        recalls[i] = len(ground_truth_set.intersection(retrieved_set)) / float(k)  # Normalize recall
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    return mean_recall, std_recall, recalls

def calculate_and_print_recalls(ground_truth_indices, ground_truth_distances, retrieved_indices, retrieved_distances, k, epsilon=1e-3):
    logger = logging.getLogger()
    
    # Calculate recall based on indices
    mean_recall_indices, std_recall_indices, recalls_indices = calculate_recall_with_index_matching(ground_truth_indices, retrieved_indices, k)

    # Calculate recall based on distances using k-NN threshold
    mean_recall_knn, std_recall_knn, recalls_knn = get_recall_values(ground_truth_distances, retrieved_distances, count=k, threshold=knn_threshold, epsilon=epsilon)

    # Calculate recall based on distances using epsilon threshold
    mean_recall_epsilon, std_recall_epsilon, recalls_epsilon = get_recall_values(ground_truth_distances, retrieved_distances, count=k, threshold=epsilon_threshold, epsilon=epsilon)

    # Log all recalls
    logger.info(f'Index-Based Mean Recall: {mean_recall_indices}')
    logger.info(f'Index-Based Std Recall: {std_recall_indices}')
    logger.info(f'Index-Based Recalls: {recalls_indices}')

    logger.info(f'k-NN Mean Recall: {mean_recall_knn}')
    logger.info(f'k-NN Std Recall: {std_recall_knn}')
    logger.info(f'k-NN Recalls: {recalls_knn}')

    logger.info(f'Epsilon Mean Recall: {mean_recall_epsilon}')
    logger.info(f'Epsilon Std Recall: {std_recall_epsilon}')
    logger.info(f'Epsilon Recalls: {recalls_epsilon}')

    # Log the maximum recall for each query
    combined_recalls = np.maximum.reduce([recalls_indices, recalls_knn, recalls_epsilon])
    logger.info(f'Maximum Recalls: {combined_recalls}')
    logger.info(f'Mean of Maximum Recalls: {np.mean(combined_recalls)}')
    logger.info(f'Std of Maximum Recalls: {np.std(combined_recalls)}')

    # Log a single value for all queries
    overall_mean_recall = np.mean(combined_recalls)
    overall_std_recall = np.std(combined_recalls)
    logger.info(f'Overall Mean Recall: {overall_mean_recall}')
    logger.info(f'Overall Std Recall: {overall_std_recall}')

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
    #output_dir = os.path.join(args.persistent_dir, "results", args.dataset_name, "response")
    output_dir = os.path.join(args.persistent_dir, "response_brute_force")
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
        file_path=args.dataset_file_path
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

    index = None
    if args.index_name is not None:
        logger.info("Loading index...")
        kwargs = {}
        if args.index_path is not None:
            kwargs['index_path'] = os.path.join(args.persistent_dir, args.index_path)
        else:
            kwargs['index_path'] = os.path.join(args.persistent_dir, 'nq/index/hnsw/index.dpr')
        start = time.time()
        index = load_index(args.index_name, **kwargs)
        end = time.time()
        logging.info("Loading index time: %f", end - start)
        logging.info("Index size: %f", index.__len__())


    retriever = None
    logger.info("Loading retriever...")
    retriever = load_retriever(
        args.retriever_name,
        index,
        retriever_cached_results_fp=args.retriever_cached_results_fp,
    )

    prompt_template = load_template(args.prompt_type)

    os.makedirs(
        f"{args.persistent_dir}/response_brute_force", exist_ok=True
    )

    #runner()
    r_dict = retriever.retrieve(dataset.get_queries(dataset), 10)
    retrieved_indices_list = r_dict["indices"]
    retrieved_distances_list = np.sqrt(r_dict["scores"])

    # Load the ground truth CSV file
    ground_truth_df = pd.read_csv(os.path.join(f"{args.persistent_dir}/response_brute_force", 'nearest_neighbors.csv'))

    ground_truth_df['indices'] = ground_truth_df['indices'].apply(lambda x: eval(x))  # Convert string representation of lists back to lists
    ground_truth_df['distances'] = ground_truth_df['distances'].apply(lambda x: eval(x))

    # Prepare dataset distances
    ground_truth_indices_list = ground_truth_df['indices'].tolist()
    ground_truth_distances_list = ground_truth_df['distances'].tolist()

    # Calculate k-NN recall
    #print("dataset_distances ", ground_truth_distances_list[:2])
    #print("retrieved_distances_list ", retrieved_distances_list)
    calculate_and_print_recalls(
        ground_truth_indices_list, ground_truth_distances_list,
        retrieved_indices_list, retrieved_distances_list, k=10
    )

    end = time.time()
    logging.info("Total execution time: %f", end - start)
