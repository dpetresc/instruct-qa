from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.response_runner_retrieval import ResponseRunner

import time
import resource
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device=None)  # device=None resets stats for all GPUs
else:
    print("HERE")
start_time = time.time()

collection = load_collection("dpr_wiki_collection")
index = load_index("dpr-nq-multi-hnsw")
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
prompt_template = load_template("qa")

queries = ["what is haleys comet"]#*100

start_runner = time.time()
runner = ResponseRunner(
    retriever=retriever,
    document_collection=collection,
    prompt_template=prompt_template,
    queries=queries,
    batch_size=10,
)
prompts = runner()
print(prompts)

end_time = time.time()

print(f"Runtime runner is {end_time - start_runner} seconds")

total_time = end_time - start_time
print(f"Total runtime of the program is {total_time} seconds")

peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"Peak memory usage: {peak_memory_usage/(1024. ** 2)} GB")

if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated()  # Returns max memory allocated in bytes
    max_memory_mb = max_memory / (1024 ** 2)
    print(f"Maximum GPU memory usage: {max_memory_mb} MB")