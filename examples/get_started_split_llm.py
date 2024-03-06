from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner_llm import ResponseRunner

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


#model = load_model("flan-t5-xxl")
path_to_llama = "/home/dpetresc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
model = load_model("meta-llama/Llama-2-7b-chat-hf", weights_path=path_to_llama)

start_runner = time.time()
runner = ResponseRunner(
    model=model,
)

responses = runner()
#print(responses[0]["response"])
print(responses)
"""
Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley...
"""


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
