import subprocess
import logging
import os
import psutil
# CProfile and pstats are used to profile the code, 
# e.g. to measure the time it takes to execute a function 
import cProfile
import pstats
from memory_profiler import profile as mem_profile
from functools import wraps
from torch.profiler import profile, record_function, ProfilerActivity


from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_resource_usage(func):
    """Decorator to log CPU, RAM, and Disk usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        py = psutil.Process(pid)

        # Log CPU usage before execution
        cpu_before = py.cpu_percent()
        logging.info(f"CPU usage before: {cpu_before}%")

        # Log RAM usage before execution
        ram_before = psutil.virtual_memory().used / (1024 ** 3)
        logging.info(f"RAM usage before: {ram_before:.2f} GB")

        result = func(*args, **kwargs)

        # Log CPU usage after execution
        cpu_after = py.cpu_percent()
        logging.info(f"CPU usage after: {cpu_after}%")

        # Log RAM usage after execution
        ram_after = psutil.virtual_memory().used / (1024 ** 3)
        logging.info(f"RAM usage after: {ram_after:.2f} GB")

        # Disk usage (this can be modified based on specific requirements)
        disk_usage = psutil.disk_usage('/')
        logging.info(f"Disk usage: {disk_usage.used / (1024 ** 3):.2f} GB used of {disk_usage.total / (1024 ** 3):.2f} GB")

        return result

    return wrapper

def get_gpu_memory():
    """Returns the current GPU memory usage using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    memory_used = result.stdout.strip()
    return memory_used if memory_used else "Query failed"

def log_memory_usage(operation_name):
    """Logs the GPU memory usage with the given operation name."""
    memory_used = get_gpu_memory()
    logging.info(f"{operation_name}: GPU Memory Used: {memory_used} MB")

@log_resource_usage
@mem_profile
def complex_operation():
    collection = load_collection("dpr_wiki_collection")
    log_memory_usage("load_collection")
    index = load_index("dpr-nq-multi-hnsw")
    log_memory_usage("load_index")
    retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
    log_memory_usage("load_retriever")
    #model = load_model("flan-t5-xxl")
    path_to_llama = "/home/dpetresc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
    model = load_model("meta-llama/Llama-2-7b-chat-hf", weights_path=path_to_llama)
    log_memory_usage("load_model")
    prompt_template = load_template("qa")
    log_memory_usage("load_template")

    queries = ["what is haleys comet"]

    runner = ResponseRunner(
        model=model,
        retriever=retriever,
        document_collection=collection,
        prompt_template=prompt_template,
        queries=queries,
    )

    responses = runner()
    log_memory_usage("runner")
    print(responses[0]["response"])
    """
    Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley...
    """

def main():
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Execute the function you want to profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True,
             use_cuda=True) as prof:
        complex_operation()

    try:
        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=True,
                           profile_memory=True,
                           use_cuda=True)
        profiler.start()
        complex_operation()
    finally:
        if profiler:
            # test
            # Stop profiling
            profiler.disable()
            stats = pstats.Stats(profiler).sort_stats('cumulative')
            stats.print_stats()

            # Optionally, save stats to file
            stats.dump_stats('program.prof')

            print(prof.key_averages())

if __name__ == "__main__":
    main()