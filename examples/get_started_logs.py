import subprocess
import logging
import os
import psutil
# CProfile and pstats are used to profile the code, 
# e.g. to measure the time it takes to execute a function 
# frequency of function calls
import cProfile
import pstats
# An external library for profiling memory usage of Python programs line-by-line. 
# It's used here with a decorator @mem_profile to measure memory usage of the complex_operation function.
from memory_profiler import profile as mem_profile
from functools import wraps
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import argparse


from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Called once at the beginning and once at the end
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
        # measures the average CPU utilization since the last call to cpu_percent()
        # py.cpu_percent() system-wide CPU utilization
        cpu_after = py.cpu_percent()
        logging.info(f"CPU usage after: {cpu_after}%")

        # Log RAM usage after execution
        # Snapshot of the total used memory by all processes on the system at that specific moment.
        ram_after = psutil.virtual_memory().used / (1024 ** 3)
        logging.info(f"RAM usage after: {ram_after:.2f} GB")

        # Disk usage 
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

def log_cpu_usage(operation_name, py):
    # Log CPU usage before execution
    cpu_ = py.cpu_percent()
    logging.info(f"{operation_name}: CPU usage: {cpu_}%")

@log_resource_usage
@mem_profile
def complex_operation():
    pid = os.getpid()
    py = psutil.Process(pid)

    collection = load_collection("dpr_wiki_collection")
    log_cpu_usage("load_collection", py)
    log_memory_usage("load_collection")
    index = load_index("dpr-nq-multi-hnsw")
    log_cpu_usage("load_index", py)
    log_memory_usage("load_index")
    retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
    log_cpu_usage("load_retriever", py)
    log_memory_usage("load_retriever")
    #model = load_model("flan-t5-xxl")
    path_to_llama = "/home/dpetresc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
    model = load_model("meta-llama/Llama-2-7b-chat-hf", weights_path=path_to_llama)
    log_cpu_usage("load_model", py)
    log_memory_usage("load_model")
    prompt_template = load_template("qa")
    log_cpu_usage("load_template", py)
    log_memory_usage("load_model")


    queries = ["what is haleys comet"]

    runner = ResponseRunner(
        model=model,
        retriever=retriever,
        document_collection=collection,
        prompt_template=prompt_template,
        queries=queries,
    )

    responses = runner()
    log_cpu_usage("runner", py)
    log_memory_usage("runner")
    print(responses[0]["response"])
    """
    Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley...
    """

def main():
    parser = argparse.ArgumentParser(description="Run profiling for complex_operation function.")
    parser.add_argument('--use_profiler', choices=['cProfile', 'pytorch', 'none'], default='none',
                        help='Specify which profiler to use: "cProfile" or "pytorch", or none for no profiling.')

    args = parser.parse_args()

    if args.use_profiler == 'cProfile':
        # Start cProfile profiling
        # number of calls to each method and amount of time spent in them
        cprofiler = cProfile.Profile()
        cprofiler.enable()

        # Execute the function you want to cProfile
        complex_operation()

        # Stop cProfile profiling
        cprofiler.disable()
        cstats = pstats.Stats(cprofiler).sort_stats('cumulative')
        # visualize with pip install snakeviz or pip install gprof2dot
        cstats.dump_stats('logs/cprofile.prof')
        print("CSTATS NOW:     ")
        cstats.print_stats(100)

    elif args.use_profiler == 'pytorch':
        # Define the directory for TensorBoard logs
        #log_dir = "./logs"
        #writer = SummaryWriter(log_dir)
        # Execute the function with PyTorch profiler
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True) as prof:
            complex_operation()
        prof.export_chrome_trace("logs/pytorch_profiling_trace.json")
        print("PYORCH NOW:   ")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # After your profiling block
        table_str = prof.key_averages().table(sort_by="cpu_time_total")
        with open("logs/profiling_summary.txt", "w") as f:
            f.write(table_str)
        # Access detailed data
        for event in prof.key_averages():
            print(f"Name: {event.key}")
            print(f"CPU time: {event.cpu_time_total} ns")
            print(f"CUDA time: {event.cuda_time_total} ns")
            print(f"CPU memory usage: {event.cpu_memory_usage} bytes")
            print(f"CUDA memory usage: {event.cuda_memory_usage} bytes")
            print(f"Number of calls: {event.count}")
            print("------")
            #writer.add_scalar(f"Profiling/CPU time (ns) - {event.key}", event.cpu_time_total, global_step=0)
            #writer.add_scalar(f"Profiling/CUDA time (ns) - {event.key}", event.cuda_time_total, global_step=0)
            #writer.add_scalar(f"Profiling/CPU memory (bytes) - {event.key}", event.cpu_memory_usage, global_step=0)
            #writer.add_scalar(f"Profiling/CUDA memory (bytes) - {event.key}", event.cuda_memory_usage, global_step=0)
    else:
        print("Executing without profiling...")
        complex_operation()

if __name__ == "__main__":
    main()