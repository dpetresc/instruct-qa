import threading
import subprocess
import os
import shutil
import time
import gc
import pickle
import cupy as cp
from pylibraft.common import DeviceResources
from pylibraft.neighbors import cagra as pylibraft_cagra
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join("data/nq/index/cagra", "output_cagra_build.log")), logging.StreamHandler()])

class GpuMemoryMonitor(threading.Thread):
    def __init__(self, gpu_id=0):
        super().__init__()
        self.gpu_id = gpu_id
        self.max_memory_used = 0
        self.running = True

    def run(self):
        while self.running:
            stats = _get_gpu_stats(self.gpu_id)
            if stats:
                memory_used = stats[0][1]
                if memory_used > self.max_memory_used:
                    self.max_memory_used = memory_used
            time.sleep(1)

    def stop(self):
        self.running = False

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
        logging.error(f"Error running nvidia-smi: {result.stderr}")
        return []

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.

    stats = result.stdout.strip().split(os.linesep)
    stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
    return stats

def main():
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    # Load vectors from pickle file
    index_filepath = os.path.join("data/nq/index/cagra", "vectors.pickle")
    vectors = pickle.load(open(index_filepath, 'rb'))
    logging.info("LOADED VECTORS")

    logging.info("Nb vectors: %d", len(vectors))
    vectors_gpu = cp.asarray(vectors[:10000000])

    logging.info("GPU Utilization before training: %f MiB", _get_gpu_stats(0)[0][1])

    resources = DeviceResources()

    build_params = pylibraft_cagra.IndexParams(
        metric="euclidean",
    )
    
    start = time.time()

    index = pylibraft_cagra.build(build_params, vectors_gpu, handle=resources)
    resources.sync()
    end = time.time()
    logging.info("Seconds: %f", end - start)

    logging.info("GPU Utilization after training: %f MiB", _get_gpu_stats(0)[0][1])

    index_filepath = os.path.join("data/nq/index/cagra", "cagra.bin")
    pylibraft_cagra.save(index_filepath, index)
    resources.sync()

if __name__ == "__main__":
    monitor = GpuMemoryMonitor()
    monitor.start()

    try:
        main()
    finally:
        monitor.stop()
        monitor.join()

    logging.info("Maximum GPU Memory Used: %f MiB", monitor.max_memory_used)

