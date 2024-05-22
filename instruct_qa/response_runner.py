import json
import os
from pathlib import Path
import numpy as np
import requests
import re

import logging

from instruct_qa.retrieval.utils import dict_values_list_to_numpy
from instruct_qa.dataset.qa import GenericQADataset
from tqdm import tqdm

import psutil
import subprocess

import time

import threading

def get_gpu_usage():
    gpu_query = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    result = subprocess.run(gpu_query.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Error running nvidia-smi: {result.stderr.decode()}")
    output = result.stdout.decode().strip().split('\n')
    gpu_usage = [list(map(float, line.split(','))) for line in output]
    return gpu_usage

class ResourceMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cpu_max = 0
        self.ram_max = 0
        self.gpu_max = [0, 0, 0]  # Utilization, Memory Used, Memory Total
        self.running = True

    def run(self):
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=1)
            virtual_memory = psutil.virtual_memory()
            ram_percent = virtual_memory.percent
            gpu_usage = get_gpu_usage()

            self.cpu_max = max(self.cpu_max, cpu_percent)
            self.ram_max = max(self.ram_max, ram_percent)
            self.gpu_max = [max(current, new) for current, new in zip(self.gpu_max, gpu_usage[0])]

            time.sleep(1)

    def stop(self):
        self.running = False
        self.join()

    def reset(self):
        self.cpu_max = 0
        self.ram_max = 0
        self.gpu_max = [0, 0, 0]


class ResponseRunner:
    def __init__(
        self,
        model,
        retriever,
        document_collection,
        prompt_template,
        dataset=None,
        queries=None,
        output_path=None,
        k=10,
        batch_size=1,
        logging_interval=256,
        use_hosted_retriever=False,
        hosted_retriever_url="http://10.140.16.91:42010/search",
        use_cached_retrieved_results=False,
        post_process_response=False,
    ):
        self._model = model
        self._retriever = retriever
        self._document_collection = document_collection
        self._prompt_template = prompt_template

        # either dataset or queries should be specified, but not both
        assert (dataset is None) != (queries is None), "Either dataset or queries should be specified, but not both"
        if queries:
            dataset = GenericQADataset(queries)
        self._dataset = dataset
        self._output_path = output_path
        self._k = k
        self._batch_size = batch_size
        self._logging_interval = logging_interval
        self._use_hosted_retriever = use_hosted_retriever
        self._hosted_retriever_url = hosted_retriever_url
        self._use_cached_retrieved_results = use_cached_retrieved_results
        self._collection_name = document_collection.get_name()
        self._post_process_response = post_process_response

    def post_process_response(self, response):
        return self._model.post_process_response(response)

    def __call__(self):
        if self._output_path and os.path.exists(self._output_path):
            with open(self._output_path, "r") as f:
                existing_results = [json.loads(line) for line in f.readlines()]
            num_done = len(existing_results)
            if num_done >= len(self._dataset):
                print(f"Already done with {num_done} examples.")
                return
            if num_done > 0:
                print(f"Skipping {num_done} examples that are already done.")
                self._dataset.data = self._dataset.data[num_done:]
        batches = [
            self._dataset[i : i + self._batch_size]
            for i in range(0, len(self._dataset), self._batch_size)
        ]

        results = []

        for i, batch in enumerate(
            tqdm(batches, desc="Collecting responses", leave=False)
        ):
            # Start resource monitoring
            monitor = ResourceMonitor()
            monitor.start()

            queries = self._dataset.get_queries(batch)

            if self._use_hosted_retriever:
                post_results = requests.post(
                    url=self._hosted_retriever_url,
                    json={
                        "queries": queries,
                        "k": self._k,
                        "dataset": self._collection_name,
                    },
                )
                r_dict = dict_values_list_to_numpy(post_results.json())
                retrieved_indices = r_dict["indices"]
            elif self._use_cached_retrieved_results:
                retrieved_ctx_ids = self._retriever.retrieve(queries, k=self._k)
                retrieved_indices = [
                    self._document_collection.get_indices_from_ids(x)
                    for x in retrieved_ctx_ids
                ]
            else:
                r_dict = self._retriever.retrieve(queries, k=self._k)
                retrieved_indices = r_dict["indices"]

            monitor.stop()
            logging.info(f"Retrieval CPU max usage: {monitor.cpu_max}%")
            logging.info(f"Retrieval RAM max usage: {monitor.ram_max}%")
            logging.info(f"Retrieval GPU max usage: {monitor.gpu_max}")

            # Reset the monitor for the next phase
            monitor.reset()



            # Get the document texts.
            passages = [
                self._document_collection.get_passages_from_indices(indices)
                for indices in retrieved_indices
            ]

            prompts = [
                self._prompt_template(
                    sample=sample,
                    passages=p,
                )
                for sample, p in zip(batch, passages)
            ]
            
            monitor.start()

            responses = self._model(prompts)

            # Stop resource monitoring
            monitor.stop()
            logging.info(f"Generation CPU max usage: {monitor.cpu_max}%")
            logging.info(f"Generation RAM max usage: {monitor.ram_max}%")
            logging.info(f"Generation GPU max usage: {monitor.gpu_max}")

            if self._post_process_response:
                responses = [self.post_process_response(response) for response in responses]

            results.extend(
                {
                    "id_": example.id_,
                    "question": example.question,
                    "response": response,
                    "answer": example.answer,
                    "prompt": prompt,
                    "indices": indices.tolist()
                    if type(indices) == np.ndarray
                    else indices,
                }
                for example, response, prompt, indices in zip(
                    batch, responses, prompts, retrieved_indices
                )
            )

            if self._output_path and (i + 1) % self._logging_interval == 0:
                self._write_results_to_file(results)
                results = []
        if self._output_path is not None:
            self._write_results_to_file(results)
        
        return results

    def _write_results_to_file(self, results):
        # Use pathlib to create a folder of the output path if it is not created
        # already.
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "a") as f:
            f.writelines(json.dumps(result) + "\n" for result in results)
