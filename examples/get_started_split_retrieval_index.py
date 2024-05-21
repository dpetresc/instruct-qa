from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
#from instruct_qa.response_runner_retrieval import ResponseRunner

import time

def loading():
    #collection = load_collection("dpr_wiki_collection")
    print("Loading index")
    index = load_index("dpr-nq-multi-hnsw")
    #retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
    #prompt_template = load_template("qa")

    #return collection, index, retriever, model, prompt_template

    return index

def running(loading_cached):
    index = loading_cached
    print(index)
    #print(help(index.index))
    print(index.index.is_trained)
    print(index.index.metric_arg)
    print(index.index.metric_type)
    print(index.index.ntotal)
    
    print()
    print(index.index.hnsw.assign_probas)
    print(index.index.hnsw.check_relative_distance)
    print(index.index.hnsw.cum_nneighbor_per_level)
    print(index.index.hnsw.efConstruction)
    print(index.index.hnsw.efSearch)
    print(index.index.hnsw.entry_point)
    print(index.index.hnsw.levels)
    print(index.index.hnsw.max_level)
    print(index.index.hnsw.neighbors)
    print(index.index.hnsw.offsets)
    print(index.index.hnsw.rng)
    print(index.index.hnsw.search_bounded_queue)
    print(index.index.hnsw.upper_beam)
    print(help(index.index.hnsw))

#    collection, index, retriever, model, prompt_template = loading_cached

#    queries = ["what is haleys comet"]*100
#
#    start_runner = time.time()
#    runner = ResponseRunner(
#        retriever=retriever,
#        document_collection=collection,
#        prompt_template=prompt_template,
#        queries=queries,
#        batch_size=10,
#    )
#    prompts = runner()
#    print(prompts)
#
#    end_time = time.time()
#
#    print(f"Runtime runner is {end_time - start_runner} seconds")
