import torch

from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

document_collection_mapping = {
        #"dpr_wiki_collection": DPRWikiCollection,
        "topiocqa_wiki_collection": None,
        "hotpot_wiki_collection": None,
        "faithdial_collection": None,
}

#for collection in document_collection_mapping:
#    print(collection)
#    collection = load_collection(collection)
index = load_index("dpr-topiocqa-single-hnsw")
