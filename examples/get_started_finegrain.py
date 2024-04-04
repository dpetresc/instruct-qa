from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner
from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, DPRReader, DPRReaderTokenizer

import numpy as np

def loading():
    collection = load_collection("dpr_wiki_collection")
    index = load_index("dpr-nq-multi-hnsw")
    retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)

    #path_to_llama = "/home/dpetresc/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"
    #model = load_model("meta-llama/Llama-2-7b-chat-hf", weights_path=path_to_llama)

    model = load_model("flan-t5-xxl")
    prompt_template = load_template("qa")

    return collection, index, retriever, model, prompt_template

def running(loading_cached):
    collection, index, retriever, model, prompt_template = loading_cached

    # "id": sample_id,
    # "text": passage,
    # "title": title,
    # "sub_title": sub_title,
    # "index": index,

    print(len(collection.passages))
    print(collection.passages[0]['text'])

    # 'facebook/dpr-ctx_encoder-single-nq-base' is the DPR context encoder model trained on NQ alone
    # 'facebook/dpr-ctx_encoder-multiset-base' is the DPR context encoder model trained on the multiset/hybrid dataset defined in the paper. It includes Natural Questions, TriviaQA, WebQuestions and CuratedTREC
    query_model = SentenceTransformer("facebook-dpr-question_encoder-multiset-base")
    #query_model = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
    #query_model = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")
    #query_model = SentenceTransformer("facebook-dpr-ctx_encoder-multiset-base")

    # Encode the target passage (ensure it's normalized and preprocessed as needed)
    query_embedding = query_model.encode(collection.passages[0]['text'])

    # Convert the query_embedding to FAISS compatible format (numpy array, float32)
    # 768
    print(len(query_embedding))
    print(query_embedding/np.linalg.norm(query_embedding))
    query_embedding_np = np.array([query_embedding]).astype('float32')
    aux_dim = np.zeros(1, dtype="float32")
    query_embedding_np = np.hstack((query_embedding_np, aux_dim.reshape(-1, 1)))
    print(len(query_embedding_np[0]))
    print(query_embedding_np)

    print("HERE")
    print(index.index.search(query_embedding_np, k=1))
    #print(index.index.search_and_reconstruct(query_embedding_np, k=1))
    # same as index.index.reconstruct_n(start_ix, end_ix)
    embedding_0 = np.array(index.get_embeddings(0,1)).astype('float32')
    print(embedding_0)
    
    # 769
    print(len(embedding_0[0]))
    #print(np.linalg.norm(query_embedding_np-embedding_0))
    # => FAISS uses the squared L2 distance
    squared_distance_manual = np.sum(np.square(query_embedding_np - embedding_0))
    print("Squared L2 distance manually calculated:", squared_distance_manual)

    #print(help(index.index))
    # 1 The metric type 1 specifically refers to the L2 distance, also known as Euclidean distance.
    # print(index.index.metric_type)

    """ queries = ["what is haleys comet"]

    runner = ResponseRunner(
        model=model,
        retriever=retriever,
        document_collection=collection,
        prompt_template=prompt_template,
        queries=queries,
    )

    responses = runner()
    print(responses[0]["response"]) """
    """
    Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley..."""