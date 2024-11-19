from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
import tqdm
import faulthandler
# from langchain_ollama import ChatOllama
faulthandler.enable()
import os
import faulthandler

faulthandler.enable()
import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import ollama
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from tavily import TavilyClient
hf_logging.set_verbosity_error()
import chardet
from readers.datasets import ReaderDatasetWithChains
from readers.collators import CollatorWithChainsChatFormat, CollatorWithChains
from readers.metrics import ems, f1_score
from lightrag.llm import ollama_model_complete, ollama_embedding
from utils.const import *
from utils.utils import seed_everything, setup_logger, to_device
from lightrag.llm import ollama_model_complete, ollama_embedding
logger = logging.getLogger(__file__)

READER_NAME_TO_MODEL_NAME_MAP = {
    "llama3": "Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gemma": "google/gemma-7b",
}


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    parser.add_argument("--reader", type=str, default="llama3")
    parser.add_argument("--text_maxlength", type=int, default=4096)
    parser.add_argument("--answer_maxlength", type=int, default=25)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--n_context", type=int, default=None)
    parser.add_argument("--context_type", type=str, default=None)

    # experiment
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--name", type=str, default="llama3")

    args = parser.parse_args()
     # only support ["llama3", "mistral", "gemma"] reader
    return args


# 初始化 LightRAG 模型

WORKING_DIR = "./test"
# Initialize LightRAG with Ollama model


def parse_generated_answer_chat_format(answer):
    if "answer is" in answer:
        idx = answer.find("answer is")
        answer = answer[idx + len("answer is"):].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer


def parse_generated_answer(answer):
    candidate_answers = answer.split("\n")
    answer = ""
    i = 0
    while len(answer) < 1 and i < len(candidate_answers):
        answer = candidate_answers[i].strip()
        i += 1
    answer = parse_generated_answer_chat_format(answer)
    return answer

PARSE_FUNCTION_MAP = {
    "llama3": parse_generated_answer_chat_format,
    "mistral": parse_generated_answer,
    "gemma": parse_generated_answer
}

def evaluate(args, dataloader, rag):
    em_scores_list, f1_scores_list, precision_scores_list, recall_scores_list, num_tokens_list = [], [], [], [], []
    sum = 0
    cnt = 0
    for batch_inputs in tqdm(dataloader, desc="Evaluation", total=len(dataloader)):
        for i in range(sum, sum + 4):  # 从0开始，每次增加4
            print("计数", i)
            if i + 3 < len(dataloader.dataset):
                question = dataloader.dataset.get_example(i)["question"]
                # chains = dataloader.dataset.get_example(i)["chains"]
                print("问题", question)

                # 使用 LightRAG 模型生成回答
                # query = f"Given the context: {chains}, answer the question: {question}. Please provide only the final answer without any extra explanation."
                response = rag.query(question, param=QueryParam(mode="naive"))  # 使用 naive 查询模式
                ans = response["output"].strip()  # 获取生成的文本

                if ans == "noanswer":
                    # 如果回答是 'noanswer'，尝试使用 Tavily 搜索获取额外的上下文
                    tavily_client = TavilyClient(api_key="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")
                    answer = tavily_client.qna_search(query=question)
                    input_prompt = f"Given the answer: {answer} and the question: {question}, give the final answer."
                    response = rag.query(question, param=QueryParam(mode="local"))
                    ans = response["output"].strip()
                    cnt += 1
                    print("经过搜索生成的答案", ans)
                print("生成的答案", ans)

                ans = PARSE_FUNCTION_MAP[args.reader](ans)
            else:
                # 处理最后一批不完整的数据
                for j in range(i, len(dataloader.dataset)):
                    question = dataloader.dataset.get_example(j)["question"]
                    chains = dataloader.dataset.get_example(j)["chains"]
                    print("问题", question)

                    # query = f"Given the context: {chains}, answer the question: {question}, just give me the final answer."
                    response = rag.query(question, param=QueryParam(mode="local"))
                    ans = response["output"].strip()

                    if ans == "noanswer":
                        tavily_client = TavilyClient(api_key="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")
                        context = tavily_client.get_search_context(query=question)
                        input_prompt = f"Given the context: {context}, answer the question: {question}."
                        response = rag.query(input_prompt, param=QueryParam(mode="naive"))
                        ans = response["output"].strip()
                        cnt += 1
                        print("经过搜索生成的答案", ans)
                    print("生成的答案", ans)

                    ans = PARSE_FUNCTION_MAP[args.reader](ans)

            gold = dataloader.dataset.get_example(i)["answers"]
            print("准确答案", gold)
            em_scores_list.append(ems(ans, gold))
            f1, precision, recall = f1_score(ans, gold[0])
            f1_scores_list.append(f1)
            precision_scores_list.append(precision)
            recall_scores_list.append(recall)

        sum += 4

    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "precision": np.mean(precision_scores_list),
        "recall": np.mean(recall_scores_list),
    }
    return metrics

if __name__ == "__main__":

    args = setup_parser()


    # load dataset
    dataset = ReaderDatasetWithChains(data_path="data/hotpotqa/dev_with_reasoning_chains.json", n_context=args.n_context, chain_key="chains")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
        llm_model_name='llama3',  # Your model name
        # Use Ollama embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts,
                embed_model="nomic-embed-text"
            )
        ),
    )
    with open("./test.txt", 'rb') as f:
        data = f.read()
        result = chardet.detect(data)
        encoding = result['encoding']  # 获取编码，检测为 UTF-16

    # 使用检测到的编码读取文件内容
    with open("./test.txt", encoding=encoding) as f:
        text = f.read()
    rag.insert(text)


    dataloader = DataLoader(dataset, batch_size=4, drop_last=False, shuffle=False)

    metrics = evaluate(args,dataloader, rag)
    logger.info("====================== Evaluation Results ======================")
    logger.info("n_context: {}".format(args.n_context))
    logger.info("context_type: {}".format(args.context_type))
    logger.info(metrics)
    logger.info("================================================================")