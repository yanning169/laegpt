import os
import openai
from openai import OpenAI
from langchain.adapters.openai import convert_openai_messages
from langchain_community.chat_models import ChatOpenAI
from sympy.polys.polyconfig import query

from tavily import TavilyClient
from tavily import TavilyClient
import numpy as np

from web_search import web_search

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from lightrag.utils import EmbeddingFunc

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
import os
import chardet
WORKING_DIR = "./2wiki"
import json
from sklearn.metrics import f1_score
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["OPENAI_BASE_URL"] = "https://api.agicto.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO"
os.environ["HF_TOKEN"]="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"
import regex
import string
import unicodedata
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def _normalize(text):
    return unicodedata.normalize('NFD', text)

# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

# modified from HotPotQA official evaluation scirpt: https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

tavilyclient = TavilyClient(api_key="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")
def tavily_search(question):
    # 调用 GPT-4 API 创建 chat completion
    content = tavilyclient.search(question, search_depth="advanced")["results"]

    # Step 3. Setting up the OpenAI prompts
    prompt = [{
        "role": "system",
        "content": f"""You are a detailed and thorough research assistant designed to find answers by breaking down complex questions into simpler steps. Analyze the question, perform a step-by-step search, and answer with only the final, direct answer. 

If the answer is 'yes' or 'no', respond with that single word. If the information isn't found or is incomplete, respond with "noanswer". Here are some examples:

- Question: "What is the place of birth of the director of the film *How I Learned To Love Women*?"
- Steps:
  1. Identify the director of the film *How I Learned To Love Women*.
  2. Search for the director's place of birth.
- Answer: "Director's birthplace"

Other examples:
1. **Question:** "Who is the composer of the score for *Inception* and what is their nationality?"
   **Answer**: "Hans Zimmer, German"

2. **Question:** "What is the nationality of the lead actor in *The Dark Knight*?"
   **Answer**: "Christian Bale, British"

Using this format, for the following question:

**Question**: "{question}"
**Steps**:
1. [Step 1]
2. [Step 2]
**Answer**:
Below are some examples:
Question: "Who is the father of the director of film O Nanna Nalle?\n" Answer:"N. Veeraswamy"
Question: "When is the director of film Freefall (1994 Film)'s birthday?\n" Answer:May 7, 1940"
""" \

    }, {
        "role": "user",
        "content": f'Information: """{content}"""\n\n' \
                   f'Using the above information, answer the following query: "{question}". Think step by step' \

    }]

    # Step 4. Running OpenAI through Langchain
    lc_messages = convert_openai_messages(prompt)
    response = ChatOpenAI(model='gpt-4o-mini').invoke(lc_messages).content

    return response


client = OpenAI(
  api_key="sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO",
  base_url = "https://api.agicto.cn/v1"
)


def self_response(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that provides concise, accurate answers. When responding, give the most relevant information in a straightforward manner. "

            ),
        },
        {
            "role": "user",
            "content": question,  # Use question directly as a string
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure you have access to this model
        messages=messages
    )

    return response.choices[0].message.content


def generate_short_answer(question, detailed_answer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant tasked with generating concise, direct answers. "
                "Given a question and a detailed answer, respond with only the final answer as a few words or a phrase. "
                "Avoid explanations or extra context."
                "Please refer to the example I provided."
            ),
        },
        # Few-shot examples to demonstrate the expected response format
        {
            "role": "user",
            "content": "Question: Who is the father of the director of film O Nanna Nalle?\n"
                       "Detailed Answer: The director of the film 'O Nanna Nalle' is V. Ravichandran, whose father is N. Veeraswamy.\n"
                       "Provide the final answer in a few words or a phrase.",
        },
        {"role": "assistant", "content": "N. Veeraswamy"},

        {
            "role": "user",
            "content": "Question: Where was the place of death of the performer of song Look At Me (John Lennon Song)?\n"
                       "Detailed Answer: The performer of the song 'Look At Me' by John Lennon died in New York City. "
                       "John Lennon was tragically shot and killed outside The Dakota, his residence in Manhattan, New York City, on December 8, 1980.\n"
                       "Provide the final answer in a few words or a phrase.",
        },
        {"role": "assistant", "content": "New York City"},

        {
            "role": "user",
            "content": "Question: When is the director of film Freefall (1994 Film)'s birthday?\n"
                       "Detailed Answer: The director of the film 'Freefall' (1994) is John Irvin. His birthday is on May 7, 1940.\n"
                       "Provide the final answer in a few words or a phrase.",
        },
        {"role": "assistant", "content": "May 7, 1940"},

        # Actual question and detailed answer
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Detailed Answer: {detailed_answer}\n\n"
                "Provide the final answer in a few words or a phrase."
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the appropriate model
        messages=messages,
        max_tokens=10,
        temperature=0
    )

    return response.choices[0].message.content.strip()

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

# 示例JSON文件路径
json_file_path = 'data/2wiki/dev_with_reasoning_chains (1).json'

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Step 0. Importing relevant Langchain libraries



# 初始化F1分数的总和和计数


em_scores_list, f1_scores_list, precision_scores_list, recall_scores_list, num_tokens_list = [], [], [], [], []
cnt=1
for item in data:
    question = item.get('question')
    gold = item['answers']
    print("-----------------------------------------这是第{}个问题-----------------------------------------------".format(cnt))
    print("问题是：{}".format(question))
    ans = rag.query(question, param=QueryParam(mode="naive"))
    # print("详细回答是：{}".format(detailed_ans))
    # ans = generate_short_answer(question,detailed_ans)
    # 使用模型生成答案
    # ans = self_response(question)
    # ans = rag.query(question, param=QueryParam(mode="naive"))
    print("模型自身的回答是：", ans)
    # if(ans=="noanswer"):
    #      ans = web_search(question)
    #      print("网络搜索的回答是：",ans)


    cnt=cnt+1
    print("****************************************最终答案***************************************************")
    print("答案是：",ans)
    print("正确答案是：",gold)
    # 计算F1分数（这里使用'macro'模式，因为答案通常是单个词或短语）
    em_scores_list.append(ems(ans, gold))
    # if not ems(ans, gold):
    #     print(ans, "\t", gold)
    f1, precision, recall = f1_score(ans, gold[0])
    f1_scores_list.append(f1)
    precision_scores_list.append(precision)
    recall_scores_list.append(recall)

metrics = {}
metrics["exact_match"] = np.mean(em_scores_list)
metrics["f1"] = np.mean(f1_scores_list)
metrics["precision"] = np.mean(precision_scores_list)
metrics["recall"] = np.mean(recall_scores_list)

print(metrics["exact_match"],metrics["f1"],metrics["precision"],metrics["recall"])

#
# question="Where was the place of death of the director of film Pals First?"
# # ans = tavily_search(question)
# ans=web_search(question)
# # # detailed_ans = rag.query(question, param=QueryParam(mode="local"))
# # # ans=self_response(question)
# print(ans)


