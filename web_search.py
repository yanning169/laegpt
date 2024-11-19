import argparse

from langchain.chains.question_answering.map_reduce_prompt import messages
from openai import OpenAI
from tqdm import tqdm
import json
from util import extract_keywords, select_relevants
import os
import chardet
WORKING_DIR = "./2wiki"
import json
from sklearn.metrics import f1_score
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["OPENAI_BASE_URL"] = "https://api.agicto.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO"
os.environ["HF_TOKEN"]="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"
os.environ["TAVILY_API_KEY"] = "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"
import requests
from bs4 import BeautifulSoup

from transformers import T5ForSequenceClassification, T5Tokenizer

import sys

def generate_knowledge_q(questions, task, openai_key):
    if task == '2wiki':
        queries = extract_keywords(questions, task, openai_key)
    return queries

def Search(queries, search_path, search_key):
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []
    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps(
            {
                "q": query
            }
        )
        headers = {
            'X-API-KEY': search_key,
            'Content-Type': 'application/json'
        }

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except (requests.exceptions.RequestException, ValueError):
                reconnect += 1
                print('url: {} failed * {}'.format(url, reconnect))
        # result = response.text
        result = json.loads(response.text)
        if "organic" in result:
            results = result["organic"][:10]
        else:
            results = query
        responses.append(results)

        search_dict = [{"queries": query, "results":results}]
        search_results.extend(search_dict)
    if search_path != 'None':
        with open(search_path, 'w') as f:
            output = json.dumps(search_results, indent=4)
            f.write(output)
    return search_results

def test_page_loader(url):
    import requests
    from bs4 import BeautifulSoup
    import signal
    def handle(signum, frame):
        raise RuntimeError
    reconnect = 0
    while reconnect < 3:
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(180)
            response = requests.get(url)
            break
        except (requests.exceptions.RequestException, ValueError, RuntimeError):
            reconnect += 1
            print('url: {} failed * {}'.format(url, reconnect))
            if reconnect == 3:
                return []
    try:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return []
    if soup.find('h1') is None or soup.find_all('p') is None:
        return []
    paras = []
    title = soup.find('h1').text
    paragraphs = soup.find_all('p')
    for i, p in enumerate(paragraphs):
        if len(p.text) > 10:
            paras.append(title + ': ' + p.text)
    return paras

client = OpenAI(
  api_key="sk-hisVuPSQjfmRu9evs356q9PAS5fT46D81LNJXGBoz6qYuNzO",
  base_url = "https://api.agicto.cn/v1"
)


def evaluator(query, page_content):
    messages = [
        {
            "role": "system",
            "content": (
                f""""
        As an evaluator, your task is to assess the relevance of retrieved documents in answering the user's question.

        Retrieved Document:
        --------------
        {page_content}
        User Question:
        --------------
        {query}
        Evaluation Criteria:
        - Consider whether the document contains keywords or topics related to the user's question.
        - The evaluation should not be overly strict; the main goal is to identify and filter out obviously irrelevant results.

        Decision:
        - Assign a binary score to indicate the relevance of the document.
        - If the document is relevant to the question, use 'yes'; if not, use 'no'.

        Please provide your binary score ('yes' or 'no') below to indicate the relevance of the document to the user's question."""
            ),
        },
        {
            "role": "user",
            "content": query,  # Use question directly as a string
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure you have access to this model
        messages=messages
    )

    return response.choices[0].message.content

def visit_pages(question, web_results, output_file, device):
    # Ensure only 1 question is passed
    assert len(question) == 1, f"Expected 1 question, got {len(question)}"

    top_n = 5
    titles = []
    urls = []
    snippets = []
    queries = []

    # Loop over each web result and collect the page titles, urls, and snippets
    for i, result in enumerate(web_results[:]):
        title = []
        url = []
        snippet = []
        if isinstance(result["results"], list):  # Ensure the "results" is a list
            for page in result["results"][:top_n]:  # Only take the top N results
                title.append(page["title"])
                url.append(page["link"])
                snippet.append(page.get("snippet", page["title"]))  # Use title as fallback if no snippet
        else:
            titles.append([])  # In case of non-list results, handle them here
            urls.append([])
            snippets.append([result["results"]])
            queries.append(result["queries"])
            continue

        titles.append(title)
        urls.append(url)
        snippets.append(snippet)
        queries.append(result["queries"])

    output_results = []
    progress_bar = tqdm(range(1), desc="Visiting page content...")  # Only 1 question, so range(1)

    # Iterate over the results and extract relevant content
    for title, url, snippet, query in zip(titles, urls, snippets, queries):
        if not url:  # If no URLs are present, use snippet as result
            results = '; '.join(snippet)
        else:
            strips = []
            for u in url:
                strips += test_page_loader(u)  # Extract content from the URL

            if not strips:  # If no content is extracted, fallback to snippet
                results = '; '.join(snippet)
            else:
                results = []
                for page_content in strips:
                    # Use evaluator to check relevance of the page content
                    message = evaluator(query, page_content)
                    if message == "yes":
                        results.append(page_content)

                if not results:  # If no relevant results, fallback to snippet
                    results = '; '.join(snippet)

        output_results.append(results.replace('\n', ' '))  # Clean up result text
        progress_bar.update(1)

    # Save the results to the output file
    with open(output_file, 'w') as f:
        f.write('#')
        f.write('\n#'.join(output_results))

    return output_results



def web_search(question):
    # Step 1: Generate search queries based on the question
    search_queries = extract_keywords(question)

    # Step 2: Perform web search
    search_results = Search(search_queries, search_path="None", search_key=os.getenv("TAVILY_API_KEY"))

    # Step 3: Visit pages and extract content
    context_results = visit_pages([question], search_results, output_file="output_results.txt",
                                  device="cpu")

    # Step 4: Use extracted content as context for GPT response
    context = " ".join(context_results)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant that provides concise, accurate answers. "
                f"Given the context: {context}, answer the question: {question}. "
                f"Please provide only the final answer without any extra explanation. "
                "If the answer is 'yes' or 'no', respond with only 'yes' or 'no'. "
                # "If you don't know the answer, respond with 'noanswer'."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content
