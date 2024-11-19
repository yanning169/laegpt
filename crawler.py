import requests
from bs4 import BeautifulSoup
import os

def crawl_policies(base_url, save_dir):
    """爬取低空经济政策文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # 示例：提取PDF链接
    for link in soup.find_all("a", href=True):
        href = link['href']
        if href.endswith(".pdf"):
            pdf_url = href if href.startswith("http") else base_url + href
            save_path = os.path.join(save_dir, os.path.basename(href))
            download_file(pdf_url, save_path)

def download_file(url, save_path):
    """下载文件"""
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"下载完成：{save_path}")
