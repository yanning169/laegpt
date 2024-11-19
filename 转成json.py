import json
import pandas as pd
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# 加载 JSON 数据
with open('语料.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. 数据提取与分类
categories = ['政策', '行动方案', '规划', '法律法规']
category_counts = {cat: 0 for cat in categories}
documents = []

# 解析文件
for item in data:
    title = str(item.get("标题", ""))  # 确保 title 是字符串
    content = str(item.get("内容", ""))  # 确保 content 是字符串
    # source = item.get("来源", "未知")

    # 尝试提取日期
    date_match = re.search(r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})?", title + content)
    if date_match:
        date = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}"
    else:
        date = "未知"

    # 确定分类
    doc_category = "其他"
    for cat in categories:
        if cat in title:
            category_counts[cat] += 1
            doc_category = cat
            break

    # 记录文档
    documents.append({
        "标题": title,
        "时间": date,
        # "来源": source,
        "类别": doc_category
    })

# 转换为 Pandas 数据表
df = pd.DataFrame(documents)

# 2. 表格输出
print("统计表格:")
print(df)

# 3. 按月份统计数量并绘制柱状图
df['月份'] = df['时间'].str[5:7]
month_counts = df['月份'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(month_counts.index, month_counts.values, color='skyblue')
plt.xlabel('月份')
plt.ylabel('文件数量')
plt.title('文件数量按月份统计')
plt.show()

# 4. 文件关注点分析（词频 & 词云）
all_text = " ".join(item.get("内容", "") for item in data)
words = jieba.lcut(all_text)
word_counts = Counter(words)
common_words = word_counts.most_common(50)

# 生成词云
wordcloud = WordCloud(font_path="simhei.ttf", width=800, height=400, background_color="white").generate_from_frequencies(word_counts)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("文件关注点词云")
plt.show()

# 5. 统计会议相关信息
conference_keywords = ['大会', '会议', '论坛', '研讨会']
conference_data = []
for item in data:
    title = str(item.get("标题", ""))  # 确保 title 是字符串
    content = str(item.get("内容", ""))  # 确保 content 是字符串
    # source = item.get("来源", "未知")
    if any(kw in title or kw in content for kw in conference_keywords):
        date_match = re.search(r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})?", title + content)
        if date_match:
            date = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}"
        else:
            date = "未知"
        conference_data.append({
            "会议名称": title,
            "时间": date,
            # "来源": source,
            "主要文件": title,
            "演讲者": "未知"  # 如需提取演讲者可结合更具体的数据格式
        })

# 转换为 Pandas 数据表
conference_df = pd.DataFrame(conference_data)

# 输出会议统计表
print("会议统计表:")
print(conference_df)

# 按省市统计
city_counts = df['来源'].value_counts()

plt.figure(figsize=(12, 6))
city_counts.plot(kind='bar', color='green')
plt.title("按省市文件数量统计")
plt.ylabel("数量")
plt.xlabel("来源")
plt.show()
