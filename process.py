import json

# 读取JSON文件
with open('data/2WIKI/dev_with_reasoning_chains (1).json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)



# 提取所有text字段
texts = []
for item in data:
    for ctx in item['ctxs']:
        texts.append(ctx['triples'])




with open('data/2WIKI/output_1.txt', 'w', encoding='utf-8') as txt_file:
    for text in texts:
        txt_file.write(text + '\n')

print("Text has been extracted and saved to output.txt")
