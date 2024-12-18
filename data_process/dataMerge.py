# -*- encoding: utf-8 -*-
# @File: dataMerge.py
# @Time: 2024/08/05 20:38:10
# @Author: libin
# @Email: bli14@leqee.com
import json
import random


with open('data/qwen_alpaca.jsonl', 'r', encoding='utf-8') as f1:
    data1 = f1.read().strip().split('\n')
    data1 = data1[:1000]

with open('data/qwen_toolcall_zh_1k1.jsonl', 'r', encoding='utf-8') as f2:
    data2 = f2.read().strip().split('\n')
data1.extend(data2)

data1 = [json.loads(i) for i in data1]
random.shuffle(data1)

with open('data/trainData0813.jsonl', 'w', encoding='utf-8') as f3:
    for i in data1:
        f3.write(json.dumps(i) + '\n')
