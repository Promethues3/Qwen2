# -*- encoding: utf-8 -*-
# @File: alpaca2qwen.py
# @Time: 2024/08/05 13:51:41
# @Author: libin
# @Email: bli14@leqee.com
import json
with open('data/alpaca_data_zh_51k.json', 'r', encoding='utf-8') as f1:
    alpaca1 = json.loads(f1.read())

with open('data/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f2:
    alpaca2 = json.loads(f2.read())

alpaca1.extend(alpaca2)

result = {'type': 'chatml', 'messages': []}
with open('data/qwen_alpaca.jsonl', 'w', encoding='utf-8') as f3:

    for a in alpaca1:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]
        messages.append(
            {
                "role": "user",
                "content": a['instruction'] + a['input']
            }
        )
        messages.append({"role": "assistant", "content": a['output']})
        result['messages'] = messages
        f3.write(json.dumps(result, ensure_ascii=False) + '\n')
