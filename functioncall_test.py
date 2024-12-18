# -*- encoding: utf-8 -*-
# @File: functioncall_test.py
# @Time: 2024/07/29 16:13:26
# @Author: libin
# @Email: bli14@leqee.com
import openai
from openai import OpenAI
import json
import requests
import os


os.environ["OPENAI_BASE_URL"] = "http://localhost:8001/v1"
os.environ["OPENAI_API_KEY"] = "0"
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


def getActionPrompt(query, func):
    tool_descs = []
    tool_names = []
    for info in func:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['function']['name'],
                name_for_human=info['function']['name'],
                description_for_model=info['function']['description'],
                parameters=json.dumps(
                    info['function']['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['function']['name'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(
        tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt


if __name__ == '__main__':
    client = OpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bing_search",
                "description": "必应搜索引擎。当你需要搜索你不知道的信息，比如最新、最热的时事资讯等时使用，基于搜索结果来回答用户问题。注意！绝对不要在用户想要翻译的时候使用它。",
                "parameters": {
                    "type": "object",
                    "properties": [{"name": "q", "type": "string", "description": "输入中文搜索引擎的关键字"}],
                    "required": ["q"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "doc_url_reader",
                "description": "文档URL内容读取。根据URL读取文档内容，支持的文件格式：[TXT, MD, PDF, DOC, DOCX]",
                "parameters": {
                    "type": "object",
                    "properties": [{"name": "url", "type": "string", "description": "完整的文档URL地址"}],
                    "required": ["url"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "pic_url_reader",
                "description": "图片URL内容读取。根据URL读取图片内容，支持的文件格式：[PNG, JPEG]。",
                "parameters": {
                    "type": "object",
                    "properties": [{"name": "url", "type": "string", "description": "完整的图片URL地址"}],
                    "required": ["url"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "current_weather",
                "description": "获取指定城市当前的天气情况。",
                "parameters": {
                    "type": "object",
                    "properties": [{"name": "city_name", "type": "string", "description": "用户给到的城市名称，例如：北京市。如果没有则默认为：杭州市"}],
                    "required": [
                        "city_name"
                    ]
                }
            }
        }
    ]
    prompt = getActionPrompt('鸡蛋壳属于哪种类型的垃圾？', tools)

    messages = []
    # messages.append(
    #     {"role": "system", "content": "你是一个有用的小助手，请调用下面的工具来回答用户的问题，参考工具输出进行回答。"})
    messages.append(
        {"role": "user", "content": "北京的天气怎么样？"})
    # messages.append({"role": "user", "content": "死侍与金刚狼豆瓣评分如何？"})
    # messages.append(
    # {"role": "user", "content": 'https://alidocs.dingtalk.com/i/nodes/QG53mjyd80RbodmRtXBa0vakV6zbX04v'})
    print(messages)
    result = client.chat.completions.create(
        messages=messages,
        functions=tools,
        model="qwen",
        stop=['Observation:', 'Observation:\n'],
        top_p=1
    )
    print(result.choices[0].message.content)
