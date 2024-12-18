# -*- encoding: utf-8 -*-
# @File: glaive2Qwen.py
# @Time: 2024/08/01 13:09:57
# @Author: libin
# @Email: bli14@leqee.com
import json


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""


def buildReactSystem(tools):
    tools_text, tools_name_text = [], []
    system = ""
    for func_info in tools:
        name_m, name_h = func_info['name'], func_info['name']

        desc_m = func_info['description']
        tool = TOOL_DESC.format(
            name_for_model=name_m,
            name_for_human=name_h,
            description_for_model=desc_m,
            parameters=json.dumps(
                func_info["parameters"],
                ensure_ascii=False
            ),
        )
        tools_text.append(tool)
        tools_name_text.append(name_m)
    tools_text = "\n\n".join(tools_text)
    tools_name_text = ", ".join(tools_name_text)
    system += "\n\n" + REACT_INSTRUCTION.format(
        tools_text=tools_text,
        tools_name_text=tools_name_text,
    )
    system = system.lstrip("\n").rstrip()
    return system


def buildAnswer(conversations, tools):
    # 先准备回答的格式
    questionFormat = 'Question: {}\n'
    thoughtFormat = 'Thought: {}\n'
    actionFormat = 'Action: {}\n'
    actionInputFormat = 'Action Input: {}\n'
    observationFormat = 'Observation: {}\n'
    finalAnswerFormat = 'Final Answer: {}'

    result = {"type": "chatml"}  # 最终结果

    # 初始化，首先将多轮对话却别开，区分逻辑是最后一个角色为gpt
    message = []
    glaive2Messages = list()
    for conversation in conversations:
        messageConv = conversation
        message.append(messageConv)
        if conversation['from'] == 'gpt':
            if message:
                glaive2Messages.append(message)
                message = []

    # 开始记录对话
    messages = []
    if tools:
        messages.append({'role': 'system', 'content': buildReactSystem(tools)})
    else:
        messages.append(
            {'role': 'system', 'content': 'You are a helpful assistant.'})

    for glaive2Message in glaive2Messages:

        # 判断在每轮对话中是否需要调用function，如果是则需要按模板构造回答
        for g in glaive2Message:
            if g == 'true':
                continue

            if g['from'] == 'human':
                query = g['value']
                if tools:
                    question = questionFormat.format(g['value'])

                messages.append(
                    {
                        'role': 'user',
                        'content': query
                    }
                )
            elif (g['from'] == 'function_call') and ('true' not in glaive2Message):
                glaive2Message.append('true')

        if 'true' in glaive2Message:
            # 确认调用了function按模板构造回答
            for g in glaive2Message[:-1]:

                if g['from'] == 'function_call':
                    thought = thoughtFormat.format(
                        f"我应该使用{json.loads(g['value'])['name']}工具来回答这个问题"
                    )
                    action = actionFormat.format(
                        json.loads(g['value'])['name']
                    )
                    actionInput = actionInputFormat.format(
                        json.dumps(
                            json.loads(g['value'])['arguments'],
                            ensure_ascii=False
                        )
                    )
                if g['from'] == 'observation':
                    observation = observationFormat.format(g['value'])
                if g['from'] == 'gpt':
                    finalAnswer = finalAnswerFormat.format(g['value'])
            messages.append(
                {
                    'role': 'assistant',
                    'content': thought + action + actionInput + observation + finalAnswer
                }
            )

        else:
            # 没有调用function，按正常对话构造
            for g in glaive2Message:
                if g['from'] == 'gpt':
                    messages.append(
                        {
                            'role': 'assistant',
                            'content': g['value']
                        }
                    )
    result['messages'] = messages
    return result


if __name__ == '__main__':
    glaivePath0 = 'data/glaive_toolcall_zh_1k.json'
    glaivePath1 = 'data/glaive_toolcall_10k.json'
    outputPath = 'data/qwen_toolcall_zh_1k1.jsonl'

    with open(glaivePath0, 'r', encoding='utf-8') as f:
        glaiveToolCalls = json.load(f)

    with open(glaivePath1, 'r', encoding='utf-8') as f:
        glaiveToolCalls.extend(json.load(f)[:1000])

    for glaiveToolCall in glaiveToolCalls:
        result = buildAnswer(
            glaiveToolCall['conversations'],
            json.loads(glaiveToolCall['tools'])
        )
        # print(json.dumps(result, ensure_ascii=False))
        with open(outputPath, 'a', encoding='utf-8') as fo:
            fo.write(json.dumps(result, ensure_ascii=False) + '\n')

    # print(glaiveToolCall)
