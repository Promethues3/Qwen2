# -*- encoding: utf-8 -*-
# @File: api_test.py
# @Time: 2024/11/21 15:46:57
# @Author: libin
# @Email: bli14@leqee.com
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="none",  # This is the default and can be omitted
)

text = '这双鞋子搭配这款strength版型的萝卜裤，颜色相互呼应，穿着休闲又时尚。您看小红书很多博也这么搭配。'

prompt = """首先根据我给你的[文本]识别出其中英文和阿迪达斯品牌，生成10句推荐话术要包含这些英文和品牌名称，相似程度要相差很大。
请严格按照json格式回答，不用输出你的思考过程。以下是我给你的示例：
示例：
[文本]
OPT 4INCH L是我们阿迪达斯新出的一款衣服，这款衣服宽松透气，非常适合运动。
回答：
```json
{
    'answer':['我们阿迪达斯新出了一款OPT 4INCH L，非常适合年轻人穿着。', '这款OPT 4INCH L是我们阿迪新出的产品，穿在您身上非常帅气。']
}
请你作答：
[文本]
%s
```
""" % text
print(prompt)
# create a request not activating streaming response
response = client.chat.completions.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=False,
    # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
    stop=[]
)
print(response.choices[0].message.content)
