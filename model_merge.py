# -*- encoding: utf-8 -*-
# @File: model_merge.py
# @Time: 2024/03/26 19:51:59
# @Author: libin
# @Email: bli14@leqee.com
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge_model(model_path, save_path):
    # 分词
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)

    # 模型
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        save_path,
        max_shard_size="2048MB",
        safe_serialization=True
    )  # 最大分片2g


if __name__ == '__main__':
    model_path = 'examples/sft/output_qwen08133'
    save_path = 'models/qwen2-0.5b-instruct-finetune08133'
    merge_model(model_path, save_path)
