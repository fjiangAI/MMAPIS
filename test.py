import yaml
import json
from submodule.openai_api import  *
from submodule.nougat_main import  nougat_predict
from submodule.arxiv_links import get_arxiv_links
from submodule.my_utils import *
import re
import time
from logging.config import fileConfig
import logging
import os
import sys
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import random
import  requests





# params = {
#     "openai_info": openai_info,
#     "proxy": {'http': 'http://127.0.0.1:7890',
#               'https': 'http://127.0.0.1:7890',
#               'ftp': 'ftp://127.0.0.1:7890'},
#     "artile_text": 'djsaiodj',
#     "file_name": 'test.mmd',
#     "gpt_config": None
# }
#
# print('params:',params)
# url = 'http://127.0.0.1:8000/get_summaries/'
# response = requests.post(url=url, json=params)
# print('status_code:',response.status_code)
# print('text:',response.text)


import json

def chat(system_input,user_input):
    base_url = "https://api.ai-gaochao.cn/v1"
    url = base_url+"/chat/completions"
    api_key = "sk-nRjm3MuSfJ9yoDZu987f9f8c0b014052B463046c0587B01c"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    system_msg = "你是一个很有用的助手，擅长使用 JSON 格式回答论文总结\n" + system_input
    messages= [{"role": "system", "content": system_msg}]

    messages.append({'role': 'user', 'content': user_input})


    parameters = {
        "model": "gpt-3.5-turbo-1106",
        "messages": messages,
        "response_format": { "type": "json_object" },
    }

    raw_response = requests.post(url, headers=headers, json=parameters)
    response = json.loads(raw_response.content.decode("utf-8"))
    return response["choices"][0]["message"]['content']

if __name__ == '__main__':
    # logging_path = 'logging.ini'
    # logging.config.fileConfig(logging_path)
    # logger = logging.getLogger('applog')

    yaml_path = './config.yaml'
    with open(yaml_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    openai_info = config["openai"]
    with open(openai_info['prompts_path'], 'rb') as f:
        prompts = json.load(f)
    arxiv_info = config['arxiv']
    nougat_info = config["nougat"]
    proxy = arxiv_info['proxy']
    ignore_titles = openai_info['ignore_title']
    per_min =  3 if openai_info['rate_limit'] else None
    md_path = './data/test/1706.03762.mmd'
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # text = "sdadasdasd"
    init_grid = 2
    # titles, authors, affiliations, chunks = split2pieces(text.strip(), file_name="test.mmd",
    #                                                      max_tokens=16385,
    #                                                      mode="group", ignore_title=ignore_titles,
    #                                                      init_grid=init_grid)
    # system_input = prompts['section summary']['system']
    # user_input =  prompts['section summary']['abstract'] + '```'+chunks[0][1]+'```'
    # res = chat(system_input,user_input)
    # print('res:',res)


    base_url = openai_info['base_url']
    grid = 2
    prompt = prompts['section summary']['system']
    summerizer = OpenAI_Summarizer(api_key= openai_info['api_key'],
                                   proxy = proxy,
                                   summary_prompts=prompts['section summary'],
                                   resummry_prompts=prompts["blog summary"], ignore_titles=ignore_titles,
                                   acquire_mode='url',num_processes=3,base_url=base_url,requests_per_minute=per_min)
    print("summerizer:", summerizer)
    nowtime = time.time()
    filename = 'xxx.mmd'
    titles, authors, affiliations, total_resp, re_respnse = summerizer.summary_with_openai(text,
                                                                                          file_name=filename,
                                                                                          init_grid=grid)
    print(f"total time:{time.time() - nowtime}")
    print('total_resp:', total_resp)
    print('-' * 100)
    for i,re_resp in enumerate(re_respnse):
        print(f"re_resp {i}:", re_resp)
        print('-' * 100)

    print("*" * 100)
    print("*" * 100)
    print()
    n_t = time.time()
    for usage in summerizer.usages:
        print(f"summarizing {usage}...")
        enhance_resp = summerizer.Enhance_Answer(
            original_answer=total_resp,
            summarized_answer=re_respnse[0],
            usage= usage,
        )
        print(f'{usage} enhance_resp:\n', enhance_resp)
        print('-' * 50)
        print()
    print(f"total time:{time.time() - n_t}")
