import seaborn as sns
from bs4 import BeautifulSoup
import re
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from submodule.openai_api import split2pieces,assgin_prompts
import yaml
import os
from tqdm import tqdm
from submodule.img_parser import parser_img_from_pdf
from submodule.my_utils import save_mmd_file
from pathlib import Path
from submodule.openai_api import OpenAI_Summarizer
import logging




def summary_file_ls(summerizer,
                    file_list,
                    pdf_list,
                    out_dir="./res/cs_2023/temp"):
    file_paths = []
    file_list = tqdm(file_list,position=0, leave=True)
    for file_path,pdf_path in zip(file_list,pdf_list):
        file_name = os.path.basename(file_path)
        filename = file_name.split(".")[0]
        save_dir = out_dir + f"/{filename}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_list.set_description("Processing %s" % file_name)
        with open(file_path, "r",encoding="utf-8") as f:
            file_text = f.read()
        titles, authors, affiliations, summaries, re_respnse = summerizer.summary_with_openai(file_text,file_name=filename)
        with open(os.path.join(out_dir, filename + ".md"), "w",encoding="utf-8") as f:
            f.write("# [Summary]\n\n"+summaries+"\n\n# [ReSummary]\n\n"+re_respnse[0])

        img_paths = parser_img_from_pdf(pdf_path=pdf_path,save_dir=os.path.join(save_dir,"img"))
        img_content = "# Images & Tables:\n\n"
        for i,img_path in enumerate(img_paths["img"]):
            img_path = "./img/" + os.path.basename(img_path)
            img_content += f"![Image {i}]({img_path})\n\n"
        table = "# Tables:\n"+"\n\n\n".join(re_respnse[-1])
        re_respnse = "\n\n".join(re_respnse[:-1])
        summaries = img_content + table + summaries
        re_respnse = img_content + table + re_respnse
        raw_file_name = Path(file_name).stem + "_raw_" + ".mmd"
        inter_file_name = Path(file_name).stem + "_integrate_" + ".mmd"
        file_paths = save_mmd_file(save_texts=[summaries, re_respnse], file_names=[raw_file_name, inter_file_name],
                                  save_dirs=[save_dir, save_dir])
        file_paths.append(file_path)
    logging.info(f"Save file to {out_dir}")
    return file_paths

def transfer2pdf(file_list):
    file_ls = []
    for file in file_list:
        file = file.replace("raw_mmd","pdf").replace("mmd","pdf")
        file = os.path.join(os.path.dirname(file),os.path.basename(file).replace('_','.'))
        file_ls.append(file)
    return file_ls

if __name__ == '__main__':
    cs_2023_file_dir = "./res/cs_2023/raw_mmd"
    cs_2017_file_dir = "./res/cs_2017/raw_mmd"
    math_2023_file_dir = "./res/math_2023/raw_mmd"
    cs_2023_file_list = [os.path.join(cs_2023_file_dir, file_name) for file_name in os.listdir(cs_2023_file_dir)]
    cs_2017_file_list = [os.path.join(cs_2017_file_dir, file_name) for file_name in os.listdir(cs_2017_file_dir)]
    math_2023_file_list = [os.path.join(math_2023_file_dir, file_name) for file_name in os.listdir(math_2023_file_dir)]
    prompts_path = './prompts_config.json'
    with open(prompts_path, "r") as f:
        prompts_config = json.load(f)

    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ignore_titles = config["openai"]["ignore_title"]
    openai_info = config["openai"]
    summary_prompts = prompts_config["section summary"]
    summary_prompts.pop("system")
    resummary_prompts = prompts_config["blog summary"]
    arxiv_info = config['arxiv']
    nougat_info = config["nougat"]
    proxy = arxiv_info['proxy']
    arxiv_info = config['arxiv']
    ignore_titles = openai_info['ignore_title']
    headers = arxiv_info['headers']
    base_url = openai_info['base_url']





    # cs_2017_pdf_list = transfer2pdf(cs_2017_file_list)
    pdf_list = ["./data/test/1706.03762.pdf"]
    mmd_list = ["./data/test/1706.03762.mmd"]
    rate_limit = None

    summerizer = OpenAI_Summarizer(openai_info['api_key'],
                                   proxy,
                                   summary_prompts=summary_prompts,
                                   resummry_prompts=resummary_prompts,
                                   ignore_titles=ignore_titles,
                               acquire_mode='url',
                                   num_processes=6,
                                   base_url=base_url,requests_per_minute=rate_limit,
                               model_config = openai_info['model_config']
                                 )

    print("summerizer:",summerizer)
    file_paths = summary_file_ls(summerizer,
                                 file_list=mmd_list,
                                 pdf_list= pdf_list,
                                 out_dir="./data/test")
