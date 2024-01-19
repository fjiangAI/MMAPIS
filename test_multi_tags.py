import streamlit as st
import time
from typing import Dict, List, Literal, Optional, Union
import os
import yaml
import sys
import logging
import zipfile
import rarfile
from pathlib import Path
import requests
import json
import io
import pandas as pd
from datetime import datetime
from playsound import playsound
import random
from functools import partial

from streamlit_option_menu import option_menu
from stqdm import stqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from submodule.arxiv_links import *
from submodule.openai_api import *
from submodule.my_utils import *
from submodule.my_utils import init_logging,handle_errors
#--------------------------------------preprocess--------------------------------------
@st.cache_data
def init_logger():
    log_path = 'logging.ini'
    logger = init_logging(log_path)
    return logger


@st.cache_data
def filter_links(links):
    selected_index = [index for index, selected in links if selected]
    return selected_index

@st.cache_data
def load_file(uploaded_file):
    return  get_pdf_list(uploaded_file)


@st.cache_data
def init_config(set_none:bool = False):
    yaml_path = 'config.yaml'
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        logging.error(f'Config file not found at {yaml_path}')
        sys.exit(1)
    openai_info = config["openai"]
    with open(openai_info['prompts_path'], 'rb') as f:
        prompts = json.load(f)
    arxiv_info = config['arxiv']
    nougat_info = config["nougat"]
    proxy = arxiv_info['proxy']
    headers = arxiv_info['headers']
    base_url = openai_info['base_url']
    if set_none:
        proxy = headers = None
    return openai_info, arxiv_info, nougat_info,prompts, proxy, headers,base_url

class Args:
    def __init__(self,nougat_info,arxiv_info, **kwargs):
        self.checkpoint = kwargs.get("checkpoint", Path(nougat_info["check_point"]))
        self.out = kwargs.get("out", Path(nougat_info["out"]))
        self.recompute = kwargs.get("recompute", True)
        self.markdown = kwargs.get("markdown", True)
        self.pdf = kwargs.get("pdf", [Path(i) for i in nougat_info["pdf"]])
        self.num_process = kwargs.get("num_process", 3)
        self.kw = kwargs.get("kw", arxiv_info['key_word'])
        self.rate_limit = 3 if var_openai_info['rate_limit'] else None

def init_session_state(url_reset:bool = False,
                       pdf_reset:bool = False):
    print("url_reseting:",url_reset,"pdf_reseting:",pdf_reset)
    if "run_model" not in st.session_state :
        st.session_state["run_model"] = False


    if "generated_summary" not in st.session_state or url_reset:
        # st.session_state["generated_summary"][index]:[previous summary,regenerate summary]
        st.session_state["generated_summary"] = [[None] * max_page_num for _ in range(max_pdf_num)]
        st.session_state["score"] = [None] * max_pdf_num

    if "generated_result" not in st.session_state or url_reset:
        st.session_state["generated_result"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "usage" not in st.session_state or url_reset:
        st.session_state["usage"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "num_pages" not in st.session_state or url_reset:
        print("get in num_pages")
        st.session_state["num_pages"] = [0] * max_pdf_num


    if "pdf_run_model" not in st.session_state:
        st.session_state["pdf_run_model"] = False

    if "pdf_generated_summary" not in st.session_state or pdf_reset:
        # st.session_state["pdf_generated_summary"][index]:[previous summary,regenerate summary]
        st.session_state["pdf_generated_summary"] = [[None] * max_page_num for _ in range(max_pdf_num)]
        st.session_state["pdf_score"] = [None] * max_pdf_num

    if "pdf_generated_result" not in st.session_state or pdf_reset:
        st.session_state["pdf_generated_result"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "pdf_usage" not in st.session_state or pdf_reset:
        st.session_state["pdf_usage"] = [[None] * max_pdf_num for _ in range(max_pdf_num)]



    if "pdf_num_pages" not in st.session_state or pdf_reset:
        st.session_state["pdf_num_pages"] = [0] * max_pdf_num




#--------------------------------------fastapi request--------------------------------------
@st.cache_data
def get_links(keyword, proxy,max_num=10,line_length=15,searchtype='all',show_abstract='show',
              order='-announced_date_first',size=50,show_meta_data=True,daily_type = 'cs',headers = None):
    url = var_req_url + '/get_links/'
    params = {
        "proxies": proxy,
        "headers": headers,
        "max_num": max_num,
        "line_length": line_length,
        "searchtype": searchtype,
        "abstracts": show_abstract,
        "order": order,
        "size": size,
        "show_meta_data": show_meta_data,
        "daily_type": daily_type
    }
    params = dict_filter_none(params)
    params['key_word'] = keyword
    response = requests.post(url, json=params)
    json_info = custom_response_handler(response,func_name='get_links')
    time.sleep(4)
    if not "error" in json_info:
        return json_info['links'], json_info['titles'], json_info['abstract'], json_info['authors']
    else:
        return None,None,None,json_info.get('error',"error")


@st.cache_data
def get_model_predcit(_proxy = None,_headers = None,pdf_name:str=None,**kwargs):
    pdf = kwargs.get("pdf", '')[0]
    if isinstance(pdf,bytes):
        text = f"this article is about saying {pdf_name}"
    else:
        text = pdf + f"this article is about saying {pdf}"

    time.sleep(4)
    return [text],[pdf_name]





@st.cache_data
def get_summary(
                api_key:dict,
                proxy:dict,
                article:str,
                file_name:str,
                num_iterations:int=3,
                requests_per_minute:Union[int,None] = None,
                summary_prompts: dict = None,
                resummry_prompts: dict = None,
                ignore_titles: list = None,
                base_url: str = "https://openai.huatuogpt.cn/v1",
                acquire_mode:Literal['url','openai'] = 'url',
                prompt_factor:float = 0.8,  # prompt tokens / total tokens
                num_processes:int = 3,
                init_grid:int = 2,
                split_mode:str = 'group',
                gpt_config:Union[Dict,None] = None):
    # gpt_config example:
                    # {
                    #     "model": "gpt-3.5-turbo-16k-0613",
                    #     "temperature": 0.9,
                    #     "max_tokens": 16385,
                    #     "top_p": 1,
                    #     "frequency_penalty": 0,
                    #     "presence_penalty": 0,
                    # }
    title = file_name.split(".")[0] if file_name else "some title"
    authors = "xxxx,xxxx"
    affiliations = "xxxx,xxxx"
    total_resp = article
    re_respnse = ["re_summary :\n" +article,"score: -xxx : 9"]
    time.sleep(2)
    return title,authors,affiliations,total_resp,re_respnse

def get_enhance_answer(
                   api_key:str,
                   raw_summary:str,
                   regenerate_summary:str,
                   index: int,
                   page_idx:int,
                   usage:Literal['None','blog','speech','regenerate'] = 'None',
                   **kwargs):
    assert usage in ['blog','speech','regenerate'],f"usage must be in ['None','blog','speech','regenerate'],but got {usage}"
    summarizer_config = dict(**kwargs)
    params = {
        "api_key": api_key,
        "original_answer": raw_summary,
        "summarized_answer": regenerate_summary,
        "usage": usage,
        "summarizer_config": summarizer_config,
    }
    params = dict_filter_none(params)
    url = var_req_url + '/enhance_answer/'
    response = requests.post(url, json=params)
    json_info = custom_response_handler(response,func_name='enhance_answer')
    logger.info(f"enhance answer:{json_info}")
    time.sleep(2)
    if not "error" in json_info:
        text = json_info['enhanced_answer']
        st.session_state["generated_result"][index][page_idx] = text.strip()
        st.session_state["usage"][index][page_idx] = usage
        if usage == 'regenerate':
            st.session_state["num_pages"][index] += 1
            st.session_state["generated_summary"][index].append(text.strip())

    else:
        logger.error(f"enhance answer error:{json_info.get('error','error')}")



@st.cache_data
def text2speech(text):

    flag, json_info = text_to_speech_multi(
        text = text,
        num_processes=var_num_processes,
        return_type='bytes'
    )
    if flag and json_info:
        return flag,json_info
        # _place_holder.audio(json_info, format='audio/mp3')
        # _place_holder.download_button(label="下载音频", data=json_info,
        #                              file_name=f"{str(filename)}.mp3",
        #                              key=f"download_{str(filename)}_{str(temp_time)}_{random_num}")

    elif json_info is None:
        flag = False
        return flag,"文本过长，分割失败，请缩短文本长度"
        # _place_holder.text(f"文本过长，分割失败，请缩短文本长度")
    else:
        return flag,f"语音合成失败，错误信息：{json_info}"
        # _place_holder.text(f"语音合成失败，错误信息：{json_info}")




@st.cache_data
def get_enhance_answer(
                   api_key:str,
                   raw_summary:str,
                   regenerate_summary:str,
                   index: int,
                   page_idx:int,
                   usage:Literal['None','blog','speech','regenerate'] = 'None',
                   **kwargs):
    assert usage in ['blog','speech','regenerate'],f"usage must be in ['None','blog','speech','regenerate'],but got {usage}"
    summarizer_config = dict(**kwargs)
    params = {
        "api_key": api_key,
        "original_answer": raw_summary,
        "summarized_answer": regenerate_summary,
        "usage": usage,
        "summarizer_config": summarizer_config,
    }
    params = dict_filter_none(params)
    url = var_req_url + '/enhance_answer/'
    response = requests.post(url, json=params)
    json_info = custom_response_handler(response,func_name='enhance_answer')
    logger.info(f"enhance answer:{json_info}")
    if not "error" in json_info:
        text = json_info['enhanced_answer']
        st.session_state["generated_result"][index][page_idx] = text.strip()
        st.session_state["usage"][index][page_idx] = usage
        if usage == 'regenerate':
            st.session_state["num_pages"][index] += 1
            st.session_state["generated_summary"][index].append(text.strip())

    else:
        logger.error(f"enhance answer error:{json_info.get('error','error')}")

@st.cache_data
def get_pdf_enhance(api_key:str,
                   raw_summary:str,
                   regenerate_summary:str,
                   index: int,
                   page_idx:int,
                   usage:Literal['None','blog','speech','regenerate'] = 'None',
                   **kwargs):
    assert usage in ['blog', 'speech',
                     'regenerate'], f"usage must be in ['None','blog','speech','regenerate'],but got {usage}"
    summarizer_config = dict(**kwargs)
    params = {
        "api_key": api_key,
        "original_answer": raw_summary,
        "summarized_answer": regenerate_summary,
        "usage": usage,
        "summarizer_config": summarizer_config,
    }
    params = dict_filter_none(params)
    url = var_req_url + '/enhance_answer/'
    response = requests.post(url, json=params)
    json_info = custom_response_handler(response, func_name='enhance_answer')


    if not "error" in json_info:
        text = json_info['enhanced_answer']
        st.session_state["pdf_generated_result"][index][page_idx] = text.strip()
        st.session_state["pdf_usage"][index][page_idx] = usage
        if usage == 'regenerate':
            st.session_state["pdf_num_pages"][index] += 1
            st.session_state["pdf_generated_summary"][index].append(text.strip())

    else:
        logger.error(f"enhance answer error:{json_info.get('error', 'error')}")

#--------------------------------------postprocess--------------------------------------


def display_result(answer_message,
                   re_respnse,
                   summary_box_height,
                   pdf_name):
    pdf_name = pdf_name.split(".")[0] if pdf_name else "some title"
    re_respnse, score = re_respnse
    temp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    random_num = random.randint(0,100000)
    answer_message.text_area("Summary:", f"{re_respnse.strip()}", height=summary_box_height,key=f"summary_{pdf_name}_{temp_time}_{random_num}")
    answer_message.text_area("Score:", f"{score}", height=summary_box_height,key=f"score_{pdf_name}_{temp_time}_{random_num}")
    answer_message.caption(f"not enough? click the button to regenerate summary")


def display_download_button(place_holder,
                            summaries,
                            re_respnse,
                            score,
                            pdf_name,
                            usage:Literal['blog','regenerate','speech'] = 'regenerate'):
    col1, col2, col3 = place_holder.columns(3)
    ramdom_num = random.randint(0,100000)
    temp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    col1.download_button(label="下载输入结果为mmd格式", data=summaries.encode('utf-8'),
                         file_name=f"{str(pdf_name)}_summary.mmd",
                         key=f"download_{str(pdf_name)}_summary_{temp_time}_{ramdom_num}")
    col2.download_button(label=f"下载为{usage}结果为mmd格式", data=re_respnse.encode('utf-8'),
                         file_name=f"{str(pdf_name)}_re_summary.mmd",
                         key=f"download_{str(pdf_name)}_re_summary_{temp_time}_{ramdom_num}")
    col3.download_button(label="下载打分结果为mmd格式", data=str(score).encode('utf-8'),
                         file_name=f"{str(pdf_name)}_score.mmd",
                         key=f"download_{str(pdf_name)}_score_{temp_time}_{ramdom_num}")

@st.cache_data
def get_pdf_length(pdf:Union[str,bytes,Path]):
    assert isinstance(pdf,(str,bytes,Path)),f"pdf must be str,bytes or Path,but got {type(pdf)}"
    return get_pdf_doc(pdf)[1]


def plot_df_res(df):
    fig, ax = plt.subplots(figsize = (10,6))
    l1, = ax.plot(df["pdf_length"], df["get summary"], color='red', linestyle='--', label='get summary')
    l2, = ax.plot(df["pdf_length"], df["model predict"], color='blue', linestyle='--', label='model predict')
    markers = ['o', 'x', 's', 'v', 'D', 'p', 'P', 'h', '8', 'd', 'H', '*', 'X']
    lengens = []
    for index, row in df.iterrows():
        # color = plt.cm.viridis(index / len(df))
        a,b,c = row["pdf_length"],row["get summary"],row["model predict"]
        ax.scatter([a],[b],color="red",marker=markers[index])
        ax.scatter([a],[c],color="blue",marker=markers[index])
        # cur_marker = mlines.Line2D([], [], color='black', marker='o', markersize=10, label=f"pdf_{index}")
        cur_marker = plt.scatter([], [], color='black', marker=markers[index], label=f"pdf_{index}")
        lengens.append(cur_marker)
    lengens.extend([l1,l2])
    ax.set_title('model predict & get sumamry time cost with different pdf length')
    ax.set_xlabel('length of pdf(page)')
    ax.set_ylabel('time cost(s)')
    ax.legend(handles=lengens, loc="best")
    return fig


if __name__ == "__main__":
    #--------------------------------------init--------------------------------------
    max_pdf_num = 25
    max_page_num = 10
    color_ls = ["blue", "green", "orange", "red", "violet", "gray", "rainbow"]

    # Streamlit应用程序标题
    st.set_page_config(page_title="paper parser demo", page_icon=":smiley:", layout="wide", initial_sidebar_state="auto")
    var_openai_info, var_arxiv_info, var_nougat_info,var_prompts, var_proxy,var_headers,var_base_url = init_config(set_none= True)
    logger = init_logger()
    var_ignore_titles = var_openai_info['ignore_title']




    ##--------------------------------------sidebar--------------------------------------##
    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    if add_selectbox == 'Email':
        user_email = st.sidebar.text_input("Your email", key="email")
    elif add_selectbox == 'Home phone':
        user_phone = st.sidebar.text_input("Your phone", key="phone")
    else:
        user_mobile = st.sidebar.text_input("Your mobile phone", key="mobile")

    # 显示模型超参数
    var_linelength = st.sidebar.slider("line length",10,40,18,help="line length of arxiv search result")
    summary_box_height = st.sidebar.slider("summary box height",100,1000,400,step=50,help=" height of summary box")
    st.sidebar.header("openai info")
    var_max_links = st.sidebar.slider("max links", 0, max_pdf_num, 5,help="max links of arxiv search result")
    var_api_key = st.sidebar.text_input("openai api key", key="api_key",help="your openai api key, e.g. sk-xxxxxx")
    # var_openai_info['api_key'] = var_api_key if var_api_key else var_openai_info['api_key']
    var_summary_prompt = None
    var_resummary_prompt = None
    with st.sidebar.expander("prompts optional"):
        var_summary_prompt = st.file_uploader("summary prompt", type=["json"], key="summary_prompt",help = "summary prompt json file,e.g.{'system':'','general_summary':''}")
        var_resummary_prompt = st.file_uploader("resummary prompt", type=["json"], key="resummary_prompt",help = "resummary prompt json file,e.g.{'system':'','overview':''}")
        var_summary_prompt = json.loads(var_summary_prompt.read()) if var_summary_prompt else None
        var_prompts['section summary'] = var_summary_prompt if var_summary_prompt else var_prompts['section summary']
        var_resummary_prompt = json.loads(var_resummary_prompt.read()) if var_resummary_prompt else None
        var_prompts["blog summary"] = var_resummary_prompt if var_resummary_prompt else var_prompts["blog summary"]

    with st.sidebar.expander("Model config Options"):
        var_num_processes = st.slider("num processes", 1, 15, 6, help="num processes of gpt-3.5 model")
        var_max_token = st.slider("max token", 0, 16385, 16000,help="max token of total tokens(including prompt tokens) of gpt-3.5 model")
        var_temperature = st.slider("temperature", 0.0, 2.0, 0.5,help="temperature of gpt-3.5 model")
        var_top_p = st.slider("top p", 0.0, 1.0, 1.0,help="top p of gpt-3.5 model")
        var_frequency_penalty = st.slider("frequency penalty", -2.0, 2.0, 0.1, help="frequency penalty of gpt-3.5 model")
        var_presence_penalty = st.slider("presence penalty", -2.0, 2.0, 0.2, help="presence penalty of gpt-3.5 model")



    #--------------------------------------main--------------------------------------
    class Args:
        def __init__(self, nougat_info, arxiv_info, **kwargs):
            self.checkpoint = kwargs.get("checkpoint", Path(nougat_info["check_point"]))
            self.out = kwargs.get("out", Path(nougat_info["out"]))
            self.recompute = kwargs.get("recompute", True)
            self.markdown = kwargs.get("markdown", True)
            self.pdf = kwargs.get("pdf", [Path(i) for i in nougat_info["pdf"]])
            self.num_process = kwargs.get("num_process", var_num_processes)
            self.kw = kwargs.get("kw", arxiv_info['key_word'])
            self.rate_limit = 3 if var_openai_info['rate_limit'] else None

    args = Args(nougat_info=var_nougat_info,arxiv_info=var_arxiv_info)
    st.header("使用nougat进行arxiv论文摘要demo")
    var_req_url = st.text_input('requesting the url of server:',
                                "http://127.0.0.1:8000",
                                key="request_url",
                                help="request url of server,default: http://61.241.103.32:5010,if local,http://127.0.0.1:8000")
    # url_search_tab, file_upload_tab = st.tabs(["**url搜索**🔍", "**文件上传**📁"])
    var_save_dir = './res/app_res'  # st.text_input('保存路径','./app_res', key="save_dir",help="save dir of result")
    selection = option_menu(None, ["url搜索", "文件上传"],
                            icons=['house', 'cloud-upload'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    # with url_search_tab:
    if selection == "url搜索":
        st.subheader("选择搜索关键词排序方式：")
        var_daily_type = st.selectbox('daily type',
                                     ('cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'eess', 'econ', 'astro-ph', 'cond-mat', 'gr-qc',
                                     'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nucl-ex', 'nucl-th', 'quant-ph'),
                                     help="daily type of arxiv search, when keyword is None, this option will be used")
        col1, col2, col3, col4 = st.columns(4)
        var_show_abstracts = col1.selectbox("显示摘要", ['是', '否'], key="show_abstracts",help="show abstract of arxiv search result")
        var_searchtype = col2.selectbox("选择搜索类型:", ['all', 'title', 'abstract', 'author', 'comment',
                                        'journal_ref', 'subject_class', 'report_num', 'id_list'],
                                        key="searchtype",help="""search type of arxiv search, all->all types, title->title of paper, 
                                                                abstract->abstract of paper, author->author of paper, comment->comment of paper, 
                                                                journal_ref->journal_ref of paper, subject_class->subject_class of paper, 
                                                                report_num->report_num of paper, id_list->id_list of paper""")
        var_order = col3.selectbox("选择排序方式:", ['-announced_date_first',
                                           'submitted_date',
                                           '-submitted_date', 'announced_date_first', ''],
                                      key="order",help="""order of arxiv search result,
                                                            -announced_date_first->announced_date_first of paper,
                                                            submitted_date->submitted_date of paper,
                                                            -submitted_date->-submitted_date of paper,
                                                            announced_date_first->announced_date_first of paper""")

        var_size = col4.selectbox("选择返回结果数量:", [25, 50, 100, 200], key="size",help="size of arxiv search result")
        st.subheader("输入关键词搜索arxiv论文：")

        var_keyword = st.text_input("输入关键字:", "None", key="keyword",help="keyword of arxiv search, e.g. 'quantum computer'")

        result_placeholder = st.empty()
        if var_keyword.lower() == "none":
            var_keyword = None
            result_placeholder.text("没有关键词输入，自动推荐当天新投递论文")
        else:
            result_placeholder.text(f"搜索{var_keyword}中，请等待...")
        var_links,var_titles,var_abstract,var_authors = get_links(var_keyword,
                                                                  var_proxy,
                                                                  max_num=var_max_links,
                                                                  line_length=var_linelength,
                                                                  searchtype=var_searchtype,
                                                                  order=var_order,
                                                                  size=var_size,
                                                                  daily_type=var_daily_type,
                                                                  headers=var_headers)

        if var_links is None:
            result_placeholder.text(f"搜索失败,错误信息：{var_authors},请重试")
            st.stop()

        var_options = [(i,False) for i in range(len(var_links))]

        if var_keyword == "None":
            result_placeholder.text(f"获取到当天新投递的{var_max_links}篇论文(您可以选择多个链接)：")
        else:
            result_placeholder.text(f"搜索{var_keyword}成功，获取{var_max_links}篇相关内容，链接如下(您可以选择多个链接)：")

        # choice_list: store checkbox and abstract text,if not selected,abstract text will be hidden
        choice_list = []
        for i,link in enumerate(var_links):
            selected = st.checkbox(f"{var_titles[i]}", key=f"link{i + 1}",value=False)
            var_options[i] = (i,selected)
            abs_text = st.empty()
            abs_text.markdown(f"{var_authors[i]}<br><br>{var_abstract[i]}",unsafe_allow_html=True)
            choice_list.append([selected,abs_text])
        st.divider()
        selected_record = st.empty()
        var_num_selected = len([i for i,selected in var_options if selected])
        selected_record.text(f"您选择了{var_num_selected}个链接")

        run_button = st.button("运行模型", key="run_model_url",help="run model to generate summary")
        st.divider()

        st.text("您选择了以下链接：")
        var_selected_links = filter_links(var_options)
        logger.info(f"selected links:{var_selected_links}")
        var_variables = {key: value for key, value in globals().items() if key.startswith("var_") and not callable(value)}
        init_session_state()
        st.text(f"st.session_state:{st.session_state}")

        if "var_variables" not in st.session_state:
            st.session_state["var_variables"] = var_variables
        elif st.session_state["var_variables"] != var_variables:
            # key_difference = set(var_variables.keys() ^ st.session_state["var_pdf_variables"].keys()))
            # value_diff = {key: var_variables.get(key, None) for key in key_difference}
            # value_diff |= {key: st.session_state["var_pdf_variables"].get(key, None) for key in key_difference}
            # print("key_difference:", key_difference, "value_diff:", value_diff)
            st.session_state["var_variables"] = var_variables
            st.session_state["run_model"] = False
        else:
            st.session_state["run_model"] = True
        regenrate_options = []
        if  run_button or st.session_state["run_model"] :
            if var_num_selected == 0:
                st.error("您没有选择任何链接，请选择链接后再运行模型")

            # if init below,will reset page0,when init page1
            if run_button:
                init_session_state(url_reset=True)

            progress_text = '正在运行模型...'
            progress_bar = st.progress(0,text=progress_text)
            if st.session_state["run_model"]:
                choice_list = [[x[0],x[1].empty()] for x in choice_list if not x[0]]

            time_record = []
            pdf_record = []
            length_record = []

            for i,index in enumerate(var_selected_links):
                progress_time = []
                args.pdf = [var_links[index]]
                user_input = st.chat_message("user")
                user_input.caption(f"Input Pdf :")
                user_input.markdown(f"{var_titles[index]}")
                now_time = time.time()
                args_dict = vars(args)
                logger.info(f"args_dict:{args_dict}")
                progress_step = ["model predict","get summary"]
                answer_message = st.chat_message("assistant")

                step_bar = stqdm(total=len(progress_step),st_container= answer_message)
                step_bar.set_description(f"processing {var_titles[index]} in step:{progress_step[step_bar.n]}")
                model_result,pdf_name = get_model_predcit(_proxy=var_proxy,_headers=var_headers,**args_dict)

                elapsed_time = step_bar.format_dict['elapsed']
                progress_time.append(elapsed_time)

                step_bar.update(1)
                step_bar.set_description(f"processing {var_titles[index]} in step:{progress_step[step_bar.n]}")
                if model_result is None:
                    st.error(f"模型运行失败，失败信息：{pdf_name}，请重试")
                    st.stop()
                else:
                    model_result, pdf_name = model_result[0], pdf_name[0]
                    parser_titles,parser_authors,parser_affiliations,summaries,re_respnse = get_summary(
                                                                                                        api_key=var_openai_info['api_key'],
                                                                                                        article=model_result,
                                                                                                        file_name= pdf_name,
                                                                                                        init_grid=2,

                                                                                                        proxy = var_proxy,
                                                                                                        requests_per_minute= args.rate_limit,
                                                                                                        ignore_titles=var_ignore_titles,
                                                                                                        summary_prompts = var_prompts['section summary'],
                                                                                                        resummry_prompts = var_prompts["blog summary"],
                                                                                                        base_url=var_base_url,
                                                                                                        gpt_config={
                                                                                                            "max_tokens": var_max_token,
                                                                                                            "temperature": var_temperature,
                                                                                                            "top_p": var_top_p,
                                                                                                            "frequency_penalty": var_frequency_penalty,
                                                                                                            "presence_penalty": var_presence_penalty,
                                                                                                        },
                                                                                                        num_processes=var_num_processes
                                                                                                        )
                    elapsed_time = step_bar.format_dict['elapsed']
                    progress_time.append(elapsed_time - progress_time[-1])
                    step_bar.update(1)
                    step_bar.close()
                    df = pd.DataFrame([progress_time], columns=progress_step,index=["used time"])
                    answer_message.caption(f"Parser Result:")
                    answer_message.dataframe(df,use_container_width=True,column_config=
                    {
                        "model predict": st.column_config.NumberColumn(
                            "used time of model predict",
                            format="%.2f s",
                            ),
                        "get summary": st.column_config.NumberColumn(
                            "used time of get summary",
                            format="%.2f s",
                            ),
                    }
                    )
                    if parser_titles is None:
                        st.error(f"摘要生成失败，失败信息：{re_respnse}，请重试")
                        st.stop()
                    # 显示模型运行时间和结果
                    parser_authors = "Authors:<br>" + parser_authors if parser_authors else var_authors[index]
                    parser_affiliations = parser_affiliations if parser_affiliations else ''

                    if parser_affiliations:
                        left_author,right_affiliation = answer_message.columns(2)
                        left_author.markdown(f"{parser_authors}",unsafe_allow_html=True)
                        right_affiliation.markdown(f"Affiliations:<br>{parser_affiliations}",unsafe_allow_html=True)

                    else:
                        answer_message.markdown(f"{parser_authors}",unsafe_allow_html=True)

                    if run_button:
                        print("run_button cliked,init_session_state(url_reset=True),st.session_state['num_pages']:",st.session_state["num_pages"])
                        st.session_state["generated_summary"][i] = [re_respnse[0]]
                        st.session_state["score"][i] = re_respnse[1]
                    tabs = [f":{color_ls[k]}[res {k}]" for k in range(st.session_state["num_pages"][i]+1)]
                    print("st.session_state['num_pages']:",st.session_state["num_pages"])
                    print('st.session_state["generated_summary"]:',st.session_state["generated_summary"])
                    print('re_respnse:',re_respnse)

                    for page_idx,tab_idx in enumerate(answer_message.tabs(tabs)):
                        response = [st.session_state["generated_summary"][i][page_idx],st.session_state["score"][i]]

                        display_result(tab_idx,
                                       response,
                                       summary_box_height,
                                       pdf_name)

                        enhance_answer = partial(get_enhance_answer,api_key = var_openai_info['api_key'],raw_summary = summaries,
                                                 regenerate_summary = st.session_state["generated_summary"][i][page_idx],
                                                 index = i,
                                                 page_idx = page_idx,
                                                 proxy = var_proxy,requests_per_minute= args.rate_limit,
                                                    summary_prompts = var_prompts['section summary'],
                                                    resummry_prompts = var_prompts["blog summary"],
                                                    base_url=var_base_url,
                                                    gpt_config={
                                                        "max_tokens": var_max_token,
                                                        "temperature": var_temperature,
                                                        "top_p": var_top_p,
                                                        "frequency_penalty": var_frequency_penalty,
                                                        "presence_penalty": var_presence_penalty,
                                                    },
                                                    )



                        rerun_col, blog_col, speech_col = tab_idx.columns(3)
                        rerun_button = rerun_col.button(":repeat: regenerate", #arrows_counterclockwise: regenerate",
                                                        on_click=enhance_answer,
                                                        kwargs={"usage": 'regenerate'},
                                                        help="regenerate summary",
                                                        key=f"rerun_{i}_{page_idx}")
                        blog_button = blog_col.button(":memo: blog",
                                                      on_click=enhance_answer,
                                                      kwargs={"usage": 'blog'},
                                                      help="blog summary",
                                                      key=f"blog_{i}_{page_idx}")
                        speech_button = speech_col.button(":loud_sound: speech",
                                                          on_click = enhance_answer,
                                                          kwargs={"usage": 'speech'},
                                                          help="speech summary",
                                                          key=f"speech_{i}_{page_idx}")

                        usage = st.session_state["usage"][i][page_idx]
                        print("st.session_state['usage']:",st.session_state["usage"])
                        if usage:
                            text = st.session_state["generated_result"][i][page_idx]
                            tab_idx.text_area(f"{usage} Summary:", f"{text.strip()}", height=summary_box_height,
                                                     key=f"{usage}_summary_{i}_{page_idx}")

                            display_download_button(place_holder=tab_idx,
                                                    summaries = st.session_state["generated_summary"][i][page_idx],
                                                    re_respnse = st.session_state["generated_result"][i][page_idx],
                                                    score = st.session_state["score"][i],
                                                    pdf_name = pdf_name,
                                                    usage = usage)
                            if usage == 'speech':
                                temp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                random_num = random.randint(0, 100000)
                                flag,content = text2speech( text=text)
                                if flag:
                                    tab_idx.audio(content, format='audio/mp3')
                                    tab_idx.download_button(label="下载音频", data=content,
                                                             file_name=f"{str(pdf_name)}.mp3",
                                                             key=f"download_{str(pdf_name)}_{str(temp_time)}_{random_num}")
                                else:
                                    tab_idx.text(content)

                        dur = int(time.time() - now_time)
                        tab_idx.write(f"模型运行时间：{dur}")
                        progress_bar.progress((i+1)/var_num_selected,text=progress_text)
                    time_record.append(progress_time)
                    pdf_record.append(var_links[index])
                    st.divider()
                time_df = pd.DataFrame(time_record,columns=progress_step)
                length_record = [get_pdf_length(pdf) for pdf in pdf_record]
            st.subheader("数据统计：")
            st.text(f"选择了{var_num_selected}个链接，运行时间如下：")
            time_df.insert(0,"pdf_length",length_record)
            fig = plot_df_res(time_df)
            st.pyplot(fig)
            time_df.insert(0,"pdf_link",pdf_record)
            st.text(f"pdf数据细节说明：")
            st.dataframe(time_df,column_config=
            {
                "pdf_link": st.column_config.LinkColumn(
                    "pdf link",
                ),
                "pdf_length": st.column_config.NumberColumn(
                    "number of pages",
                    format="%d pages",
                ),
                "model predict": st.column_config.NumberColumn(
                    "used time of model predict",
                    format="%.2f s",
                ),
                "get summary": st.column_config.NumberColumn(
                    "used time of get summary",
                    format="%.2f s",
                ),
            },
                 hide_index=True,
                 use_container_width=True
            )


    else:
    # with file_upload_tab:
        uploaded_file = st.file_uploader(label='上传本地pdf文件(pdf/zip)', accept_multiple_files=True,
                                            type=['pdf', 'zip'], key="upload_file",help="upload pdf/zip file")
        logger.info(f"uploaded_file:{uploaded_file},len:{len(uploaded_file)}")
        tmpdir = var_save_dir
        if uploaded_file is not None:
            # pdf_content_list:[bytes,bytes,...],var_file_names:[str,str,...]
            pdf_content_list, var_file_names = load_file(uploaded_file)
            # remove duplicate pdf
            pdf_info_df = pd.DataFrame({"pdf_name": var_file_names, "pdf_content": pdf_content_list})
            pdf_info_df = pdf_info_df.drop_duplicates(subset=["pdf_name"])
            pdf_content_list, var_file_names = pdf_info_df["pdf_content"].tolist(), pdf_info_df["pdf_name"].tolist()
            dupulicate_pdf = len(uploaded_file) - len(pdf_content_list)
            if dupulicate_pdf > 0:
                st.text(f"检测到{dupulicate_pdf}个重复pdf文件，已自动删除")
            if len(var_file_names) == 0 or any([pdf is None for pdf in var_file_names]):
                st.error(r"请上传pdf/zip文件")

            pdf_button = st.button("确定", key="upload_run_model",help="run model to generate summary")


            #--------------------------------------session state--------------------------------------
            var_variables = {key: value for key, value in globals().items() if
                             key.startswith("var_") and not callable(value)}
            init_session_state()
            if not "var_pdf_variables" in st.session_state:
                st.session_state["var_pdf_variables"] = var_variables
            elif st.session_state["var_pdf_variables"] != var_variables:
                print("not eq")
                # key_difference = set(set(var_variables.keys() ^ st.session_state["var_pdf_variables"].keys()))
                # value_diff = {key: var_variables.get(key, None) for key in key_difference}
                # value_diff |= {key: st.session_state["var_pdf_variables"].get(key, None) for key in key_difference}
                # print("key_difference:", key_difference, "value_diff:", value_diff)

                st.session_state["var_pdf_variables"] = var_variables
                st.session_state["pdf_run_model"] = False
            else:
                st.session_state["pdf_run_model"] = True


            st.text(f"run_model:{st.session_state['run_model']}")

            if pdf_button or st.session_state["pdf_run_model"]:
                if pdf_button:
                    init_session_state(pdf_reset=True)
                num_selected = len(var_file_names)
                st.write(f"您上传了{num_selected}个pdf文件,")
                st.write(f"获取到的pdf:\n{var_file_names}")
                progress_text = '正在运行模型...'
                logger.info(f"pdf_list:{var_file_names}")
                process_bar = st.progress(0, text=progress_text)
                for i, pdf, name in zip(range(num_selected), pdf_content_list, var_file_names):
                    time_list = []
                    args.pdf = [pdf]
                    user_input = st.chat_message("user")
                    user_input.caption(f"input pdf :")
                    user_input.markdown(f"{name}")
                    user_input.download_button(label=f"下载{name}", data=pdf,
                                               file_name=f"{str(name)}", key=f"download_{str(name)}")
                    now_time = time.time()
                    args_dict = vars(args)
                    model_result, pdf_name = get_model_predcit(_proxy=var_proxy, _headers=var_headers, pdf_name=name, **args_dict)
                    model_result, pdf_name = model_result[0], pdf_name[0]
                    titles, authors, affiliations, summaries, re_respnses = get_summary(
                        api_key=var_openai_info['api_key'],
                        article=model_result,
                        file_name=pdf_name,
                        init_grid=2,
                        proxy=var_proxy,
                        requests_per_minute=args.rate_limit,
                        ignore_titles=var_ignore_titles,
                        summary_prompts=var_prompts['section summary'],
                        resummry_prompts=var_prompts["blog summary"],
                        base_url=var_base_url,
                        gpt_config={
                            "max_tokens": var_max_token,
                            "temperature": var_temperature,
                            "top_p": var_top_p,
                            "frequency_penalty": var_frequency_penalty,
                            "presence_penalty": var_presence_penalty,
                        },
                        num_processes=var_num_processes
                    )
                    response_msg = st.chat_message("assistant")
                    response_msg.caption(f"parser result:")
                    if titles is not None:
                        temp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

                        if affiliations:
                            col1, col2 = response_msg.columns(2)
                            col1.text_area(f"Auhors:", f"{authors.strip()}", height=10,key=f"author_{i}_pdf_{temp_time}")
                            col2.text_area(f"Affiliations:", f"{affiliations.strip()}", height=10,key=f"affiliation_{i}_pdf_{temp_time}")
                        else:
                            response_msg.text_area("Author:",f"{authors.strip()}",height=10,key=f"author_{i}_pdf_{temp_time}")

                    else:
                        response_msg.write("due to the parser error, the title is None")


                    if pdf_button:
                        st.session_state["pdf_generated_summary"][i] = [re_respnses[0]]
                        st.session_state["pdf_score"][i] = re_respnses[1]
                    pdf_tabs = [f":{color_ls[k]}[res {k}]" for k in range(st.session_state["pdf_num_pages"][i] + 1)]

                    for pdf_page_idx, pdf_tab_idx in enumerate(response_msg.tabs(pdf_tabs)):
                        response = [st.session_state["pdf_generated_summary"][i][pdf_page_idx],
                                    st.session_state["pdf_score"][i]]
                        display_result(pdf_tab_idx,
                                       response,
                                       summary_box_height,
                                       pdf_name)

                        enhance_answer = partial(get_pdf_enhance, api_key=var_openai_info['api_key'],
                                                 raw_summary=summaries,
                                                 regenerate_summary=st.session_state["pdf_generated_summary"][i][pdf_page_idx],
                                                 index=i,
                                                 page_idx=pdf_page_idx,
                                                 proxy=var_proxy,
                                                 requests_per_minute=args.rate_limit,
                                                 summary_prompts=var_prompts['section summary'],
                                                 resummry_prompts=var_prompts["blog summary"],
                                                 base_url=var_base_url,
                                                 gpt_config={
                                                     "max_tokens": var_max_token,
                                                     "temperature": var_temperature,
                                                     "top_p": var_top_p,
                                                     "frequency_penalty": var_frequency_penalty,
                                                     "presence_penalty": var_presence_penalty,
                                                 },
                                                 )

                        rerun_col, blog_col, speech_col = pdf_tab_idx.columns(3)
                        rerun_button = rerun_col.button(":arrows_counterclockwise: regenerate",
                                                        on_click=enhance_answer,
                                                        kwargs={"usage": 'regenerate'},
                                                        help="regenerate summary",
                                                        key=f"rerun_{i}_pdf_{pdf_page_idx}")
                        blog_button = blog_col.button(":memo: blog", key=f"blog_{i}_pdf_{pdf_page_idx}",
                                                        on_click=enhance_answer,
                                                        kwargs={"usage": 'blog'},
                                                        help="blog summary")
                        speech_button = speech_col.button(":loud_sound: speech", key=f"speech_{i}_pdf_{pdf_page_idx}",
                                                        on_click=enhance_answer,
                                                        kwargs={"usage": 'speech'},
                                                        help="speech summary")
                        usage = st.session_state["pdf_usage"][i][pdf_page_idx]
                        if usage :
                            text = st.session_state["pdf_generated_result"][i][pdf_page_idx]
                            pdf_tab_idx.text_area(f"{usage} Summary:", f"{text.strip()}", height=summary_box_height,
                                         key=f"{usage}_summary_{i}_pdf_{pdf_page_idx}")
                            display_download_button(
                                place_holder=pdf_tab_idx,
                                summaries=st.session_state["pdf_generated_summary"][i][pdf_page_idx],
                                re_respnse=st.session_state["pdf_generated_result"][i][pdf_page_idx],
                                score=st.session_state["pdf_score"][i],
                                pdf_name=pdf_name,
                                usage=usage
                            )
                            if usage == 'speech':
                                temp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                random_num = random.randint(0, 100000)
                                flag,pdf_bytes = text2speech(text=text)
                                if flag:
                                    pdf_tab_idx.audio(pdf_bytes, format='audio/mp3')
                                    pdf_tab_idx.download_button(label="下载音频", data=pdf_bytes,
                                                             file_name=f"{str(pdf_name)}.mp3",
                                                             key=f"download_{str(pdf_name)}_{str(temp_time)}_{random_num}")
                                else:
                                    pdf_tab_idx.text(pdf_bytes)




                    dur = int(time.time() - now_time)
                    response_msg.write(f"模型运行时间：{dur}")
                    process_bar.progress((i + 1) / num_selected, text=progress_text)
                    st.divider()

        else:
            st.error("无法获取文件，请上传文件")