import re
import streamlit as st
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from MMAPIS.tools import *
from MMAPIS.config.config import LOGGER_MODES,OPENAI_CONFIG,ARXIV_CONFIG,NOUGAT_CONFIG,GENERAL_CONFIG,PROMPTS,SECTION_PROMPTS,INTEGRATE_PROMPTS,APPLICATION_PROMPTS,ALIGNMENT_CONFIG,TTS_CONFIG
from typing import Dict, List, Literal, Optional, Union
import logging
from pathlib import Path
import requests
import json
import pandas as pd
from datetime import datetime
import random
from functools import partial
from streamlit_option_menu import option_menu
from stqdm import stqdm
from matplotlib import pyplot as plt
import webbrowser


@st.cache_data
def init_logger():
    logger = init_logging(logger_mode=LOGGER_MODES)
    return logger


@st.cache_data
def filter_links(links):
    selected_index = [index for index, selected in links if selected]
    return selected_index

@st.cache_data
def load_file(uploaded_file):
    return  get_pdf_list(uploaded_file)



class Args:
    def __init__(self,
                 markdown: Optional[bool] = True,
                 pdf: Optional[List[Union[str,bytes,Path]]] = None,
                 pdf_name: Optional[Union[str,Path]] = None):
        self.markdown = markdown
        self.pdf = pdf
        self.pdf_name = pdf_name


def init_session_state(url_reset:bool = False,
                       pdf_reset:bool = False):
    if "run_model" not in st.session_state :
        st.session_state["run_model"] = False


    if "generated_summary" not in st.session_state or url_reset:
        st.session_state["generated_summary"] = [[None] * max_page_num for _ in range(max_pdf_num)]
        st.session_state["score"] = [None] * max_pdf_num

    if "generated_result" not in st.session_state or url_reset:
        st.session_state["generated_result"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "usage" not in st.session_state or url_reset:
        st.session_state["usage"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "num_pages" not in st.session_state or url_reset:
        st.session_state["num_pages"] = [0] * max_pdf_num


    if "pdf_run_model" not in st.session_state:
        st.session_state["pdf_run_model"] = False

    if "pdf_generated_summary" not in st.session_state or pdf_reset:
        st.session_state["pdf_generated_summary"] = [[None] * max_page_num for _ in range(max_pdf_num)]
        st.session_state["pdf_score"] = [None] * max_pdf_num

    if "pdf_generated_result" not in st.session_state or pdf_reset:
        st.session_state["pdf_generated_result"] = [[None] * max_page_num for _ in range(max_pdf_num)]

    if "pdf_usage" not in st.session_state or pdf_reset:
        st.session_state["pdf_usage"] = [[None] * max_pdf_num for _ in range(max_pdf_num)]



    if "pdf_num_pages" not in st.session_state or pdf_reset:
        st.session_state["pdf_num_pages"] = [0] * max_pdf_num




@st.cache_data
def get_links(keyword,
              max_num=10,
              line_length=15,
              searchtype='all',
              show_abstract='show',
              order='-announced_date_first',
              size=50,
              daily_type = 'cs',
              markdown=True,
              ):
    url = var_req_url + '/get_links/'
    params = {
        "key_word": keyword,
        "searchtype": searchtype,
        "abstracts": show_abstract,
        "order": order,
        "size": size,
        "max_return": max_num,
        "line_length": line_length,
        "return_md": markdown,
        "daily_type": daily_type
    }
    logging.info(f"get links params:\n{params}")
    response = requests.post(url, json=params)
    json_info = custom_response_handler(response,func_name='get_links')
    if json_info["status"] == "success":
        links = []
        titles = []
        abstracts = []
        authors = []
        for item in json_info["message"]:
            links.append(item["pdf_url"])
            titles.append(item["title"])
            abstracts.append(item["abstract"])
            authors.append(item["author"])
        return links, titles, abstracts, authors
    else:
        return None,None,None,json_info.get('message',"error")



@st.cache_data
def get_model_predcit(pdf_content:List[bytes]=None,**kwargs):
    nougat_url = var_req_url + '/pdf2md/'
    _args = kwargs
    if pdf_content:
        data = {
            "pdf_name": _args['pdf_name'],
            "markdown": _args['markdown'],
        }
        if not isinstance(pdf_content,List):
            pdf_content = [pdf_content]
        files = []
        for i in range(len(pdf_content)):
            files.append(('pdf_content', bytes2io(pdf_content[i])))
        data = dict_filter_none(data)
        response = requests.post(nougat_url, files=files, data=data)

    else:
        _args['pdf'] = _args['pdf'] if isinstance(_args['pdf'], list) else [_args['pdf']]
        _args['pdf_name'] = _args['pdf_name'] if isinstance(_args['pdf_name'], list) else [_args['pdf_name']]
        _args['pdf'] = [str(i) for i in _args['pdf']]
        params = {
            "pdf": _args['pdf'],
            "pdf_name": _args['pdf_name'],
            "markdown": _args['markdown'],
        }
        params = dict_filter_none(params)
        response = requests.post(nougat_url, data=params)
    json_info = custom_response_handler(response,func_name='model_predict')
    if json_info["status"] == "success":
        file_names = []
        article_ls = []
        for item in json_info["message"]:
            file_names.append(item["file_name"])
            article_ls.append(item["text"])
        return article_ls,file_names
    else:
        return None,json_info.get('message',"error")




@st.cache_data
def get_document_summary(
                        api_key:str,
                        base_url:str,
                        article:str,
                        file_name:str=None,
                        init_grid:int = 2,
                        max_grid:int = 4,
                        summary_prompts:Dict = None,
                        integrate_prompts:Dict = None,
                        summarizer_params:Dict = None,
                        pdf:Optional[Union[str,bytes,Path]] = None,
                        pdf_content:Optional[bytes] = None,
                        img_width:int = 600,
                        threshold:float = 0.8,
                        ):


    url = var_req_url + '/app/document_level_summary/'
    params = {
        "api_key": api_key,
        "base_url": base_url,
        "article": article,
        "pdf": pdf,
        "file_name": file_name,
        "init_grid": init_grid,
        "max_grid": max_grid,
        "summary_prompts": json.dumps(summary_prompts),
        "integrate_prompts": json.dumps(integrate_prompts),
        "img_width": img_width,
        "threshold": threshold,
        "summarizer_params": json.dumps(summarizer_params)
    }
    params = dict_filter_none(params)

    if pdf_content:
        pdf_content = bytes2io(pdf_content)
        files = [('pdf_content', pdf_content)]
        response = requests.post(url, files=files, data=params)
    else:
        response = requests.post(url, data=params)
    json_info = custom_response_handler(response,func_name='alignment')
    if json_info["status"] == "success":
        data = json_info["message"]
        document_level_summary = data["document_level_summary"]
        section_level_summary = data["section_level_summary"]
        document_level_summary_aligned = data["document_level_summary_aligned"]
        return document_level_summary, section_level_summary,document_level_summary_aligned
    else:
        return None, None,json_info.get('message',"error")


def get_enhance_answer(
                   api_key:str,
                   base_url:str,
                   document_level_summary:str,
                   index: int,
                   page_idx:int,
                   usage: Literal['blog', 'speech', 'regenerate', 'recommend', 'qa'] = 'regenerate',
                   raw_md_text:str=None,
                   section_summary:str = None,
                   prompts:Dict = None,
                   pdf:Optional[Union[str,bytes,None]] = None,
                   init_grid:int = ALIGNMENT_CONFIG['init_grid'],
                   max_grid:int = ALIGNMENT_CONFIG['max_grid'],
                   threshold:float = ALIGNMENT_CONFIG['threshold'],
                   tts_api_key:str = TTS_CONFIG['api_key'],
                   tts_base_url:str = TTS_CONFIG['base_url'],
                   app_secret:str = TTS_CONFIG['app_secret'],
                   summarizer_params:Dict = None,
                   pdf_content:Optional[bytes] = None,
                   url_mode:bool = True,
                   **kwargs):
    if not usage in ['blog', 'speech', 'regenerate','recommend','qa']:
        raise ValueError(f"usage must be in ['blog','speech','regenerate','recommend','qa'],but got {usage}")
    app_url = var_req_url + '/app/'
    params = {
        "api_key": api_key,
        "base_url": base_url,
        "document_level_summary": document_level_summary,
        "usage": usage,
        "section_summary": section_summary,
        "raw_md_text": raw_md_text,
        "prompts": json.dumps(prompts),
        "pdf": pdf,
        "init_grid": init_grid,
        "max_grid": max_grid,
        "img_width": 600,
        "threshold": threshold,
        "tts_api_key": tts_api_key,
        "tts_base_url": tts_base_url,
        "app_secret": app_secret,
        "summarizer_params": json.dumps(summarizer_params),
    }
    params = dict_filter_none(params)
    if pdf_content is None:
        response = requests.post(app_url, data=params)
    else:
        pdf_content = bytes2io(pdf_content)
        files = [('pdf_content', pdf_content)]
        response = requests.post(app_url, files=files, data=params)
    if response.status_code == 200:
        webbrowser.open(response.url)
    else:
        logger.error(f"get enhance answer error:{response.text}")
        st.error(f"get enhance answer error:{response.text}")

    st.session_state["usage"][index][page_idx] = usage
    if usage == 'regenerate':
        pattern = re.compile(r'<script id="article" type="text/plain" style="display: none;">(.*?)</script>', re.DOTALL)
        text = pattern.search(response.text)
        if text:
            text = text.group(1)
        else:
            text = response.text
        if url_mode:
            st.session_state["num_pages"][index] += 1
            st.session_state["generated_summary"][index].append(text.strip())
        else:
            st.session_state["pdf_num_pages"][index] += 1
            st.session_state["pdf_generated_summary"][index].append(text.strip())


@st.cache_data
def get_pdf_length(pdf:Union[str,bytes,Path]):
    assert isinstance(pdf,(str,bytes,Path)),f"pdf must be str,bytes or Path,but got {type(pdf)}"
    return get_pdf_doc(pdf)[1]


# def plot_df_res(df):
#     fig, ax = plt.subplots(figsize = (10,6))
#     df = df.sort_values(by="pdf_length")
#     l1, = ax.plot(df["pdf_length"], df["get summary"], color='red', linestyle='--', label='get summary')
#     l2, = ax.plot(df["pdf_length"], df["model predict"], color='blue', linestyle='--', label='model predict')
#     markers = ['o', 'x', 's', 'v', 'D', 'p', 'P', 'h', '8', 'd', 'H', '*', 'X']
#     lengens = []
#     for index, row in df.iterrows():
#         # color = plt.cm.viridis(index / len(df))
#         a,b,c = row["pdf_length"],row["get summary"],row["model predict"]
#         ax.scatter([a],[b],color="red",marker=markers[index])
#         ax.scatter([a],[c],color="blue",marker=markers[index])
#         # cur_marker = mlines.Line2D([], [], color='black', marker='o', markersize=10, label=f"pdf_{index}")
#         cur_marker = plt.scatter([], [], color='black', marker=markers[index], label=f"pdf_{index}")
#         lengens.append(cur_marker)
#     lengens.extend([l1,l2])
#     ax.set_title('model predict & get sumamry time cost with different pdf length')
#     ax.set_xlabel('length of pdf(page)')
#     ax.set_ylabel('time cost(s)')
#     ax.legend(handles=lengens, loc="best")
#     return fig


if __name__ == "__main__":
    #--------------------------------------init--------------------------------------
    max_pdf_num = 25
    max_page_num = 10
    color_ls = ["blue", "green", "orange", "red", "violet", "gray", "rainbow"]

    # Streamlit应用程序标题
    st.set_page_config(page_title="paper parser demo", page_icon=":smiley:", layout="wide", initial_sidebar_state="auto")
    logger = init_logger()

    # var_openai_info = OPENAI_CONFIG
    # var_arxiv_info = ARXIV_CONFIG
    # var_nougat_info = NOUGAT_CONFIG
    # var_prompts = PROMPTS
    # var_base_url = var_openai_info['base_url']
    # var_ignore_titles = var_openai_info['ignore_title']




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

    # display options
    var_linelength = st.sidebar.slider("line length",10,40,18,help="line length of arxiv search result in each line")
    var_max_links = st.sidebar.slider("max links", 0, max_pdf_num, 5,help="max links of arxiv search result")
    var_img_width = st.sidebar.slider("img width", 100, 1000, 600, help="img width of img in alignment result")
    var_api_key = st.sidebar.text_input("openai api key", key="api_key",help="your openai api key, e.g. sk-xxxxxx")
    var_base_url = st.sidebar.text_input("openai base url", key="base_url",
                                         value="https://api.openai.com/v1",
                                         help="your openai base url, e.g. https://api.openai.com/v1")

    var_summary_prompt = None
    var_integrate_prompt = None
    with st.sidebar.expander("prompts optional"):
        var_prompt_ratio = st.slider("prompt ratio", 0.0, 1.0, 0.8,help="prompt ratio of gpt-3.5 model, e.g. 0.8 means (up to)80% prompt and 20% response content")
        var_summary_prompt = st.file_uploader("summary prompt", type=["json"], key="summary_prompt",help = "summary prompt json file,e.g.{'system':'','general_summary':''}")
        var_integrate_prompt = st.file_uploader("integrate prompt",
                                                type=["json"],
                                                key="integrate_prompt",
                                                help = "integrate prompt json file,e.g.{'integrate_system':'','integrate':'','integrate_input':''}")
        var_summary_prompt = json.loads(var_summary_prompt.read()) if var_summary_prompt else None
        var_integrate_prompt = json.loads(var_integrate_prompt.read()) if var_integrate_prompt else None


    with st.sidebar.expander("Model config Options"):
        var_num_processes = st.slider("num processes", 0, 15, 3, help="num processes of gpt-3.5 model, if your api key is not limited, you can set it to 0, else set it to 3 if you're limited by 3 requests per minute")
        var_max_token = st.slider("max token", 0, 16385, 16385,help="max token of total tokens(including prompt tokens) of gpt-3.5 model")
        var_temperature = st.slider("temperature", 0.0, 2.0, 0.5,help="temperature of gpt-3.5 model")
        var_top_p = st.slider("top p", 0.0, 1.0, 1.0,help="top p of gpt-3.5 model")
        var_frequency_penalty = st.slider("frequency penalty", -2.0, 2.0, 0.1, help="frequency penalty of gpt-3.5 model")
        var_presence_penalty = st.slider("presence penalty", -2.0, 2.0, 0.2, help="presence penalty of gpt-3.5 model")

    with st.sidebar.expander("tts config Options"):
        var_tts_api_key = st.text_input("tts api key", key="tts_api_key",help="your tts api key")
        var_tts_base_url = st.text_input("tts base url", key="tts_base_url",help="your tts base url")
        var_app_secret = st.text_input("app secret", key="app_secret",help="your app secret")

    with st.sidebar.expander("alignment config Options"):
        var_threshold = st.slider("threshold", 0.0, 1.0, 0.8,help="threshold of title-like keyword in similarity")
        var_ignore_titles = ["reference","appendix","acknowledg"]



    #--------------------------------------main--------------------------------------

    args = Args()
    st.header("MMAPIS: Multi-Modal Automated Academic Papers Interpretation System Demo")
    var_req_url = st.text_input('requesting the url of server:',
                                "http://127.0.0.1:8000",
                                key="request_url",
                                help="request url of server,default your localhost, i.e. ,http://127.0.0.1:8000")
    selection = option_menu(None, ["search with url", "upload pdf"],
                            icons=['house', 'cloud-upload'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    if not var_api_key:
        st.error("please input your openai api key")
        st.stop()
    # with url_search_tab:
    if selection == "search with url":
        st.subheader("Choose search options:")
        var_daily_type = st.selectbox('daily type',
                                     ('cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'eess', 'econ', 'astro-ph', 'cond-mat', 'gr-qc',
                                     'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nucl-ex', 'nucl-th', 'quant-ph'),
                                     help="daily type of arxiv search, when keyword is None, this option will be used")
        col1, col2, col3, col4 = st.columns(4)
        var_show_abstracts = col1.selectbox("show abstracts", ['yes', 'no'],
                                            key="show_abstracts",help="show abstract of arxiv search result")
        var_searchtype = col2.selectbox("choose search type",
                                        ['all', 'title', 'abstract', 'author', 'comment',
                                        'journal_ref', 'subject_class', 'report_num', 'id_list'],
                                        key="searchtype",help="""search type of arxiv search, all->all types, title->title of paper, 
                                                                abstract->abstract of paper, author->author of paper, comment->comment of paper, 
                                                                journal_ref->journal_ref of paper, subject_class->subject_class of paper, 
                                                                report_num->report_num of paper, id_list->id_list of paper""")
        var_order = col3.selectbox("choose search order",
                                   ['-announced_date_first','submitted_date',
                                    '-submitted_date', 'announced_date_first', ''],
                                      key="order",help="""order of arxiv search result,
                                                            -announced_date_first->  Employing this parameter prioritizes search results by the descending order of the earliest announcement date of the papers, arranging the most recently announced papers at the forefront of the search results.
                                                            submitted_date-> Using this value sorts the search results in ascending order based on the date of paper submission. Consequently, papers submitted earliest will be sequenced at the beginning.
                                                            -submitted_date-> This value organizes the research findings in descending order according to submission date. Therefore, the most recent submissions will take precedence in the ordering.
                                                            announced_date_first-> Execution of this key arranges search results based on ascending order of the earliest announcement date, placing the earliest announced papers first.
                                                            """)

        var_size = col4.selectbox("seach size",
                                  [25, 50, 100, 200], key="size",help="size of arxiv search result")
        st.text("input keyword to search specific field in arxiv papers")

        var_keyword = st.text_input("keyword of arxiv search",
                                    "None",
                                    key="keyword",
                                    help="keyword of arxiv search, e.g. 'quantum computer'")

        result_placeholder = st.empty()
        if var_keyword.lower() == "none":
            var_keyword = None
            result_placeholder.text("No keyword input, automatically recommend papers delivered today")
        else:
            result_placeholder.text(f"searching {var_keyword}...")
        var_links,var_titles,var_abstract,var_authors = get_links(var_keyword,
                                                                  max_num=var_max_links,
                                                                  line_length=var_linelength,
                                                                  searchtype=var_searchtype,
                                                                  order=var_order,
                                                                  size=var_size,
                                                                  daily_type=var_daily_type,
                                                                  markdown=True,
                                                                  )

        if var_links is None:
            result_placeholder.text(f"sorry,search failed,error message:{var_authors},please retry")
            st.stop()

        var_options = [(i,False) for i in range(len(var_links))]

        if var_keyword == "None":
            result_placeholder.text(f"Total {var_max_links} papers delivered today, please choose the links you want to parse(You can choose multiple links):")
        else:
            result_placeholder.text(f"search {var_keyword} successfully, get {var_max_links} related contents, links as follows(You can choose multiple links):")

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
        selected_record.text(f"You have selected {var_num_selected} links")

        run_button = st.button("run model",
                               key="run_model_url",help="run model to generate summary")
        st.divider()

        st.text("You have selected the following links:")
        var_selected_links = filter_links(var_options)
        logger.info(f"selected links:{var_selected_links}")
        var_variables = {key: value for key, value in globals().items() if key.startswith("var_") and not callable(value)}
        init_session_state()

        if "var_variables" not in st.session_state:
            st.session_state["var_variables"] = var_variables
        elif st.session_state["var_variables"] != var_variables:
            # key_difference = set(var_variables.keys() ^ st.session_state["var_variables"].keys())
            # value_diff = {key: var_variables.get(key, None) for key in key_difference}
            # value_diff |= {key: st.session_state["var_variables"].get(key, None) for key in key_difference}
            # print("key_difference:", key_difference, "value_diff:", value_diff)
            st.session_state["var_variables"] = var_variables
            st.session_state["run_model"] = False
        else:
            st.session_state["run_model"] = True
        regenrate_options = []
        if  run_button or st.session_state["run_model"] :
            if var_num_selected == 0:
                st.error("You haven't selected any links, please select links before running the model")

            # if init below,will reset page tab 0,when init page tab 1
            if run_button:
                init_session_state(url_reset=True)

            progress_text = 'progressing...'
            progress_bar = st.progress(0,text=progress_text)
            if st.session_state["run_model"]:
                choice_list = [[x[0],x[1].empty()] for x in choice_list if not x[0]]

            time_record = []
            pdf_record = []
            length_record = []
            progress_step = ["model predict", "get summary"]

            for i,index in enumerate(var_selected_links):
                progress_time = []
                args.pdf = [var_links[index]]
                user_input = st.chat_message("user")
                user_input.caption(f"Input Pdf :")
                user_input.markdown(f"{var_titles[index]}")
                now_time = time.time()
                answer_message = st.chat_message("assistant")
                step_bar = stqdm(total=len(progress_step),st_container= answer_message)
                step_bar.set_description(f"processing {var_titles[index]} in step:{progress_step[step_bar.n]}")
                model_result,pdf_name = get_model_predcit(pdf_content=None,
                                                          **vars(args)
                                                          )

                elapsed_time = step_bar.format_dict['elapsed']
                progress_time.append(elapsed_time)

                step_bar.update(1)
                step_bar.set_description(f"processing {var_titles[index]} in step:{progress_step[step_bar.n]}")
                if model_result is None:
                    st.error(f"Mdoel run failed, error message:{pdf_name},please retry")
                    st.stop()
                else:
                    model_result, pdf_name = model_result[0], pdf_name[0]
                    document_summary, section_summary, document_summary_aligned = get_document_summary(
                                                                                            api_key=var_api_key,
                                                                                            base_url=var_base_url,
                                                                                            article=model_result,
                                                                                            file_name= pdf_name,
                                                                                            init_grid=3,
                                                                                            max_grid=4,
                                                                                            summary_prompts=var_summary_prompt,
                                                                                            integrate_prompts=var_integrate_prompt,
                                                                                            pdf=args.pdf[0],
                                                                                            img_width=var_img_width,
                                                                                            threshold=0.8,
                                                                                            summarizer_params=
                                                                                            {
                                                                                                "rpm_limit": var_num_processes,
                                                                                                "ignore_titles":var_ignore_titles,
                                                                                                "prompt_ratio":var_prompt_ratio,
                                                                                                "gpt_model_params":
                                                                                                    {
                                                                                                        "max_tokens": var_max_token,
                                                                                                        "temperature": var_temperature,
                                                                                                        "top_p": var_top_p,
                                                                                                        "frequency_penalty": var_frequency_penalty,
                                                                                                        "presence_penalty": var_presence_penalty,
                                                                                                    }
                                                                                            }
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
                    if not document_summary:
                        st.error(f"Abstract generation failed, error message:{document_summary_aligned},please retry")
                        st.stop()

                    if run_button:
                        st.session_state["generated_summary"][i] = [document_summary_aligned]
                    tabs = [f":{color_ls[k]}[generation {k}]" for k in range(st.session_state["num_pages"][i]+1)]

                    for page_idx,tab_idx in enumerate(answer_message.tabs(tabs)):
                        tab_idx.markdown(st.session_state["generated_summary"][i][page_idx],unsafe_allow_html=True)
                        enhance_answer = partial(get_enhance_answer,
                                                 api_key=var_api_key,
                                                 base_url=var_base_url,
                                                 section_summary=section_summary,
                                                 index=i,
                                                 page_idx=page_idx,
                                                 raw_md_text=model_result,
                                                 pdf=args.pdf[0],
                                                 init_grid=3,
                                                 max_grid=4,
                                                 threshold=0.8,
                                                 summarizer_params={
                                                     "rpm_limit": OPENAI_CONFIG["rpm_limit"],
                                                     "ignore_titles": var_ignore_titles,
                                                     "num_processes": var_num_processes,
                                                     "prompt_ratio": var_prompt_ratio,
                                                     "gpt_model_params":
                                                         {
                                                             "max_tokens": var_max_token,
                                                             "temperature": var_temperature,
                                                             "top_p": var_top_p,
                                                             "frequency_penalty": var_frequency_penalty,
                                                             "presence_penalty": var_presence_penalty,
                                                         }
                                                 },
                                                )



                        rerun_col, blog_col, speech_col,recommend_col,qa_col = tab_idx.columns(5)
                        rerun_button = rerun_col.button(":repeat: regenerate",
                                                        on_click=enhance_answer,
                                                        kwargs={"usage": 'regenerate',
                                                                "prompts": APPLICATION_PROMPTS["regenerate_prompts"],
                                                                "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                                },
                                                        help="regenerate summary",
                                                        key=f"rerun_{i}_{page_idx}")
                        blog_button = blog_col.button(":memo: blog",
                                                      on_click=enhance_answer,
                                                      kwargs={"usage": 'blog',
                                                              "prompts": APPLICATION_PROMPTS["blog_prompts"],
                                                              "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                              },
                                                      help="blog summary",
                                                      key=f"blog_{i}_{page_idx}")
                        speech_button = speech_col.button(":loud_sound: speech",
                                                          on_click = enhance_answer,
                                                          kwargs={"usage": 'speech',
                                                                  "prompts": APPLICATION_PROMPTS["broadcast_prompts"],
                                                                  "tts_api_key": TTS_CONFIG['api_key'],
                                                                    "tts_base_url": TTS_CONFIG['base_url'],
                                                                    "app_secret": TTS_CONFIG['app_secret'],
                                                                    "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                                                                    },
                                                          help="speech summary",
                                                          key=f"speech_{i}_{page_idx}")
                        recommend_button = recommend_col.button(":loud_sound: recommend",
                                                            on_click = enhance_answer,
                                                            kwargs={"usage": 'recommend',
                                                                    "prompts": APPLICATION_PROMPTS["score_prompts"],
                                                                    "document_level_summary": clean_img(st.session_state["generated_summary"][i][page_idx])
                        },
                                                            help="recommend summary",
                                                            key=f"recommend_{i}_{page_idx}")
                        qa_button = qa_col.button(":question: qa",
                                                    on_click = enhance_answer,
                                                    kwargs={"usage": 'qa',
                                                            "prompts": APPLICATION_PROMPTS["multimodal_qa"],
                                                            "document_level_summary": st.session_state["generated_summary"][i][page_idx]
                        },
                                                    help="qa summary",
                                                    key=f"qa_{i}_{page_idx}")


                        dur = int(time.time() - now_time)
                        tab_idx.write(f"Mdoel run time:{dur} s")
                        progress_bar.progress((i+1)/var_num_selected,text=progress_text)
                    time_record.append(progress_time)
                    pdf_record.append(var_links[index])
                    st.divider()
                time_df = pd.DataFrame(time_record,columns=progress_step)
                length_record = [get_pdf_length(pdf) for pdf in pdf_record]
            st.subheader("Running Result:")
            st.text(f"You have selected {var_num_selected} links, the running time is as follows:")
            if var_num_selected == 0:
                time_df = pd.DataFrame(columns=progress_step)
            time_df.insert(0,"pdf_length",length_record)
            # fig = plot_df_res(time_df)
            # st.pyplot(fig)
            time_df.insert(0, "pdf_link", pdf_record)
            st.text(f"running time details:")
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
        uploaded_file = st.file_uploader(label='Upload local pdf file(pdf/zip)',
                                         accept_multiple_files=True,
                                        type=['pdf', 'zip'], key="upload_file",help="upload pdf/zip file")
        logger.info(f"uploaded_file:{uploaded_file},len:{len(uploaded_file)}")
        if uploaded_file is not None:
            # pdf_content_list:[bytes,bytes,...],var_file_names:[str,str,...]
            pdf_content_list, var_file_names = load_file(uploaded_file)
            # remove duplicate pdf
            pdf_info_df = pd.DataFrame({"pdf_name": var_file_names, "pdf_content": pdf_content_list})
            pdf_info_df = pdf_info_df.drop_duplicates(subset=["pdf_name"])
            pdf_content_list, var_file_names = pdf_info_df["pdf_content"].tolist(), pdf_info_df["pdf_name"].tolist()
            dupulicate_pdf = len(uploaded_file) - len(pdf_content_list)
            if dupulicate_pdf > 0:
                st.text(f"Find {dupulicate_pdf} duplicate pdf files and automatically delete them")
            if len(var_file_names) == 0 or any([pdf is None for pdf in var_file_names]):
                st.error(r"Plese upload pdf/zip file")

            pdf_button = st.button("run model",
                                   key="upload_run_model",help="run model to generate summary")


            #--------------------------------------session state--------------------------------------
            var_variables = {key: value for key, value in globals().items() if
                             key.startswith("var_") and not callable(value)}
            init_session_state()
            if not "var_pdf_variables" in st.session_state:
                st.session_state["var_pdf_variables"] = var_variables
            elif st.session_state["var_pdf_variables"] != var_variables:
                # key_difference = set(set(var_variables.keys() ^ st.session_state["var_pdf_variables"].keys()))
                # value_diff = {key: var_variables.get(key, None) for key in key_difference}
                # value_diff |= {key: st.session_state["var_pdf_variables"].get(key, None) for key in key_difference}
                # print("key_difference:", key_difference, "value_diff:", value_diff)

                st.session_state["var_pdf_variables"] = var_variables
                st.session_state["pdf_run_model"] = False
            else:
                st.session_state["pdf_run_model"] = True

            if pdf_button or st.session_state["pdf_run_model"]:
                if pdf_button:
                    init_session_state(pdf_reset=True)
                num_selected = len(var_file_names)
                st.write(f"You have uploaded {num_selected} pdf files, details as follows:\n{var_file_names}")
                progress_text = 'progressing...'
                logger.debug(f"pdf_list:{var_file_names}")
                process_bar = st.progress(0, text=progress_text)
                for i, pdf, name in zip(range(num_selected), pdf_content_list, var_file_names):
                    progress_time_list = []
                    progress_step = ["model predict", "get summary"]
                    args.pdf_name = name
                    user_input = st.chat_message("user")
                    user_input.caption(f"input pdf :")
                    user_input.markdown(f"{name}")
                    user_input.download_button(label=f"Download {name}", data=pdf,
                                               file_name=f"{str(name)}", key=f"download_{str(name)}")
                    response_msg = st.chat_message("assistant")
                    step_bar = stqdm(total=len(progress_step), st_container=response_msg)
                    model_result, pdf_name = get_model_predcit(pdf_content=pdf, **vars(args))
                    elapsed_time = step_bar.format_dict['elapsed']
                    progress_time_list.append(elapsed_time)
                    step_bar.update(1)
                    step_bar.set_description(f"processing {name} in step:{progress_step[step_bar.n]}")
                    if model_result is None:
                        st.error(f"Processing {name} failed, error message:{pdf_name},please retry")
                        st.stop()
                    model_result, pdf_name = model_result[0], pdf_name[0]
                    document_summary, section_summary, document_summary_aligned  = get_document_summary(
                        api_key=var_api_key,
                        base_url=var_base_url,
                        article=model_result,
                        file_name=pdf_name,
                        init_grid=3,
                        max_grid=4,
                        summary_prompts=var_summary_prompt,
                        integrate_prompts=var_integrate_prompt,
                        pdf_content=pdf,
                        img_width=var_img_width,
                        threshold=var_threshold,
                        summarizer_params=
                        {
                            "rpm_limit": var_num_processes,
                            "ignore_titles": var_ignore_titles,
                            "prompt_ratio": var_prompt_ratio,
                            "gpt_model_params":
                                {
                                    "max_tokens": var_max_token,
                                    "temperature": var_temperature,
                                    "top_p": var_top_p,
                                    "frequency_penalty": var_frequency_penalty,
                                    "presence_penalty": var_presence_penalty,
                                }
                        }
                    )
                    elapsed_time = step_bar.format_dict['elapsed']
                    progress_time_list.append(elapsed_time - progress_time_list[-1])
                    step_bar.update(1)
                    step_bar.close()
                    df = pd.DataFrame([progress_time_list], columns=progress_step, index=["used time"])

                    response_msg.caption(f"parser result:")
                    response_msg.dataframe(df, use_container_width=True, column_config=
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
                    if document_summary is None:
                        response_msg.error(f"Abstract generation failed, error message:{document_summary_aligned},please retry")
                        st.stop()
                    if pdf_button:
                        st.session_state["pdf_generated_summary"][i] = [document_summary_aligned]
                    pdf_tabs = [f":{color_ls[k]}[res {k}]" for k in range(st.session_state["pdf_num_pages"][i] + 1)]

                    for pdf_page_idx, pdf_tab_idx in enumerate(response_msg.tabs(pdf_tabs)):
                        pdf_tab_idx.markdown(st.session_state["pdf_generated_summary"][i][pdf_page_idx],unsafe_allow_html=True)
                        enhance_answer = partial(get_enhance_answer,
                                                    api_key=var_api_key,
                                                    base_url=var_base_url,
                                                    section_summary=section_summary,
                                                    index=i,
                                                    page_idx=pdf_page_idx,
                                                    raw_md_text=model_result,
                                                    pdf_content=pdf,
                                                    init_grid=3,
                                                    max_grid=4,
                                                    threshold=var_threshold,
                                                    url_mode = False,
                                                    summarizer_params={
                                                        "rpm_limit": var_num_processes,
                                                        "ignore_titles": var_ignore_titles,
                                                        "prompt_ratio": var_prompt_ratio,
                                                        "gpt_model_params":
                                                            {
                                                                "max_tokens": var_max_token,
                                                                "temperature": var_temperature,
                                                                "top_p": var_top_p,
                                                                "frequency_penalty": var_frequency_penalty,
                                                                "presence_penalty": var_presence_penalty,
                                                            }
                                                    },
                                                        )

                        rerun_col, blog_col, speech_col, recommend_col,qa_col = pdf_tab_idx.columns(5)
                        rerun_button = rerun_col.button(":arrows_counterclockwise: regenerate",
                                                        on_click=enhance_answer,
                                                        kwargs={
                                                            "usage": 'regenerate',
                                                            "prompts": APPLICATION_PROMPTS["regenerate_prompts"],
                                                            "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                        },
                                                        help="regenerate summary",
                                                        key=f"rerun_{i}_pdf_{pdf_page_idx}")
                        blog_button = blog_col.button(":memo: blog", key=f"blog_{i}_pdf_{pdf_page_idx}",
                                                        on_click=enhance_answer,
                                                        kwargs={
                                                            "usage": 'blog',
                                                            "prompts": APPLICATION_PROMPTS["blog_prompts"],
                                                            "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                        },
                                                        help="blog summary")
                        speech_button = speech_col.button(":loud_sound: speech",
                                                          key=f"speech_{i}_pdf_{pdf_page_idx}",
                                                          on_click=enhance_answer,
                                                          kwargs={"usage": 'speech',
                                                                    "prompts": APPLICATION_PROMPTS["broadcast_prompts"],
                                                                  "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                  },
                                                          help="speech summary")
                        recommend_button = recommend_col.button(":loud_sound: recommend",
                                                            key=f"recommend_{i}_pdf_{pdf_page_idx}",
                                                            on_click=enhance_answer,
                                                            kwargs={"usage": 'recommend',
                                                                    "prompts": APPLICATION_PROMPTS["score_prompts"],
                                                                    "document_level_summary":clean_img(st.session_state["pdf_generated_summary"][i][pdf_page_idx]),
                                                                    },
                                                            help="recommend summary")
                        qa_button = qa_col.button(":question: qa",
                                                    key=f"qa_{i}_pdf_{pdf_page_idx}",
                                                    on_click=enhance_answer,
                                                    kwargs={"usage": 'qa',
                                                            "prompts": APPLICATION_PROMPTS["multimodal_qa"],
                                                            "document_level_summary":st.session_state["pdf_generated_summary"][i][pdf_page_idx],
                                                            },
                                                    help="qa summary")


                    process_bar.progress((i + 1) / num_selected, text=progress_text)
                    st.divider()
        else:
            st.error("Unable to get file, please upload file")





