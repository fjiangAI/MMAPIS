import re
import streamlit as st
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from MMAPIS.tools import init_logging, custom_response_handler, dict_filter_none, bytes2io,get_pdf_list, get_pdf_doc,clean_img
from MMAPIS.server import Article
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
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import concurrent.futures
import httpx
import asyncio
import uuid
import streamlit.components.v1 as components



MAX_ENTRIES = 50
TTL = 3*3600

@st.cache_data
def show_links(new_url,usage):
    link_style = """
            <style>
            .styled-link {
                color: #0066cc;
                text-decoration: none;
                font-size: 18px;
                padding: 10px 15px;
                background-color: #e6f2ff;
                border-radius: 5px;
            }
            .styled-link:hover {
                background-color: #c3e5ff;
            }
            </style>
        """
    styled_link = f'<a href="{new_url}" target="_blank" class="styled-link">Click here to {usage} document</a>'
    return link_style + styled_link

def save_config(config_path="./user.yaml"):
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def reset_register_option(option):
    st.session_state.update({"register_option": option})
    st.rerun()

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
        st.session_state["generated_summary"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]
        st.session_state["score"] = [None] * MAX_PDF_NUM

    if "generated_result" not in st.session_state or url_reset:
        st.session_state["generated_result"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]

    if "usage" not in st.session_state or url_reset:
        st.session_state["usage"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]


    if "num_pages" not in st.session_state or url_reset:
        st.session_state["num_pages"] = [0] * MAX_PDF_NUM


    if "pdf_run_model" not in st.session_state:
        st.session_state["pdf_run_model"] = False

    if "pdf_generated_summary" not in st.session_state or pdf_reset:
        st.session_state["pdf_generated_summary"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]
        st.session_state["pdf_score"] = [None] * MAX_PDF_NUM

    if "pdf_generated_result" not in st.session_state or pdf_reset:
        st.session_state["pdf_generated_result"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]

    if "pdf_usage" not in st.session_state or pdf_reset:
        st.session_state["pdf_usage"] = [[None] * MAX_PAGE_NUM for _ in range(MAX_PDF_NUM)]

    if "pdf_num_pages" not in st.session_state or pdf_reset:
        st.session_state["pdf_num_pages"] = [0] * MAX_PDF_NUM

def create_post_form(url, data):
    js_code = """
        function postToNewTab(url, data) {
                window.onload = function() {
                    console.log('Creating form element...');
                    var form = document.createElement('form');
                    console.log('Form element created before:', form);
                    form.method = 'POST';
                    form.action = url;
                    form.target = '_blank';
                    for (var key in data) {
                        if (data.hasOwnProperty(key)) {
                            var input = document.createElement('input');
                            input.type = 'hidden';
                            input.name = key;
                            input.value = data[key];
                            form.appendChild(input);
                        }
                    }

 // Append the form to the body to connect it to the DOM
            document.body.appendChild(form);
            console.log('Form element connected to DOM:', form);

            // Submit the form
            form.submit();
            console.log('Form submitted.');

            // Optional: Remove the form after submission
            document.body.removeChild(form);
        };
    }
    """
    print("url:", url)
    print("data:", data['usage'])
    form_data_json = json.dumps(data)
    html_code = f"""
    <script>
    {js_code}
    console.log('Calling postToNewTab with data:', {form_data_json});
    postToNewTab("{url}", {form_data_json});
    </script>
    """

    return html_code

async def upload_pdf_to_api(pdf_bytes: List[bytes],
                            user_name: str,
                            file_names: List[str],
                            file_ids: List[str],
                            temp_file: bool = False,
                            file_type: str = "pdf"):
    url = GENERAL_CONFIG["middleware_url"] + "/upload_zip_file/"
    if not isinstance(pdf_bytes, List):
        pdf_bytes = [pdf_bytes]
    if not isinstance(file_ids, List):
        file_ids = [file_ids]
    if not isinstance(file_names, List):
        file_names = [file_names]
    async def upload_single_file(pdf_bytes, file_id, file_name):
        try:
            async with httpx.AsyncClient() as client:
                pdf_bytes_io = bytes2io(pdf_bytes)
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                files = {
                    "zip_content": (f"{file_name}", pdf_bytes_io, "application/pdf")
                }
                data = {
                    "user_id": user_name,
                    "file_id": file_id,
                    "temp_file": temp_file,
                    "file_type": file_type
                }
                # 发送 POST 请求
                response = await client.post(url, files=files, data=data)
                json_info = custom_response_handler(response)

                if json_info["status"] == "success":
                    return True, json_info["message"]
                else:
                    return False, json_info.get('message', "Upload PDF failed")

        except Exception as e:
            error_msg = f"Exception occurred during upload: {e}"
            return False, error_msg

    results = await asyncio.gather(
        *(upload_single_file(pdf_bytes, file_id, file_name)
          for pdf_bytes, file_id, file_name in zip(pdf_bytes, file_ids, file_names)))
    print("save pdf results:", results)
    flag = [result[0] for result in results]
    if all(flag):
        return True, [result[1] for result in results]
    else:
        for result in results:
            if not result[0]:
                return False, result[1]

@st.cache_data(max_entries=MAX_ENTRIES, ttl=TTL)
def upload_pdf_sync_cache(*args, **kwargs):
    return asyncio.run(upload_pdf_to_api(*args, **kwargs))

@st.cache_data(max_entries=MAX_ENTRIES, ttl=TTL)
def get_links(keyword,
              max_num=10,
              line_length=15,
              searchtype='all',
              show_abstract='show',
              order='-announced_date_first',
              size=50,
              daily_type = 'cs',
              markdown=True,
              user_name:str=None,
              cur_time:datetime=None,
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



@st.cache_data(max_entries=MAX_ENTRIES, ttl=TTL)
def get_model_predcit(pdf_content:List[bytes]=None,user_name:str = None,**kwargs):
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
    global is_completed
    is_completed = True
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




@st.cache_data(max_entries=MAX_ENTRIES, ttl=TTL)
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
                        img_width:int = ALIGNMENT_CONFIG['img_width'],
                        threshold:float = 0.8,
                        user_name:str = None,
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
    # time.sleep(15)
    # return "document_level_summary", "section_level_summary", "document_level_summary_aligned"
    if pdf_content:
        pdf_content = bytes2io(pdf_content)
        files = [('pdf_content', pdf_content)]
        response = requests.post(url, files=files, data=params)
    else:
        response = requests.post(url, data=params)
    global is_completed
    is_completed = True
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
                   img_width:int = ALIGNMENT_CONFIG['img_width'],
                   user_name:str = None,
                   file_id:str = None,
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
        "img_width": img_width,
        "threshold": threshold,
        "tts_api_key": tts_api_key,
        "tts_base_url": tts_base_url,
        "app_secret": app_secret,
        "summarizer_params": json.dumps(summarizer_params),
    }
    params = dict_filter_none(params)
    print("params:",params.get("pdf","None"))
    if usage == 'regenerate':
        app_url = app_url + 'regeneration/'
        if pdf_content is None:
            response = requests.post(app_url, data=params)
        else:
            pdf_content = bytes2io(pdf_content)
            files = [('pdf_content', pdf_content)]
            response = requests.post(app_url, files=files, data=params)
        json_info = custom_response_handler(response,func_name='regeneration')
        if json_info["status"] == "success":
            res = json_info["message"]
        else:
            res = f"Get Generation Answer Error:{json_info.get('message','error')}"
            logger.error(f"get enhance answer error:{res}")

        text = res.strip()
        if url_mode:
            st.session_state["num_pages"][index] += 1
            st.session_state["generated_summary"][index].append(text.strip())
        else:
            st.session_state["pdf_num_pages"][index] += 1
            st.session_state["pdf_generated_summary"][index].append(text.strip())
    else:
        mid_url = GENERAL_CONFIG["middleware_url"] + "/app/"
        js_code = create_post_form(url = mid_url + f"{user_name}/{file_id}/", data=params)
        components.html(js_code, height=0)


@st.cache_data
def get_pdf_length(pdf:Union[str,bytes,Path]):
    assert isinstance(pdf,(str,bytes,Path)),f"pdf must be str,bytes or Path,but got {type(pdf)}"
    return get_pdf_doc(pdf)[1]

@st.cache_data
def get_max_tokens(model:str):
    max_tokens_map = {
        "gpt-3.5-turbo": 16385,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000
    }
    return max_tokens_map.get(model,16385)

@st.cache_data
def predict_nougat_model_time(num_pages:int,
                       batch_size:int,
                       time_cost_per_batch:float
                          ):
    return num_pages/batch_size * time_cost_per_batch


@st.cache_data
def predict_openai_request_time(
        num_sections:int,
        api_model:str,
    ):
    model_time_cost_map = {
        "gpt-3.5-turbo": 15,
        "gpt-4-turbo": 60,
        "gpt-4o": 30,
        "gpt-4o-mini": 40
    }
    return model_time_cost_map[api_model]


@st.cache_data
def estimate_model_time(current_stage:int,
                        input_size:int
                        ):
    model_time_cost_map = {
        "4090": [7, 15],
    }
    if current_stage == 1:
        return predict_nougat_model_time(input_size, batch_size=model_time_cost_map["4090"][0], time_cost_per_batch=model_time_cost_map["4090"][1])
    else:
        return predict_openai_request_time(input_size,var_model)

@st.cache_data(show_spinner=False,max_entries=MAX_ENTRIES, ttl=TTL)
def run_model_with_progress(_stage_function,
                            estimated_time,
                            current_stage,
                            total_stages:int=2,
                            *args,
                            **kwargs):
    progress_bar = stqdm(total=100)
    start_time = time.time()
    global is_completed
    is_completed = False

    def update_progress(estimated_time):
        st_processing_percentage = int(100 * (current_stage - 1) / total_stages)
        end_processing_percentage = int(100 * current_stage / total_stages)
        stage_processing_percentage = end_processing_percentage - st_processing_percentage
        progress_bar.n = int(st_processing_percentage)
        while not is_completed:
            elapsed = time.time() - start_time
            # 显示实际运行时间和预计运行时间
            if current_stage == 1:
                description = f"[Stage {current_stage}/{total_stages}] Nougat Model Prediction: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
            else:
                description = f"[Stage {current_stage}/{total_stages}] Document-Level Summary Generation: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
            progress_bar.set_description(description)
            if elapsed < estimated_time:
                # 如果还在预计时间内，正常更新进度
                progress = min(elapsed / estimated_time, 1.0)
                progress_bar.n = int(progress * stage_processing_percentage + st_processing_percentage)
            else:
                progress_bar.n = end_processing_percentage

            progress_bar.refresh()
            time.sleep(0.1)

        # 任务完成后，确保进度条到达100%
        progress_bar.n = end_processing_percentage
        final_time = time.time() - start_time
        progress_bar.set_description(f"Stage {current_stage} Completed: {final_time:.2f}s")
        progress_bar.refresh()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_stage_function, *args, **kwargs)
        update_progress(estimated_time)
        try:
            # 等待API请求完成
            results = future.result()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # 确保进度更新线程结束
            is_completed = True
    end_time = time.time()
    return results

@st.cache_data
def load_json_file(file):
    if file is not None:
        return json.loads(file.read())
    return None



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
    st.set_page_config(page_title="MMAPIS", page_icon=":smiley:", layout="wide",
                       initial_sidebar_state="auto")
    logger = init_logger()
    config_path = "./user.yaml"
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    options = ["Register", "Login", "Reset Password"]
    option_dict = {option: index for index, option in enumerate(options)}
    egister_option_placeholder = st.empty()
    option_menu_placeholder = st.empty()
    with option_menu_placeholder:
        register_option = option_menu("Options",
                                      options=options,
                                      default_index=option_dict.get(st.session_state.get("register_option"), 1), # default_index=1 if st.session_state.get("register_option", None) is None else option_dict[st.session_state.get("register_option")],
                                      icons=['house', 'gear', 'key'],
                                      menu_icon="cast",
                                      orientation="horizontal")

    st.session_state["register_option"] = register_option
    if st.session_state["register_option"] == "Register":
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
                pre_authorization=False,
                fields={'Form name': 'Register user',
                        'Email': 'Email',
                        'Username': 'Username(used to log in)',
                        'Password': 'Password',
                        'Repeat password': 'Repeat password',
                        'Register': 'Register'})
            if email_of_registered_user:
                save_config(config_path=config_path)
                reset_register_option("Login")
                st.success('User registered successfully')

        except Exception as e:
            st.error(e)
    elif st.session_state["register_option"] == "Reset Password":
        try:
            username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password()
            if username_of_forgotten_password:
                config["credentials"]["usernames"][username_of_forgotten_password]["password"] = new_random_password
                save_config(config_path=config_path)
                st.success(
                    f"New password({new_random_password}) has been set for {username_of_forgotten_password}.")
                # The developer should securely transfer the new password to the user.
            elif username_of_forgotten_password == False:
                st.error('Username not found')
            else:
                st.warning('Please enter your username and email')
        except Exception as e:
            st.error(f"A problem occurred: {e}")

    else:
        authenticator.login(
            fields={'Form name': 'Login Interface',
                    'Username': 'your username',
                    'Password': 'your password',
                    'Login': 'Login'})
        foget_password_button = st.empty()
        t = foget_password_button.button("Forget Password",
                                         key="forget_password",
                                         on_click=reset_register_option,
                                         args=("Reset Password",))
        if st.session_state["authentication_status"] is False:
            st.error('Username or password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
        elif st.session_state["authentication_status"]:
            option_menu_placeholder.empty()
            foget_password_button.empty()
            st.sidebar.write(f"Hello,{st.session_state.get('name', 'user')}. Welcome to MMAPIS")
            authenticator.logout(location="sidebar")
            #--------------------------------------init--------------------------------------
            MAX_PDF_NUM = 25
            MAX_PAGE_NUM = 10
            color_ls = ["blue", "green", "orange", "red", "violet", "gray", "rainbow"]
            REQUEST_BACKEND_URL = GENERAL_CONFIG["backend_url"]
            ##--------------------------------------sidebar--------------------------------------##


            # display options

            var_api_key = st.sidebar.text_input("OpenAI API Key", OPENAI_CONFIG["api_key"], key="api_key",
                                                help="Your OpenAI API key")
            var_base_url = st.sidebar.text_input("OpenAI Base URL",
                                                 key="base_url",
                                                 value=OPENAI_CONFIG["base_url"],
                                                 help="Your OpenAI base URL, e.g. https://api.openai.com/v1")

            var_model = st.sidebar.selectbox("Model",
                                            ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
                                            key="openai api model",
                                            help="OpenAI API model, e.g. gpt-3.5-turbo")

            with st.sidebar.expander("Display Options"):
                var_linelength = st.slider("Line Length", 10, 40, 18,
                                                   help="Line length of arXiv search results in each line")
                var_max_links = st.slider("Max Links", 0, MAX_PDF_NUM, 5,
                                                  help="Max links of arXiv search results")
                var_img_width = st.slider("Image Width", 100, 1000, ALIGNMENT_CONFIG['img_width'],
                                                  help="Image width of images in alignment results")

            with st.sidebar.expander("MMAPIS Configuration Options"):
                var_init_grid = st.slider("Initial Grid", 1, 5, 2,
                                          help="""Initial grid size for section summary; e.g., init_grid = 2 means the article is split based on subsections (`## Subsection`). 
                                          Summaries will perform better with bigger init_grid, but it may take longer to process(since more fine-grained sections are generated)
                                          and more API credits. 
                                          """)
                var_max_grid = st.slider("Maximum Grid", var_init_grid, 5, 4,
                                         help="Maximum grid size for section summary")

                ignore_title_options = ["abstract", "introduction", "background", "related work", "reference",
                                        "appendix", "acknowledge"]

                ignore_title_map = {
                    "abstract": "abs",
                    "introduction": "intro",
                    "acknowledge": "acknowledg"
                }
                # Define the multiselect widget
                var_ignore_titles = st.multiselect(
                    "Ignore Titles",
                    ignore_title_options,
                    default=["reference", "appendix", "acknowledge"],
                    help="Titles to ignore in the processing, e.g., 'abstract', 'introduction', 'acknowledge'"
                )
                var_threshold = st.slider("Threshold", 0.0, 1.0, 0.8,
                                          help="Threshold for title-like keywords in similarity when aligning the document-level summary with the corresponding image")
                ignore_title_mapping = lambda x: ignore_title_map.get(x, x)
                var_ignore_titles = list(map(ignore_title_mapping, var_ignore_titles))

            # Model configuration options
            with st.sidebar.expander("OpenAI Model Configuration Options"):
                var_num_processes = st.slider("Openai API Rate Limitation",
                                                      0, 15, 0,
                                                      help="Number of processes for GPT-3.5 model, if your API key is not limited, you can set it to 0, otherwise set it to 3 if you're limited by 3 requests per minute")
                max_tokens = get_max_tokens(var_model)
                var_max_token = st.slider("Max Tokens", 0, max_tokens, max_tokens,
                                                  help="Max tokens for the GPT-3.5 model")
                var_temperature = st.slider("Temperature", 0.0, 2.0, 0.5, help="Temperature of GPT-3.5 model")
                var_top_p = st.slider("Top P", 0.0, 1.0, 1.0, help="Top P of GPT-3.5 model")
                var_frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.1,
                                                          help="Frequency penalty of GPT-3.5 model")
                var_presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.2,
                                                         help="Presence penalty of GPT-3.5 model")


            with st.sidebar.expander("Prompts Optional"):
                var_prompt_ratio = st.slider("Prompt Ratio", 0.0, 1.0, 0.8,
                                                     help="Prompt ratio of GPT-3.5 model, e.g., 0.8 means up to 80% prompt and 20% response content")
                var_summary_prompt = st.file_uploader("Summary Prompt", type=["json"], key="summary_prompt",
                                                              help="Summary prompt JSON file, e.g., {'system': '', 'general_summary': ''}")
                var_integrate_prompt = st.file_uploader("Integrate Prompt", type=["json"],
                                                                key="integrate_prompt",
                                                                help="Integrate prompt JSON file, e.g., {'integrate_system': '', 'integrate': '', 'integrate_input': ''}")
                var_summary_prompt = load_json_file(var_summary_prompt)
                var_integrate_prompt = load_json_file(var_integrate_prompt)





            with st.sidebar.expander("Text-to-Speech Configuration Options"):
                var_tts_api_key = st.text_input("TTS API Key", key="tts_api_key", help="Your TTS API key")
                var_tts_base_url = st.text_input("TTS Base URL", key="tts_base_url", help="Your TTS base URL")
                var_app_secret = st.text_input("App Secret", key="app_secret", help="Your app secret")



            #--------------------------------------main--------------------------------------

            args = Args()
            st.header("MMAPIS: Multi-Modal Academic Papers Interpretation System Demo")
            var_req_url = st.text_input(
                'Requesting the URL of server:',
                REQUEST_BACKEND_URL,
                key="request_url",
                help="Request URL of server, default is your localhost, e.g., http://127.0.0.1:8000"
            )
            selection = option_menu(
                None,
                ["Search with URL", "Upload PDF"],
                icons=['house', 'cloud-upload'],
                menu_icon="cast",
                default_index=1,
                orientation="horizontal"
            )
            # Check for OpenAI API key
            if not var_api_key:
                st.error("Please input your OpenAI API key.")
                st.stop()
            tag_pattern = re.compile(r"\\tag\{(\d+)\}")
            # Search with URL logic
            if selection == "Search with URL":
                st.subheader("Choose search options:")
                var_daily_type = st.selectbox(
                    'Daily Type',
                    ('cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'eess', 'econ', 'astro-ph', 'cond-mat', 'gr-qc',
                     'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nucl-ex', 'nucl-th', 'quant-ph'),
                    help="Daily type of arXiv search, when keyword is None, this option will be used."
                )

                col1, col2, col3, col4 = st.columns(4)
                var_show_abstracts = col1.selectbox(
                    "Show Abstracts",
                    ['yes', 'no'],
                    key="show_abstracts",
                    help="Show abstract of arXiv search result."
                )
                var_searchtype = col2.selectbox(
                    "Choose Search Type",
                    ['all', 'title', 'abstract', 'author', 'comment',
                     'journal_ref', 'subject_class', 'report_num', 'id_list'],
                    key="searchtype",
                    help=("Search type of arXiv search. "
                          "Options include: All, Title, Abstract, Author, Comment, "
                          "Journal Ref, Subject Class, Report Num, ID List.")
                )
                var_order = col3.selectbox(
                    "Choose Search Order",
                    ['-announced_date_first', 'submitted_date',
                     '-submitted_date', 'announced_date_first', ''],
                    key="order",
                    help=("Order of arXiv search result. "
                          "-announced_date_first: Prioritizes by descending order of the earliest announcement date. "
                          "submitted_date: Sorts in ascending order based on submission date. "
                          "-submitted_date: Organizes in descending order by submission date. "
                          "announced_date_first: Arranges by ascending order of the earliest announcement date.")
                )
                var_size = col4.selectbox(
                    "Search Size",
                    [25, 50, 100, 200],
                    key="size",
                    help="Size of arXiv search result."
                )
                # Keyword input
                st.text("Input keyword to search specific fields in arXiv papers.")
                var_keyword = st.text_input(
                    "Keyword of arXiv Search",
                    "None",
                    key="keyword",
                    help="Keyword of arXiv search, e.g., 'quantum computer'."
                )

                result_placeholder = st.empty()
                if var_keyword.lower() == "none":
                    var_keyword = None
                    result_placeholder.text("No keyword input, automatically recommending papers delivered today.")
                else:
                    result_placeholder.text(f"Searching '{var_keyword}'...")
                var_links,var_titles,var_abstract,var_authors = get_links(var_keyword,
                                                                          max_num=var_max_links,
                                                                          line_length=var_linelength,
                                                                          searchtype=var_searchtype,
                                                                          order=var_order,
                                                                          size=var_size,
                                                                          daily_type=var_daily_type,
                                                                          markdown=True,
                                                                          cur_time=datetime.now().date(),
                                                                          user_name=st.session_state.get("name",None)
                                                                          )

                if var_links is None:
                    result_placeholder.text(f"Sorry, search failed. Error message: {var_authors}. Please retry.")
                    st.stop()

                var_options = [(i,False) for i in range(len(var_links))]
                # Display search results
                if var_keyword == "None":
                    result_placeholder.text(
                        f"Total {var_max_links} papers delivered today. Please choose the links you want to parse (you can choose multiple links):")
                else:
                    result_placeholder.text(
                        f"Searched '{var_keyword}' successfully. Found {var_max_links} related contents. Links as follows (you can choose multiple links):")

                # Choice list for checkboxes and abstracts
                choice_list = []
                for i,link in enumerate(var_links):
                    selected = st.checkbox(f"{var_titles[i]}", key=f"link{i + 1}",value=False)
                    var_options[i] = (i,selected)
                    abs_text = st.empty()
                    abs_text.markdown(f"{var_authors[i]}<br><br>{var_abstract[i]}",unsafe_allow_html=True)
                    choice_list.append([selected,abs_text])

                # Divider
                st.divider()

                # Selected record
                selected_record = st.empty()
                var_num_selected = sum(selected for _, selected in var_options)
                selected_record.text(f"You have selected {var_num_selected} links.")
                run_button = st.button(
                    "Run Model",
                    key="run_model_url",
                    help="Run model to generate summaries."
                )

                st.divider()

                # Display selected links
                st.text("You have selected the following links:")
                var_selected_links = filter_links(var_options)
                logger.info(f"selected links:{var_selected_links}")

                # Session state management
                var_variables = {key: value for key, value in globals().items() if key.startswith("var_") and not callable(value)}
                init_session_state()

                if "var_variables" not in st.session_state:
                    st.session_state["var_variables"] = var_variables

                # Update session state variables
                elif st.session_state["var_variables"] != var_variables:
                    # key_difference = set(var_variables.keys() ^ st.session_state["var_variables"].keys())
                    # value_diff = {key: var_variables.get(key, None) for key in key_difference}
                    # value_diff |= {key: st.session_state["var_variables"].get(key, None) for key in key_difference}
                    # print("key_difference:", key_difference, "value_diff:", value_diff)
                    st.session_state["var_variables"] = var_variables
                    st.session_state["run_model"] = False
                else:
                    st.session_state["run_model"] = True

                if run_button or st.session_state["run_model"]:
                    if var_num_selected == 0:
                        st.error("You haven't selected any links. Please select links before running the model.")
                        st.stop()

                    # Initialize session state if needed
                    if run_button:
                        init_session_state(url_reset=True)

                    progress_text = 'Processing...'
                    progress_bar = st.progress(0, text=progress_text)

                    # Clear previous selections if necessary
                    if st.session_state["run_model"]:
                        choice_list = [[x[0],x[1].empty()] for x in choice_list if not x[0]]

                    time_record = []
                    pdf_record = []
                    length_record = []
                    # Processing each selected PDF
                    progress_step = ["Nougat Model Prediction", "Document-Level Summary Generation"]

                    for i,index in enumerate(var_selected_links):
                        progress_text = f"Processing PDF {i + 1}/{var_num_selected}: {var_titles[index]}"
                        progress_bar.progress((i) / var_num_selected, text=progress_text)
                        progress_time = []
                        args.pdf = [var_links[index]]
                        user_input = st.chat_message("user")
                        user_input.caption(f"Input Pdf :")
                        user_input.markdown(f"{var_titles[index]}")

                        now_time = time.time()
                        answer_message = st.chat_message("assistant")

                        # Nougat Model Prediction
                        pdf_page_num = get_pdf_length(args.pdf[0])
                        model_result,pdf_name = run_model_with_progress(
                            estimated_time=estimate_model_time(current_stage=1,input_size=pdf_page_num),
                            current_stage=1,
                            _stage_function=get_model_predcit,
                            total_stages=2,
                            pdf_content=None,
                            user_name=st.session_state.get("name", None),
                            **vars(args)
                        )

                        elapsed_time = time.time() - now_time
                        progress_time.append(elapsed_time)

                        if model_result is None:
                            st.error(f"Model run failed. Error message: {pdf_name}. Please retry.")
                            st.stop()
                        else:
                            model_result, pdf_name = model_result[0], pdf_name[0]
                            num_sections = len(Article(model_result,grid=var_init_grid,max_grid=var_max_grid).sections)
                            document_summary, section_summary, document_summary_aligned = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=2, input_size=num_sections),
                                current_stage=2,
                                _stage_function=get_document_summary,
                                total_stages=2,
                                api_key=var_api_key,
                                base_url=var_base_url,
                                article=model_result,
                                file_name=pdf_name,
                                init_grid=var_init_grid,
                                max_grid=var_max_grid,
                                summary_prompts=var_summary_prompt,
                                integrate_prompts=var_integrate_prompt,
                                pdf=args.pdf[0],
                                img_width=var_img_width,
                                threshold=0.8,
                                summarizer_params=
                                {
                                    "rpm_limit": var_num_processes,
                                    "ignore_titles": var_ignore_titles,
                                    "prompt_ratio": var_prompt_ratio,
                                    "gpt_model_params":
                                        {
                                            "model": var_model,
                                            "max_tokens": var_max_token,
                                            "temperature": var_temperature,
                                            "top_p": var_top_p,
                                            "frequency_penalty": var_frequency_penalty,
                                            "presence_penalty": var_presence_penalty,
                                        }
                                },
                                user_name=st.session_state.get("name", None),
                            )

                            elapsed_time = time.time() - now_time
                            progress_time.append(elapsed_time - progress_time[-1])

                            # Display processing times
                            df = pd.DataFrame([progress_time], columns=progress_step,index=["Used Time"])
                            answer_message.caption(f"Parser Result:")
                            answer_message.dataframe(df,use_container_width=True,column_config=
                            {
                                "Nougat Model Prediction": st.column_config.NumberColumn(
                                    "Used Time of Model Prediction", format="%.2f s"),
                                "Document-Level Summary Generation": st.column_config.NumberColumn(
                                    "Used Time of Summary Generation", format="%.2f s"),
                            })

                            if not document_summary:
                                st.error(f"Summary generation failed, error message:{document_summary_aligned},please retry")
                                st.stop()

                            if run_button:
                                st.session_state["generated_summary"][i] = [re.sub(tag_pattern, "", document_summary_aligned)]
                            tabs = [f":{color_ls[k]}[Generation  {k}]" for k in range(st.session_state["num_pages"][i]+1)]

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
                                                         init_grid=var_init_grid,
                                                         max_grid=var_max_grid,
                                                         threshold=0.8,
                                                         img_width=var_img_width,
                                                         summarizer_params={
                                                             "rpm_limit": OPENAI_CONFIG["rpm_limit"],
                                                             "ignore_titles": var_ignore_titles,
                                                             "num_processes": var_num_processes,
                                                             "prompt_ratio": var_prompt_ratio,
                                                             "gpt_model_params":
                                                                 {
                                                                     "model": var_model,
                                                                     "max_tokens": var_max_token,
                                                                     "temperature": var_temperature,
                                                                     "top_p": var_top_p,
                                                                     "frequency_penalty": var_frequency_penalty,
                                                                     "presence_penalty": var_presence_penalty,
                                                                 }
                                                         },
                                                         user_name=st.session_state.get("name", None),
                                                         file_id= pdf_name,
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
                            time_record.append(progress_time)
                            pdf_record.append(var_links[index])
                            length_record.append(pdf_page_num)
                            st.divider()
                        time_df = pd.DataFrame(time_record,columns=progress_step)
                    progress_bar.progress(var_num_selected / var_num_selected, text=f"{var_num_selected} PDF Completed")
                    st.subheader("Running Result:")
                    if var_num_selected == 0:
                        st.warning("No links selected. Please select links before viewing the running result.")
                    else:
                        st.text(f"You have selected {var_num_selected} links, the running time is as follows:")
                        time_df.insert(0, "pdf_length", length_record)
                        time_df.insert(0, "pdf_link", pdf_record)
                        st.text(f"Running time details:")
                        st.dataframe(time_df, column_config={
                            "pdf_link": st.column_config.LinkColumn("PDF Link"),
                            "pdf_length": st.column_config.NumberColumn("Number of Pages", format="%d pages"),
                            **{step: st.column_config.NumberColumn(step, format="%.2f s") for step in progress_step}
                        }, hide_index=True, use_container_width=True)

            else:
                uploaded_file = st.file_uploader(label='Upload Local PDF Files (PDF/ZIP)',
                                                 accept_multiple_files=True,
                                                type=['pdf', 'zip'], key="upload_file",help="Upload one or more PDF or ZIP files.")
                logger.info(f"Uploaded files: {uploaded_file}, count: {len(uploaded_file)}")

                progress_step = ["Nougat Model Prediction", "Document-Level Summary Generation"]
                if uploaded_file:
                    # pdf_content_list:List[Bytes],var_file_names:List[str]
                    pdf_content_list, var_file_names = load_file(uploaded_file)
                    # remove duplicate pdf
                    pdf_info_df = pd.DataFrame({"pdf_name": var_file_names, "pdf_content": pdf_content_list})
                    pdf_info_df = pdf_info_df.drop_duplicates(subset=["pdf_name"])
                    pdf_content_list, var_file_names = pdf_info_df["pdf_content"].tolist(), pdf_info_df["pdf_name"].tolist()
                    # flag, pdf_content_list = asyncio.run(upload_pdf_to_api(
                    #     pdf_bytes=pdf_content_list,
                    #     user_name=st.session_state.get("name", "user").replace(" ", "_"),
                    #     file_names=var_file_names,
                    #     file_ids=var_file_names,
                    #     temp_file= False,
                    #     file_type="pdf"
                    # ))
                    flag, pdf_content_list = upload_pdf_sync_cache(
                        pdf_bytes=pdf_content_list,
                        user_name=st.session_state.get("name", "user").replace(" ", "_"),
                        file_names=var_file_names,
                        file_ids=var_file_names,
                        temp_file= False,
                        file_type="pdf"
                    )
                    if not flag:
                        st.error(f"Upload PDF to API failed. Due to {pdf_content_list}")
                        st.stop()
                    duplicate_pdf_count = len(uploaded_file) - len(pdf_content_list)
                    var_pdf_list = pdf_content_list

                    if duplicate_pdf_count > 0:
                        st.text(f"Found {duplicate_pdf_count} duplicate PDF files, which were automatically removed.")
                    if len(var_file_names) == 0 or any([pdf is None for pdf in var_file_names]):
                        st.error("Please upload valid PDF/ZIP files.")

                    pdf_button = st.button("Run Model",
                                           key="upload_run_model",
                                           help="Run the model to generate summaries")


                    #--------------------------------------session state--------------------------------------
                    var_variables = {key: value for key, value in globals().items() if
                                     key.startswith("var_") and not callable(value)}

                    # Initialize session state and handle updates.
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

                    # Process PDFs when the button is clicked or when session state indicates a change.
                    if pdf_button or st.session_state["pdf_run_model"]:
                        if pdf_button:
                            init_session_state(pdf_reset=True)

                        num_selected = len(var_file_names)
                        st.write(f"You have uploaded {num_selected} PDF files: {var_file_names}")

                        # Initialize progress bar with a clear description.
                        progress_text = "Processing..."
                        process_bar = st.progress(0, text=progress_text)

                        # Process each PDF file.
                        for i, pdf, name in zip(range(num_selected), pdf_content_list, var_file_names):
                            progress_text = f"Processing PDF {i + 1}/{num_selected}: {name}"
                            process_bar.progress((i) / num_selected, text=progress_text)

                            # Prepare arguments and display information about the input PDF.
                            progress_time_list = []
                            args.pdf_name = name
                            user_input = st.chat_message("user")
                            user_input.caption(f"input pdf :")
                            user_input.markdown(f"{name}")
                            user_input.download_button(label=f"Download {name}", data=pdf,
                                                       file_name=f"{str(name)}", key=f"download_{str(name)}")

                            # Start processing the PDF.
                            response_msg = st.chat_message("assistant")
                            st_time = time.time()

                            # Get number of pages and run the model.
                            num_pages = get_pdf_length(pdf)
                            args.pdf = [pdf]
                            model_result, pdf_name = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=1, input_size=num_pages),
                                current_stage=1,
                                _stage_function=get_model_predcit,
                                total_stages=2,
                                pdf_content=None,
                                user_name=st.session_state.get("name", None),
                                **vars(args)
                            )
                            elapsed_time = time.time() - st_time
                            progress_time_list.append(elapsed_time)

                            # Handle errors during model execution.
                            if model_result is None:
                                st.error(f"Processing {name} failed, error message: {pdf_name}, please retry")
                                st.stop()

                            # Extract results from the tuple.
                            model_result, pdf_name = model_result[0], pdf_name[0]

                            # Generate summaries.
                            document_summary, section_summary, document_summary_aligned  = run_model_with_progress(
                                estimated_time=estimate_model_time(current_stage=2, input_size=len(Article(model_result,grid=var_init_grid,max_grid=var_max_grid).sections)),
                                current_stage=2,
                                _stage_function=get_document_summary,
                                total_stages=2,
                                api_key=var_api_key,
                                base_url=var_base_url,
                                article=model_result,
                                file_name=pdf_name,
                                init_grid=var_init_grid,
                                max_grid=var_max_grid,
                                summary_prompts=var_summary_prompt,
                                integrate_prompts=var_integrate_prompt,
                                pdf=args.pdf[0],
                                img_width=var_img_width,
                                threshold=var_threshold,
                                summarizer_params=
                                {
                                    "rpm_limit": var_num_processes,
                                    "ignore_titles": var_ignore_titles,
                                    "prompt_ratio": var_prompt_ratio,
                                    "gpt_model_params":
                                        {
                                            "model": var_model,
                                            "max_tokens": var_max_token,
                                            "temperature": var_temperature,
                                            "top_p": var_top_p,
                                            "frequency_penalty": var_frequency_penalty,
                                            "presence_penalty": var_presence_penalty,
                                        }
                                },
                                user_name=st.session_state.get("name", None),
                            )

                            # Display the processing times.
                            elapsed_time = time.time() - st_time
                            progress_time_list.append(elapsed_time - progress_time_list[-1])
                            df = pd.DataFrame([progress_time_list], columns=progress_step, index=["Used Time"])

                            response_msg.caption(f"parser result:")
                            response_msg.dataframe(df, use_container_width=True, column_config=
                            {
                                "Nougat Model Prediction": st.column_config.NumberColumn(
                                    "Used Time of Model Prediction", format="%.2f s",
                                ),
                                "Document-Level Summary Generation": st.column_config.NumberColumn(
                                    "Used Time of Summary Generation", format="%.2f s",
                                ),
                            })

                            # Handle errors during summary generation.
                            if document_summary is None:
                                response_msg.error(f"Abstract generation failed, error message:{document_summary_aligned},please retry")
                                st.stop()

                            # Store the generated summary.
                            if pdf_button:
                                st.session_state["pdf_generated_summary"][i] = [re.sub(tag_pattern, "", document_summary_aligned)]
                            pdf_tabs = [f":{color_ls[k]}[res {k}]" for k in range(st.session_state["pdf_num_pages"][i] + 1)]

                            for pdf_page_idx, pdf_tab_idx in enumerate(response_msg.tabs(pdf_tabs)):
                                pdf_tab_idx.markdown(st.session_state["pdf_generated_summary"][i][pdf_page_idx],unsafe_allow_html=True)

                                # Function for enhancing answers (partial application).
                                enhance_answer = partial(get_enhance_answer,
                                                            api_key=var_api_key,
                                                            base_url=var_base_url,
                                                            section_summary=section_summary,
                                                            index=i,
                                                            pdf=pdf,
                                                            page_idx=pdf_page_idx,
                                                            raw_md_text=model_result,
                                                            init_grid=var_init_grid,
                                                            max_grid=var_max_grid,
                                                            threshold=var_threshold,
                                                            url_mode = False,
                                                            img_width=var_img_width,
                                                            summarizer_params={
                                                                "rpm_limit": var_num_processes,
                                                                "ignore_titles": var_ignore_titles,
                                                                "prompt_ratio": var_prompt_ratio,
                                                                "gpt_model_params":
                                                                    {
                                                                        "model": var_model,
                                                                        "max_tokens": var_max_token,
                                                                        "temperature": var_temperature,
                                                                        "top_p": var_top_p,
                                                                        "frequency_penalty": var_frequency_penalty,
                                                                        "presence_penalty": var_presence_penalty,
                                                                    }
                                                            },
                                                            user_name=st.session_state.get("name", None),
                                                            file_id=pdf_name,
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

                            st.divider()
                        process_bar.progress(num_selected / num_selected,
                                              text=f"{num_selected} PDF Completed")
                else:
                    st.error("Unable to get file, please upload file")





