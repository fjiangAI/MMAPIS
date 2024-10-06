import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Literal, Tuple
import requests
import yaml
import httpx
import asyncio
import concurrent.futures
from tqdm import tqdm as stqdm
from pathlib import Path
import streamlit.components.v1 as components
import streamlit as st
from stqdm import stqdm
from MMAPIS.client.config import ALIGNMENT_CONFIG, GENERAL_CONFIG, LOGGER_MODES, TTS_CONFIG
from MMAPIS.client.utils.logger import init_logging
from MMAPIS.client.utils.client_models import MMAPIS_CLIENT_CONFIG
import io
import fitz
from functools import singledispatch
import zipfile
import re



USER_CONFIG_PATH = GENERAL_CONFIG["user_config_path"]
MAX_PDF_NUM = MMAPIS_CLIENT_CONFIG.num_pdf.default
MAX_ENTRIES = MMAPIS_CLIENT_CONFIG.max_entries
TTL = MMAPIS_CLIENT_CONFIG.ttl
MAX_PAGE_NUM = MMAPIS_CLIENT_CONFIG.max_page_num

init_logging()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_MODES)



def clean_img(text:str):
    if not text:
        return text
    block_pattern = re.compile(r'<div.*?</div>\n+</div>', re.DOTALL)
    blocks = block_pattern.findall(text)
    for block in blocks:
        text = text.replace(block, '')
    return text

def _process_zip_file(uploaded_file) -> Tuple[List[bytes], List[str]]:
    """Extract contents and names from a ZIP file."""
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), 'r') as zip_ref:
        file_names = zip_ref.namelist()
        file_content = [zip_ref.read(name) for name in file_names]
    return file_content, file_names


def _process_pdf_file(uploaded_file) -> Tuple[List[bytes], List[str]]:
    """Process a PDF file, returning its content and name."""
    return [uploaded_file.getvalue()], [uploaded_file.name]

def process_single_file(uploaded_file) -> Optional[Tuple[List[bytes], List[str]]]:
    """
    Process a single uploaded file based on its type.

    This function handles ZIP and PDF files, extracting their contents and names.
    RAR files are not supported due to potential errors with the 'unrar' library.

    Args:
        uploaded_file: A file-like object with 'name' and 'getvalue' attributes.

    Returns:
        A tuple containing two lists: file contents (as bytes) and file names,
        or None if the file type is not supported.

    Raises:
        SystemExit: If an unsupported file type is encountered.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "zip":
        return _process_zip_file(uploaded_file)
    elif file_extension == "pdf":
        return _process_pdf_file(uploaded_file)
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        raise Exception(f"Unsupported file type: {file_extension}")


def bytes2io(bytes_data):
    return io.BufferedReader(io.BytesIO(bytes_data))


def get_pdf_list(uploaded_files:List):
    pdf_list, file_names = [], []
    for uploaded_file in uploaded_files:
        tmp_pdf_list, tmp_file_names = process_single_file(uploaded_file = uploaded_file)
        pdf_list.extend(tmp_pdf_list)
        file_names.extend(tmp_file_names)
    return pdf_list, file_names


def dict_filter_none(d:dict):
    return {k:v for k,v in d.items() if v and v != 'null'}

def log_response(func_name, status_code, content_type=None, message=None, error=False):
    """
    Logs the response status, including function name, status code, content type, and any additional message.

    Parameters:
    - func_name (str): The name of the function where the log is being made.
    - status_code (int): HTTP response status code.
    - content_type (str, optional): The content type of the response. Default is None.
    - message (str, optional): Additional message to log. Default is None.
    - error (bool, optional): Whether the log is for an error. Default is False.
    """
    if error:
        logger.error(f"[{func_name}] Error - Status: {status_code}, Content-Type: {content_type}, Message: {message}")
    else:
        logger.info(f"[{func_name}] Success - Status: {status_code}, Content-Type: {content_type}, Message: {message}")


def handle_json_response(response, func_name):
    """
    Handles responses with JSON content-type.

    Parameters:
    - response (requests.Response): The HTTP response object.
    - func_name (str): The name of the function handling the response.

    Returns:
    dict: A dictionary containing the status and message of the response.
    """
    json_info = response.json()
    if response.status_code == 200 and json_info:
        log_response(func_name, response.status_code)
        return {
            'status': "success",
            'message': json_info.get('message', 'Operation successful')
        }
    else:
        error_status = json_info.get('status', response.status_code)
        error_message = json_info.get('message', 'Unknown error')
        log_response(func_name, response.status_code, error=True, message=error_message)
        return {
            'status': "error",
            'message': f"{error_status}: {error_message}"
        }


def handle_binary_response(response, func_name):
    """
    Handles responses with binary content-type (audio, zip, etc.).

    Parameters:
    - response (requests.Response): The HTTP response object.
    - func_name (str): The name of the function handling the response.

    Returns:
    dict: A dictionary containing the status and binary content.
    """
    log_response(func_name, response.status_code)
    return {
        'status': "success",
        "message": response.content
    }


def custom_response_handler(response: requests.Response, func_name: str = ''):
    """
    A custom handler for processing HTTP responses based on content type and status code.

    Parameters:
    - response (requests.Response): The HTTP response object.
    - func_name (str): The name of the function making the HTTP request (for logging purposes).

    Returns:
    dict: A dictionary containing the result status and message of the response.
    """
    try:
        content_type = response.headers.get('content-type', '')

        if 'application/json' in content_type:
            return handle_json_response(response, func_name)

        elif 'audio/mp3' in content_type or 'application/zip' in content_type:
            return handle_binary_response(response, func_name)

        else:
            log_response(func_name, response.status_code, content_type, error=True, message="Unknown content type")
            msg = response.text
            return {
                "status": "success" if response.status_code == 200 else "error",
                "message": msg
            }

    except Exception as e:
        log_response(func_name, response.status_code, error=True, message=str(e))
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }


@st.cache_data
def generate_cache_key(**kwargs):
    """
    Generate a cache key based on the function's parameters.

    :param kwargs: Keyword arguments passed to the function.
    :return: A unique hash string representing the parameters.
    """
    # transform the parameters into a string
    params_str = json.dumps(kwargs, sort_keys=True)
    # calculate the SHA-256 hash value of the string
    hash_object = hashlib.sha256(params_str.encode())
    # return the hexadecimal representation of the hash value
    return hash_object.hexdigest()


def estimate_num_sections(text:str,min_grained_level:int = 2):
    section_pattern = re.compile(
        r'\n+(#{{1,{}}}\s+.*?)\n+(.*?)(?=\n+#{{1,{}}}\s+|$)'.format(min_grained_level, min_grained_level),
        re.DOTALL
    )
    return len(section_pattern.findall(text))


def save_config(config: Dict,
                USER_CONFIG_PATH=USER_CONFIG_PATH):
    with open(USER_CONFIG_PATH, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def reset_register_option(option):
    st.session_state.update({"register_option": option})
    st.rerun()

@st.cache_data
def init_logger():
    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(LOGGER_MODES)
    return logger


@st.cache_data
def filter_links(links):
    selected_index = [index for index, selected in links if selected]
    return selected_index

@st.cache_data
def load_file(uploaded_file):
    return get_pdf_list(uploaded_file)



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
    """
    Create a POST form and submit it using JavaScript.

    Parameters:
    - url (str): The URL to submit the form to.
    - data (dict): The form data to send.
    """
    # Ensuring that changes between two identical calls will be detected, otherwise the form will not be submitted twice with the same data.
    current_time = time.time()
    js_code = """
    function createAndSubmitForm(url, data,current_time) {
        var form = document.createElement('form');
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
        document.body.appendChild(form);
        form.submit();
        document.body.removeChild(form);
    }
    """

    form_data_json = json.dumps(data)
    html_code = f"""
    <script>
    {js_code}
    document.addEventListener('DOMContentLoaded', function() {{
        createAndSubmitForm("{url}", {form_data_json},{current_time});
    }});
    </script>
    """

    st.components.v1.html(html_code, height=0)


async def upload_pdf_to_api(pdf_bytes: List[bytes],
                            user_name: str,
                            file_names: List[str],
                            file_ids: List[str],
                            temp_file: bool = False,
                            file_type: str = "pdf"):
    if not isinstance(pdf_bytes, List):
        pdf_bytes = [pdf_bytes]
    if not isinstance(file_ids, List):
        file_ids = [file_ids]
    if not isinstance(file_names, List):
        file_names = [file_names]
    async def upload_single_file(pdf_bytes, file_id, file_name):
        try:
            url = GENERAL_CONFIG["middleware_url"] + f"/upload_zip_file/{user_name}/{file_id}/"
            async with httpx.AsyncClient() as client:
                pdf_bytes_io = bytes2io(pdf_bytes)
                file_name = file_name if file_name.endswith(".pdf") else f"{file_name}.pdf"
                files = {
                    "zip_content": (f"{file_name}", pdf_bytes_io, "application/pdf")
                }
                data = {
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
              ):
    url = GENERAL_CONFIG['middleware_url'] + '/get_links/'
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
def get_model_predcit(pdf_content:List[bytes]=None,user_name:str = None,file_id:str = None,**kwargs):
    nougat_url = GENERAL_CONFIG['middleware_url'] + f'/pdf2md/{user_name}/{file_id}/'
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
    json_info = response.json()
    if json_info["status"] == "success":
        file_names = []
        article_ls = []
        for item in json_info["message"]:
            file_names.append(item["file_name"])
            article_ls.append(item["text"])
        return article_ls,file_names
    else:
        return None,json_info.get('message',"error")




def get_document_level_summary(
                        api_key:str,
                        base_url:str,
                        raw_md_text:str,
                        file_name:str=None,
                        min_grained_level:int = 2,
                        max_grained_level:int = 4,
                        summary_prompts:Dict = None,
                        document_prompts:Dict = None,
                        summarizer_params:Dict = None,
                        pdf:Optional[Union[str,bytes,Path]] = None,
                        pdf_content:Optional[bytes] = None,
                        img_width:int = ALIGNMENT_CONFIG['img_width'],
                        threshold:float = 0.8,
                        user_name:str = None,
                        file_id:str = None,
                        request_id:str = None,
                        from_middleware:bool = False):
    url = GENERAL_CONFIG['middleware_url'] + f'/summary/{user_name}/{file_id}/{request_id}/'
    params = {
        "api_key": api_key,
        "base_url": base_url,
        "raw_md_text": raw_md_text,
        "pdf": pdf,
        "file_name": file_name,
        "min_grained_level": min_grained_level,
        "max_grained_level": max_grained_level,
        "summary_prompts": json.dumps(summary_prompts),
        "document_prompts": json.dumps(document_prompts),
        "img_width": img_width,
        "threshold": threshold,
        "summarizer_params": json.dumps(summarizer_params),
        "from_middleware": from_middleware
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
                   usage: Literal['blog', 'speech', 'regenerate', 'recommend', 'qa'] = 'regenerate',
                   raw_md_text:str=None,
                   section_level_summary:str = None,
                   prompts:Dict = None,
                   pdf:Optional[Union[str,bytes,None]] = None,
                   min_grained_level:int = ALIGNMENT_CONFIG['min_grained_level'],
                   max_grained_level:int = ALIGNMENT_CONFIG['max_grained_level'],
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
                   request_id:str = None,
                   **kwargs):
    if not usage in ['blog', 'speech', 'regenerate','recommend','qa']:
        raise ValueError(f"usage must be in ['blog','speech','regenerate','recommend','qa'],but got {usage}")
    app_url = GENERAL_CONFIG['middleware_url'] + '/app/'
    params = {
        "api_key": api_key,
        "base_url": base_url,
        "document_level_summary": document_level_summary,
        "usage": usage,
        "section_level_summary": section_level_summary,
        "raw_md_text": raw_md_text,
        "prompts": json.dumps(prompts),
        "pdf": pdf,
        "min_grained_level": min_grained_level,
        "max_grained_level": max_grained_level,
        "img_width": img_width,
        "threshold": threshold,
        "tts_api_key": tts_api_key,
        "tts_base_url": tts_base_url,
        "app_secret": app_secret,
        "summarizer_params": json.dumps(summarizer_params),
    }
    params = dict_filter_none(params)

    if usage == 'regenerate':
        app_url = app_url + f'regeneration/{user_name}/{file_id}/{request_id}/'
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
        js_code = create_post_form(url = app_url + f"{user_name}/{file_id}/{request_id}/", data=params)
        components.html(js_code, height=0)


@st.cache_data
def get_pdf_length(pdf:Union[str,bytes,Path]):
    assert isinstance(pdf,(str,bytes,Path)),f"pdf must be str,bytes or Path,but got {type(pdf)}"
    return get_pdf_doc(pdf)[1]

@st.cache_data
def get_max_tokens(max_tokens_map:Dict,
                   model:str):
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
                        input_size:int,
                        model:str,
                        ):
    model_time_cost_map = {
        "4090": [7, 15],
    }
    if current_stage == 1:
        return predict_nougat_model_time(input_size, batch_size=model_time_cost_map["4090"][0], time_cost_per_batch=model_time_cost_map["4090"][1])
    else:
        return predict_openai_request_time(input_size,api_model=model)

# def run_model_with_progress(_stage_function,
#                             estimated_time,
#                             current_stage,
#                             total_stages:int=2,
#                             *args,
#                             **kwargs):
#     progress_bar = stqdm(total=100)
#     start_time = time.time()
#     print(f"running function:{_stage_function.__name__}, at time:{start_time}")
#     def update_progress(estimated_time, future=None):
#         st_processing_percentage = int(100 * (current_stage - 1) / total_stages)
#         end_processing_percentage = int(100 * current_stage / total_stages)
#         stage_processing_percentage = end_processing_percentage - st_processing_percentage
#         progress_bar.n = int(st_processing_percentage)
#         while not future.done():
#             elapsed = time.time() - start_time
#             # 显示实际运行时间和预计运行时间
#             if current_stage == 1:
#                 description = f"[Stage {current_stage}/{total_stages}] Nougat Model Prediction: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
#             else:
#                 description = f"[Stage {current_stage}/{total_stages}] Document-Level Summary Generation: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
#             progress_bar.set_description(description)
#             if elapsed < estimated_time:
#                 # 如果还在预计时间内，正常更新进度
#                 progress = min(elapsed / estimated_time, 1.0)
#                 progress_bar.n = int(progress * stage_processing_percentage + st_processing_percentage)
#             else:
#                 progress_bar.n = end_processing_percentage
#
#             progress_bar.refresh()
#             time.sleep(0.1)
#
#         # 任务完成后，确保进度条到达100%
#         progress_bar.n = end_processing_percentage
#         final_time = time.time() - start_time
#         progress_bar.set_description(f"Stage {current_stage} Completed: {final_time:.2f}s")
#         progress_bar.refresh()
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(_stage_function, *args, **kwargs)
#         update_progress(estimated_time, future)
#         try:
#             # 等待API请求完成
#             results = future.result()
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#     print("future done:",future.done())
#     end_time = time.time()
#     return results


def run_model_with_progress(_stage_function,
                            estimated_time,
                            current_stage,
                            total_stages:int=2,
                            *args,
                            **kwargs):
    st.session_state["progress_bar"] = stqdm(total=100)
    st.session_state["progress_bar"] = stqdm(total=100)
    start_time = time.time()
    print(f"running function:{_stage_function.__name__}, at time:{start_time}")
    def update_progress(estimated_time, future=None):
        st_processing_percentage = int(100 * (current_stage - 1) / total_stages)
        end_processing_percentage = int(100 * current_stage / total_stages)
        stage_processing_percentage = end_processing_percentage - st_processing_percentage
        st.session_state["progress_bar"].n = int(st_processing_percentage)
        while not future.done():
            elapsed = time.time() - start_time
            # 显示实际运行时间和预计运行时间
            if current_stage == 1:
                description = f"[Stage {current_stage}/{total_stages}] Nougat Model Prediction: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
            else:
                description = f"[Stage {current_stage}/{total_stages}] Document-Level Summary Generation: {elapsed:.2f}s elapsed / {estimated_time:.2f}s estimated"
            st.session_state["progress_bar"].set_description(description)
            if elapsed < estimated_time:
                # 如果还在预计时间内，正常更新进度
                progress = min(elapsed / estimated_time, 1.0)
                st.session_state["progress_bar"].n = int(progress * stage_processing_percentage + st_processing_percentage)
            else:
                st.session_state["progress_bar"].n = end_processing_percentage

            st.session_state["progress_bar"].refresh()
            time.sleep(0.1)

        # 任务完成后，确保进度条到达100%
        st.session_state["progress_bar"].n = end_processing_percentage
        final_time = time.time() - start_time
        st.session_state["progress_bar"].set_description(f"Stage {current_stage} Completed: {final_time:.2f}s")
        st.session_state["progress_bar"].refresh()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_stage_function, *args, **kwargs)
        update_progress(estimated_time, future)
        try:
            # 等待API请求完成
            results = future.result()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    end_time = time.time()
    return results

@st.cache_data
def load_json_file(file):
    if file is not None:
        return json.loads(file.read())
    return None


@singledispatch
def get_pdf_doc(pdf: Union[str, bytes, Path],
                proxy: Optional[dict] = None,
                headers: Optional[dict] = None,
                pdf_name: Optional[str] = None) -> Tuple[fitz.Document, int, str]:
    """
    Open and process a PDF document from various sources.

    This function uses singledispatch to handle different input types:
    - URL (string starting with 'http')
    - Local file path (string or Path object)
    - Bytes object

    Args:
        pdf: The PDF source (URL, file path, or bytes).
        proxy: Optional proxy settings for URL requests.
        headers: Optional headers for URL requests.
        pdf_name: Optional name for the PDF when source is bytes.

    Returns:
        A tuple containing:
        - fitz.Document object
        - Number of pages in the PDF
        - Name of the PDF file

    Raises:
        NotImplementedError: If an unsupported input type is provided.
    """
    raise NotImplementedError(f"Unsupported type: {type(pdf)}")


@get_pdf_doc.register(str)
def _(pdf: str, proxy: Optional[dict] = None, headers: Optional[dict] = None,
      pdf_name: Optional[str] = None) -> Tuple[fitz.Document, int, str]:
    if pdf.startswith("http"):
        name = pdf.split("/")[-1].replace('.', '_')
        # self.size = len(fitz.open(stream=urllib.request.urlopen(pdf).read(), filetype="pdf"))
        pdf_doc_obj = fitz.open(stream=requests.get(pdf, proxies=proxy, headers=headers).content, filetype="pdf")
        size = len(pdf_doc_obj)

    else:
        name = pdf.split("/")[-1]
        pdf_doc_obj = fitz.open(Path(pdf), filetype="pdf")
        size = len(pdf_doc_obj)
    return pdf_doc_obj, size, name

@get_pdf_doc.register(bytes)
def _(pdf: bytes, proxy: Optional[dict] = None, headers: Optional[dict] = None,
      pdf_name: Optional[str] = None) -> Tuple[fitz.Document, int, str]:
    pdf_doc = fitz.open(stream=pdf, filetype="pdf")
    name = pdf_name or f"unknown_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return pdf_doc, len(pdf_doc), name

@get_pdf_doc.register(Path)
def _(pdf: Path, proxy: Optional[dict] = None, headers: Optional[dict] = None,
      pdf_name: Optional[str] = None) -> Tuple[fitz.Document, int, str]:
    pdf_doc = fitz.open(pdf, filetype="pdf")
    return pdf_doc, len(pdf_doc), pdf.name


def check_session_state(var_variables: Dict,upload_pdf:bool = False):
    if upload_pdf:
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
    else:
        if "var_variables" not in st.session_state:
            # Initialize session state with provided variables if not already present.
            st.session_state["var_variables"] = var_variables

        # Update Streamlit's session state if there are changes in the provided variables.
        elif st.session_state["var_variables"] != var_variables:
            st.session_state["var_variables"] = var_variables
            st.session_state["run_model"] = False
        else:
            st.session_state["run_model"] = True