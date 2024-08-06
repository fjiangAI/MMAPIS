from functools import singledispatch
from typing import List, Union
import io
from pathlib import Path
import fitz
import logging
import zipfile
import sys
import requests
from datetime import datetime
import json
import tiktoken
import re
import os
import os.path as osp
from urllib.parse import urljoin
from MMAPIS.config.config import GENERAL_CONFIG
from io import BytesIO
import base64


def get_best_gpu(choice_list=None):
    import torch
    total_gpus = torch.cuda.device_count()

    if not choice_list:
        choice_list = list(range(total_gpus))

    choice_list = list(filter(lambda x: x < total_gpus, choice_list))
    max_memory = 0
    best_gpu = None

    for gpu in choice_list:
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = meminfo.free
        if free_memory > max_memory:
            max_memory = free_memory
            best_gpu = gpu

    return best_gpu, max_memory/1024**3



def get_batch_size():
    import torch
    if torch.cuda.is_available():
        best_gpu, free_memory = get_best_gpu([0])
        BATCH_SIZE = int(
            free_memory * 0.3
        )
        # BATCH_SIZE = int(
        #     torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        # )
        logging.info(f"Best GPU: {best_gpu}. Batch size: {BATCH_SIZE}")
        if BATCH_SIZE == 0:
            logging.warning("GPU VRAM is too small. Computing on CPU.")

    else:
        # don't know what a good value is here. Would not recommend to run on CPU
        BATCH_SIZE = 1
        logging.warning("No GPU found. Conversion on CPU is very slow.")
    return BATCH_SIZE


def get_pdf_list(uploaded_files:List):
    pdf_list, file_names = [], []
    for uploaded_file in uploaded_files:
        tmp_pdf_list, tmp_file_names = process_single_file(uploaded_file = uploaded_file)
        print("tmp_file_names",tmp_file_names)
        pdf_list.extend(tmp_pdf_list)
        file_names.extend(tmp_file_names)
    return pdf_list, file_names


def process_single_file(uploaded_file):
    if uploaded_file.name.endswith(".zip"):
        logging.info(f"get zip file{uploaded_file.name}")
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            file_content = [zip_ref.read(name) for name in file_names]
        return file_content, file_names

    ## not recommend to use ra, dur to unrar error
    elif uploaded_file.name.endswith(".rar"):
        logging.error("rar file not support")
        # file_bytes = io.BytesIO(uploaded_file.getvalue())
        # with rarfile.RarFile(file_bytes, 'r') as rar_ref:
        #     file_names = rar_ref.namelist()
        #     file_content = [rar_ref.read(name) for name in file_names]
        # return file_content, file_names

    elif uploaded_file.name.endswith(".pdf"):
        logging.info(f"get pdf file{uploaded_file.name}")

        file_content = [uploaded_file.getvalue()]
        file_names = [uploaded_file.name]
        return file_content, file_names
    else:
        logging.error("File type not supported.")
        sys.exit(1)


@singledispatch
def get_pdf_doc(pdf,proxy=None,headers=None,pdf_name=None):
    # pdf options:
    # 1. [str] url
    # 2. [str] path
    # 3. [Path] path
    # 4. bytes
    raise NotImplementedError("Unsupported type")

@get_pdf_doc.register(str)
def _(pdf,proxy=None,headers=None,pdf_name=None):

    if "http" in pdf:
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
def _(pdf,proxy=None,headers=None,pdf_name=None):

    logging.info(f"Opening PDF to rasterize from bytes format")
    pdf_doc_obj = fitz.open(stream=pdf, filetype="pdf")
    size = len(pdf_doc_obj)
    name = pdf_name if pdf_name else f"unk_pdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return pdf_doc_obj, size, name

@get_pdf_doc.register(Path)
def _(pdf,proxy=None,headers=None,pdf_name=None):
    logging.info(f"Opening PDF to rasterize from Path format")
    pdf_doc_obj = fitz.open(pdf, filetype="pdf")
    name = pdf.name
    size = len(pdf_doc_obj)
    return pdf_doc_obj, size, name



def custom_response_handler(response: requests.Response,
                            func_name:str=''):

    try:
        if 'application/json' in response.headers['content-type']:
            json_info = response.json()
            if response.status_code == 200 and json_info:
                logging.info(f"{func_name} process success")
                return {
                    'status': "success",
                    'message': json_info['message']
                }
            elif response.status_code == 400:
                logging.error(f"request body error[{func_name}], status: {json_info.get('status', response.status_code)}")
            elif response.status_code == 500:
                logging.error(f"internal error[{func_name}], status: {json_info.get('status', response.status_code)}")
            else:
                logging.error(f"unknown response error[{func_name}], status code: {response.status_code}")
            error_msg =  json_info.get('message', 'Unknown error')
            error_status = json_info.get('status', response.status_code)
            json_msg = {
                'status': "error",
                'message': error_status + error_msg
                        }
            return json_msg

        elif 'audio/mp3' in response.headers['content-type']:
            logging.info(f"{func_name} process success")
            return {
                'status': "success",
                "message": response.content
            }


        elif "application/zip" in response.headers['content-type']:
            logging.info(f"{func_name} process success")
            return {
                'status': "success",
                "message": response.content
            }


        else:
            logging.error(f"unknown response type:{response.headers.get('content-type','unknown')} error[{func_name}], status code: {response.status_code}")
            msg = response.text
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": msg
                }
            else:
                return {
                    "status": "error",
                    "message": msg
                }

    except Exception as e:
        logging.error(f"response error[{func_name}], an unexpected error occurred: {str(e)}")
        return {"status": "error",
                "message": f"An unexpected error occurred: {str(e)}"}



def dict_filter_none(d:dict):
    return {k:v for k,v in d.items() if v and v != 'null'}


def handle_request(url:str,parameters = None,proxy=None, headers = None):
        success = False
        response = None
        try:
            if proxy is None:
                raw_response = requests.post(url, headers=headers, json=parameters)
            else:
                raw_response = requests.post(url, headers=headers, json=parameters, proxies=proxy)

            raw_response.raise_for_status()
            response = json.loads(raw_response.content.decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            success = True
        except requests.exceptions.RequestException as e:
            content = f"Request Error: {str(e)}"
        except json.JSONDecodeError as e:
            content = f"JSON Decode Error: {str(e)}"
        except KeyError as e:
            content = f"KeyError: {str(e)}"
        except Exception as e:
            content = f"Unexpected Error: {str(e)}"

        return response,content, success


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613",
                            detailed_img:bool=False):
    """
        Returns the number of tokens used by a list of messages.
        Args:
            messages: A list of messages. Each message is:
                dict (with keys "role" and "content".)
                string (in which case the role is assumed to be "user".)
                list (already normalized into a list of dicts.)
            model: The model to use. Defaults to "gpt-3.5-turbo-16k-0613".
        Returns:
            The number of tokens used by the messages.
    """
    # try:
    #     encoding = tiktoken.encoding_for_model(model)
    encode_model = None
    map_dict = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "text-embedding-ada": "cl100k_base",
        "text-davinci": "p50k_base",
        "Codex": "p50k_base",
        "davinci": "p50k_base",
    }
    for key in map_dict.keys():
        if key in model:
            encode_model = map_dict[key]
            break

    if not encode_model:
        logging.error(f"model {model} not found,load default model: cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")
    else:
        encoding = tiktoken.get_encoding(encode_model)
    if isinstance(messages, dict):
        messages = [messages]
    elif isinstance(messages, list):
        pass
    elif isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    num_tokens = 0
    if "gpt-3.5-turbo" in model:  # note: future models may deviate from this
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    elif "gpt-4" in model:
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value,str):
                    num_tokens += len(encoding.encode(value))
                else:
                    for item in value:
                        print("item:",item)
                        for k,v in item.items():
                            if isinstance(v,str):
                                num_tokens +=  len(encoding.encode(v))
                            else:
                                num_tokens += 129 if detailed_img else 65
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def strip_title(title):
    title = re.sub(r'^[^a-zA-Z]+', '', title)
    title = re.sub(r'[^a-zA-Z]+$', '', title)
    title = re.sub(r'[^a-zA-Z0-9._ ]+', '', title)
    return title




def zip_dir_to_bytes(dir_path):
    """
    compress the directory to zip file
    :param dir_path: the directory path
    :return: the zip file bytes
    """
    bytes_io = BytesIO()
    with zipfile.ZipFile(bytes_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, dir_path))

    # reset the file pointer to the beginning
    bytes_io.seek(0)
    return bytes_io.getvalue()



def extract_zip_from_bytes(zip_data,
                           extract_dir):
    """
    Extracts contents from byte data of a ZIP file into a specified directory.

    Parameters:
    - zip_data: The byte data of the ZIP file.
    - extract_dir: The path to the target directory for extraction.
    """
    # Convert the byte data into a BytesIO object for reading by the zipfile module
    zip_bytes_io = BytesIO(zip_data)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    # Use zipfile to read the BytesIO object
    with zipfile.ZipFile(zip_bytes_io, 'r') as zip_ref:
        # Extract to the specified directory
        zip_ref.extractall(extract_dir)

def get_pdf_name(pdf):
    if isinstance(pdf, Path):
        return pdf.name
    elif isinstance(pdf, bytes):
        name = f"unk_pdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        return name
    elif isinstance(pdf, str):
        if pdf.startswith("http"):
            name = pdf.split("/")[-1].replace('.', '_')
            return name
        else:
            return Path(pdf).name

def bytes2io(bytes_data):
    return io.BufferedReader(io.BytesIO(bytes_data))





def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def img_to_html(img_path_ls:List[str],
                img_height:int=200,
                margin:int=10):
    prefix = f'<div style="display: flex; overflow-x: scroll; align-items: center; padding: 5px; height: {img_height}px;">'
    img_prefix = f'<div style="flex: 0 0 auto; margin-right: {margin}px; height: 100%; background: #fff; display: flex; justify-content: center; align-items: center;">'
    img_suffix = '</div>'
    res = prefix + '\n'
    for img_path in img_path_ls:
        img_html = f'<img src="{img_path}" style="height: 100%; object-fit: scale-down;" />'
        res +=  img_prefix + '\n' + img_html + '\n'
    res += img_suffix + '\n' + '</div>'
    return res



def display_markdown(md_path:str,
                     img_height:int=300,
                     margin:int=10):
    with open(md_path, 'r') as md_file:
        md_text = md_file.read()
    block_pattern = re.compile(r'<div.*?</div>\n+</div>', re.DOTALL)
    img_path_pattern = re.compile(r'src="(.+?)"')
    block_ls = block_pattern.findall(md_text)
    for block in block_ls:
        img_path_ls = img_path_pattern.findall(block)
        if img_path_ls:
            md_text = md_text.replace(block, img_to_html(img_path_ls,
                                                         dir_path=os.path.dirname(md_path),
                                                         img_height=img_height,
                                                         margin=margin))
    return md_text

def clean_img(text:str):
    block_pattern = re.compile(r'<div.*?</div>\n+</div>', re.DOTALL)
    blocks = block_pattern.findall(text)
    for block in blocks:
        text = text.replace(block, '')
    return text




def img2url(text:str,
             base_url:str,
             img_dir:str,
             absolute:bool = False):
    import re
    source_pattern = re.compile(r'src="(.+?)"', re.DOTALL)
    source_ls = source_pattern.findall(text)
    for source in source_ls:
        if source.startswith("http"):
            continue
        else:
            img_url = path2url(
                path=f"{img_dir}/{source}",
                base_url=base_url,
                absolute=absolute
            )
            text = text.replace(source,img_url)
    return text

def path2url(path:str,
             base_url:str,
             absolute:bool = False):
    if path.startswith("."):
        path = path[1:]
    base_url = base_url + "/index/"
    # input path is relative path
    if not absolute:
        path = path.replace('\\','/')
        img_url = urljoin(base_url,path)
    else:
        path = os.path.relpath(path,GENERAL_CONFIG['app_save_dir'])
        path = path.replace('\\','/')
        img_url = urljoin(base_url,path)
    return img_url

def is_allowed_file_type(file_path: str):
    allowed_file_types = GENERAL_CONFIG['allowed_file_types']
    file_type = osp.splitext(file_path)[1]
    return file_type in allowed_file_types


def avg_score(score_dir: Union[str, Path], num_dim: int = 6) -> list:
    total_counts = 0
    dim_totals = [0] * num_dim

    # Ensure the score directory is a Path object for easier path manipulations
    score_dir = Path(score_dir)

    for file_path in score_dir.glob("*.html"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        match = re.search(r'scoreData = (\[.*?\]);', text, re.DOTALL)
        print(f"match:{match},match.group(1):{match.group(1)}")
        if match:
            try:
                scores = eval(match.group(1))
            except Exception as e:
                logging.error(f"Error evaluating score data in file {file_path}: {e}")
                continue
            total_counts += 1
            for i in range(num_dim):
                dim_totals[i] += scores[i]["score"]

    # Prevent division by zero if no valid files were found
    if total_counts == 0:
        return [0] * num_dim

    avg_scores = [round(total / total_counts, 2) for total in dim_totals]
    return avg_scores





def torch_gc(CUDA_DEVICE="cuda:0"):
    import torch
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
