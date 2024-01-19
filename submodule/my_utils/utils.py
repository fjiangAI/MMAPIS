import  torch
from functools import singledispatch
from typing import List
import io
from pathlib import Path
import fitz
import logging
import zipfile
import sys
import requests
from datetime import datetime


def get_best_gpu(choice_list=None):
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
    if torch.cuda.is_available():
        best_gpu, free_memory = get_best_gpu([0, 1])
        BATCH_SIZE = int(
            free_memory * 0.3
        )
        # BATCH_SIZE = int(
        #     torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
        # )
        logging.info(f"Best GPU: {best_gpu}. Batch size: {BATCH_SIZE}")
        if BATCH_SIZE < 1:
            logging.error("Not enough GPU memory, can not load model.")
            sys.exit(1)
    else:
        # don't know what a good value is here. Would not recommend to run on CPU
        BATCH_SIZE = 5
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
        logging.info(f"Opening PDF to rasterize from url format")
        name = pdf.split("/")[-1].replace('.', '_')
        # self.size = len(fitz.open(stream=urllib.request.urlopen(pdf).read(), filetype="pdf"))
        pdf_doc_obj = fitz.open(stream=requests.get(pdf, proxies=proxy, headers=headers).content, filetype="pdf")
        size = len(pdf_doc_obj)

    else:
        logging.info(f"Opening PDF to rasterize from str format")
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
                return json_info['message']
            elif response.status_code == 400:
                logging.error(f"request body error[{func_name}], status: {json_info.get('status', response.status_code)}")
            elif response.status_code == 500:
                logging.error(f"internal error[{func_name}], status: {json_info.get('status', response.status_code)}")
            else:
                logging.error(f"unknown response error[{func_name}], status code: {response.status_code}")
            error_msg =  json_info.get('status', '') + " "+ json_info.get('message', 'Unknown error')
            error_status = json_info.get('status', response.status_code)
            json_msg = {'error': error_msg, 'status': error_status}
            return json_msg

        elif 'audio/mp3' in response.headers['content-type']:
            logging.info(f"{func_name} process success")
            return response.content

        else:
            logging.error(f"unknown response type:{response.headers.get('content-type','unknown')} error[{func_name}], status code: {response.status_code}")
            error_msg = response.text
            return {'error': error_msg, 'status': response.status_code}

    except Exception as e:
        logging.error(f"response error[{func_name}], an unexpected error occurred: {str(e)}")
        return {'error': 'Unknown error', 'status': response.status_code if response else 502}


def dict_filter_none(d:dict):
    return {k:v for k,v in d.items() if v is not None}


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
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
    try:
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
        encoding = tiktoken.get_encoding(encode_model)
    except KeyError:
        logging.error(f"model {model} not found,load default model: cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in "gpt-3.5-turbo-16k-0613":  # note: future models may deviate from this
        if isinstance(messages, dict):
            messages = [messages]
        elif isinstance(messages, list):
            pass
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

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