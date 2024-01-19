import io

from fastapi import FastAPI,Body,File,Form
import json
from typing import List,Union,Literal,Dict
from pydantic import BaseModel,Field
from submodule.arxiv_links import *
from submodule.openai_api import *
from submodule.nougat_main import *
from submodule.my_utils import *
import logging
from pathlib import Path
from submodule.my_utils import init_logging
from fastapi.responses import ORJSONResponse,PlainTextResponse,FileResponse
from fastapi.exceptions import RequestValidationError
from typing_extensions import Annotated


logger = init_logging('logging.ini')


app = FastAPI(title="Arxiv Summarizer",description="A summarizer for arxiv papers with nougat and openai api",version="0.1.0")
file_path = 'prompts_config_zh.json'
gpt_prompts = json.load(open(file_path, 'r', encoding='utf-8'))

class LinkRequest(BaseModel):
    key_word:Union[str,None,List[str]] = Field(None,example="graph neural network")
    proxies:Union[dict,None] = None
    headers:Union[dict,None] = None
    max_num:int = 5
    line_length:int = 15
    searchtype:str = "all"
    abstracts:str = "show"
    order:str = "-announced_date_first"
    size:int = 50
    show_meta_data:bool = True
    daily_type:str = "cs"
    max_retry:int = 3
    wait_fixed:int = 1000
    model_config  = {"json_schema_extra": {
                        "examples": [
                            {
                            "description": "parameters for get arxiv links",
                            "key_word": "graph neural network",
                            "proxies": None,
                            "headers": None,
                            "max_num": 5,
                            "line_length": 15,
                            "searchtype": "all",
                            "abstracts": "show",
                            "order": "-announced_date_first",
                            "size": 50,
                            "show_meta_data": True,
                            "daily_type": "cs",
                            "max_retry": 3,
                            "wait_fixed": 1000
                            }
                        ]
                    }
                }


class basic_model_info(BaseModel):
    model: str = 'gpt-3.5-turbo-16k-0613'
    temperature:float = 0.9
    max_tokens: int = 16385
    top_p:float = 1
    frequency_penalty:float = 0
    presence_penalty:float = 0



class Summarizer_Config(BaseModel):
    requests_per_minute: Union[int, None] = None
    proxy: Union[dict, None] = None
    summary_prompts: Union[dict, str] = gpt_prompts['section summary']
    resummry_prompts: Union[dict, str] = gpt_prompts["blog summary"]
    ignore_titles: Union[List, None] = ["references", "appendix"]
    acquire_mode: Literal['url', 'openai'] = 'url'
    prompt_factor: float = 0.8  # prompt tokens / total tokens
    num_processes: int = 10
    base_url: str = 'https://api.ai-gaochao.cn/v1'
    split_mode: str = 'group'
    gpt_config: Union[basic_model_info, None] = None
    def __init__(self, **data):
        if data.get("gpt_config", None) is None:
            data["gpt_config"] = basic_model_info()
        super().__init__(**data)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "proxy": {"http": "http://xxxxxx:xxxx"},
                    "requests_per_minute": 3,
                    "summary_prompts": {"system": "xxxxxx"},
                    "resummry_prompts": {"system": "xxxxxx"},
                    "ignore_titles": ["references", "appendix"],
                    "acquire_mode": "url",
                    "prompt_factor": 0.8,  # prompt tokens / total tokens
                    "num_processes": 10,
                    "base_url": "https://api.ai-gaochao.cn/v1",
                    "split_mode": "group",
                    "gpt_config":
                        {
                            "description": "parameters for openai api",
                            "model": "gpt3.5-turbo-16k-0613",
                            "temperature": 0.9,
                            "max_tokens": 16385,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,

                        }
                }
            ]

        }
    }

class SummaryRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    artile_text:Union[str,None] = ...
    file_name:Union[str,None] = ...
    init_grid: int = 2
    summarizer_config: Summarizer_Config = Summarizer_Config()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for get arxiv links",
                    "api_key": "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "artile_text": "graph neural network",
                    "file_name": "xxxxxx",
                    "init_grid": 2,
                    "summarizer_config": {
                        "proxy": {"http": "http://xxxxxx:xxxx"},
                        "requests_per_minute": 3,
                        "summary_prompts": {"system": "xxxxxx"},
                        "resummry_prompts": {"system": "xxxxxx"},
                        "ignore_titles": ["references", "appendix"],
                        "acquire_mode": "url",
                        "prompt_factor": 0.8,  # prompt tokens / total tokens
                        "num_processes": 10,
                        "base_url": "https://api.ai-gaochao.cn/v1",
                        "split_mode": "group",
                        "gpt_config":
                            {
                                "description": "parameters for openai api",
                                "model": "gpt3.5-turbo-16k-0613",
                                "temperature": 0.9,
                                "max_tokens": 16385,
                                "top_p": 1,
                                "frequency_penalty": 0,
                                "presence_penalty": 0,
                            }
                    }
                }
            ]

        }
    }


class EnhanceRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    original_answer:str = Field(...,example="hello world")
    summarized_answer:str = Field(...,example="hello world")
    usage:Literal['regenerate','blog','speech',None] = None
    summarizer_config: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for enhance answer",
                    "original_answer": "hello world",
                    "summarized_answer": "hello world",
                    "usage": "regenerate",
                    "summarizer_config": {
                        "proxy": {"http": "http://xxxxxx:xxxx"},
                        "requests_per_minute": 3,
                        "summary_prompts": {"system": "xxxxxx"},
                        "resummry_prompts": {"system": "xxxxxx"},
                        "ignore_titles": ["references", "appendix"],
                        "acquire_mode": "url",
                        "prompt_factor": 0.8,  # prompt tokens / total tokens
                        "num_processes": 10,
                        "base_url": "https://api.ai-gaochao.cn/v1",
                        "split_mode": "group",
                        "gpt_config":
                            {
                                "description": "parameters for openai api",
                                "model": "gpt3.5-turbo-16k-0613",
                                "temperature": 0.9,
                                "max_tokens": 16385,
                                "top_p": 1,
                                "frequency_penalty": 0,
                                "presence_penalty": 0,
                            }
                    }
                }
            ]
        }
    }




class Args(BaseModel):
    batchsize:int = get_batch_size()
    checkpoint:Union[str,Path] = "./pretrained_w"
    out:Union[str,Path] = './res'
    recompute:bool = False
    markdown:bool = True
    kw:str = None
    pdf:Union[List[str],List[Path],str,Path] = None


class PredictRequest(BaseModel):
    args:Args = Field(...,arbitrary_types_allowed=True)
    proxy:Union[Dict,None] = None
    headers:Union[Dict,None] = None
    pdf_name:Union[str,None] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for nougat predict",
                    "args": {
                        "recompute": False,
                        "markdown": True,
                        "kw": "graph neural network",
                        "pdf": "https://arxiv.org/pdf/xxxx.pdf",
                    },
                    "proxy": None,
                    "headers": None
                }
            ]
        }
    }



class TextRequest(BaseModel):
    text:str = Field(...,example="hello world")
    app_key:str = '21a52d8cbabd6256'
    app_secret:str = 'MYttHedJuswCG6CNoyhd5YWH5egzQPSa'
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for text to voice",
                    "text": "hello world",
                    "app_key": "xxxxxx",
                    "app_secret": "xxxxxx"
                }
            ]
        }
    }

def str2path(path_l:Union[str,List[Path]]):
    if not isinstance(path_l,list):
        path_l = [path_l]
    res = []
    for path in path_l:
        if 'http' in path:
            res.append(path)
        else :
            res.append(Path(path))
    if len(path_l) == 1:
        return res[0]
    else:
        return res

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f'validation_exception_handler: {exc}')
    error_infos = exc.errors()
    if isinstance(error_infos,list):
        error_response = "input params error: \n"
        try:
            for i,error_info in enumerate(error_infos):
                error_response += f"error {i+1}:"
                error_response += f"type: {error_info['type']}, location: request {error_info['loc'][0]}, param {error_info['loc'][1] if len(error_info['loc']) >1 else 'Unknow param'} input:{error_info.get('input','None')}, msg: {error_info['msg']}\n"
            data = {
                "status": "request error",
                "message": error_response
            }
            return ORJSONResponse(content=data, status_code=400)
        except Exception as e:
            data = {
                "status": "request error",
                "message": str(e)
            }
            return ORJSONResponse(content=data, status_code=400)
    else:
        data = {
            "status": "request error",
            "message": str(exc)
        }
        return ORJSONResponse(content=data, status_code=400)


@app.post("/get_links/",response_class=ORJSONResponse,summary="get arxiv links",description="get basic info(links, titles, abstract, authors) in markdown format from arxiv",tags=["get_links"])
def fetch_links(link_request: LinkRequest = Body(...)):
    if link_request is None:
        link_request = LinkRequest()
    if isinstance(link_request.key_word, list):
        link_request.key_word = " ".join(link_request.key_word)
    logger.info(f'link_request: {link_request}')
    key_word = link_request.key_word if link_request.key_word else None

    try:
        links, titles, abstract, authors = get_arxiv_links(key_word=key_word,
                                                        proxies=link_request.proxies,
                                                        max_num=link_request.max_num,
                                                        line_length=link_request.line_length,
                                                        searchtype=link_request.searchtype,
                                                        abstracts=link_request.abstracts,
                                                        order=link_request.order,
                                                        size=link_request.size,
                                                        show_meta_data=link_request.show_meta_data,
                                                        daily_type= link_request.daily_type,
                                                        headers=link_request.headers,
                                                        max_retry=link_request.max_retry,
                                                        wait_fixed=link_request.wait_fixed)
        data = {
            "status": "success",
            "message": {"links": links,
                        "titles": titles,
                        "abstract": abstract,
                        "authors": authors}
        }
        return ORJSONResponse(content=data,status_code=200)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error('get_arxiv_links error: %s', e)
        data = {
            "status": "fetch links internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data, status_code=500)




@app.post("/model_predict/",response_class=ORJSONResponse,summary='model predict with nougat',
                                                          description="transfer the article format from pdf to txt with nougat model",
                                                          tags=["model_predict"])
def fetch_predictions(predict_request: PredictRequest = Body(...)):

    predict_request.args.batchsize = get_batch_size()
    predict_request.args.checkpoint = "./pretrained_w"
    predict_request.args.out,predict_request.args.pdf,predict_request.args.checkpoint = str2path(predict_request.args.out),str2path(predict_request.args.pdf),str2path(predict_request.args.checkpoint)
    if not isinstance(predict_request.args.pdf,list):
        predict_request.args.pdf = [predict_request.args.pdf]
    logger.info(f'predict_request:{predict_request}')
    try :
        article_ls,file_names = nougat_predict(
                                args = predict_request.args,
                                proxy = predict_request.proxy,
                                headers = predict_request.headers,
                                pdf_name=predict_request.pdf_name,
                            )
        data = {
            "status": "success",
            "message": {"article_ls": article_ls,
                        "file_names": file_names}
        }


        return ORJSONResponse(content=data,status_code=200)

    except Exception as e:
        logger.error('nougat_predict error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "nougat_predict internal error",
            "message": error_message
        }

        return ORJSONResponse(content=data,status_code=500)


@app.post("/bytes_predict/",response_class=ORJSONResponse,summary='bytes predict with nougat',
                                                            description="transfer the article format from pdf to txt with nougat model",
                                                            tags=["bytes_predict"])
async def fetch_bytes_predictions(pdf_content: bytes = File(...,examples=[b"pdf_content"]),
                      pdf_name:Union[str,None] = Form(None,examples=["pdf_name"])):
    """
    Upload pdf file
    """
    predict_request = PredictRequest(args=Args(),proxy=None,headers=None,pdf_name=pdf_name)
    predict_request.args.pdf = [pdf_content]

    logger.info(f'predict_request:{predict_request.pdf_name}')
    try :
        article_ls,file_names = nougat_predict(
                                args = predict_request.args,
                                proxy = predict_request.proxy,
                                headers = predict_request.headers,
                                pdf_name=predict_request.pdf_name,
                            )
        data = {
            "status": "success",
            "message": {"article_ls": article_ls,
                        "file_names": file_names}
        }


        return ORJSONResponse(content=data,status_code=200)

    except Exception as e:
        logger.error('nougat_predict error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "nougat_predict internal error",
            "message": error_message
        }

        return ORJSONResponse(content=data,status_code=500)


@app.post("/get_summaries/",response_class=ORJSONResponse,summary='summary with openai',
                                                          description="split the article into several groups, summarize each part with openai gpt and integrate the summaries into a whole summary blog",
                                                          tags=["summary"])
def fetch_summaries(summary_request: SummaryRequest = Body(...)):
    try:
        logger.info(f'summary_request: {summary_request}')
        summarizer = OpenAI_Summarizer(
                api_key=summary_request.api_key,
                proxy=summary_request.summarizer_config.proxy,
                requests_per_minute=summary_request.summarizer_config.requests_per_minute,
                acquire_mode=summary_request.summarizer_config.acquire_mode,
                prompt_factor=summary_request.summarizer_config.prompt_factor,
                summary_prompts=summary_request.summarizer_config.summary_prompts,
                resummry_prompts=summary_request.summarizer_config.resummry_prompts,
                split_mode=summary_request.summarizer_config.split_mode,
                ignore_titles=summary_request.summarizer_config.ignore_titles,
                num_processes=summary_request.summarizer_config.num_processes,
                base_url=summary_request.summarizer_config.base_url,
                model_info={
                    "model": summary_request.summarizer_config.gpt_config.model,
                    "temperature": summary_request.summarizer_config.gpt_config.temperature,
                    "max_tokens": summary_request.summarizer_config.gpt_config.max_tokens,
                    "top_p": summary_request.summarizer_config.gpt_config.top_p,
                    "frequency_penalty": summary_request.summarizer_config.gpt_config.frequency_penalty,
                    "presence_penalty": summary_request.summarizer_config.gpt_config.presence_penalty,
                }
        )
        titles, authors, affiliations, total_resp, re_respnse = summarizer.summary_with_openai(
            artile_text=summary_request.artile_text,
            file_name=summary_request.file_name,
            init_grid=summary_request.init_grid,
        )
        data = {
            "status": "success",
            "message": {"titles": titles,
                        "authors": authors,
                        "affiliations": affiliations,
                        "total_resp": total_resp,
                        "re_respnse": re_respnse}
        }
        return ORJSONResponse(content=data, status_code=200)
    except Exception as e:
        logger.error(f'summary_with_openai error: {e}')
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "summary internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data, status_code=500)


@app.post("/text2audio/",response_class=FileResponse,summary='text to voice',
                                                            description="transfer text to voice",
                                                            tags=["text2voice"])
def fetch_voice(text_request: TextRequest = Body(...)):
    logger.info(f'text:{text_request}')
    text = text_request.text
    try:
        flag, file_content = text_to_speech(text)
        print('type:',type(file_content))
        if flag:
            return FileResponse(path=file_content,media_type="audio/mp3")
        else:
            data = {
                "status": "text_to_speech response type error",
                "message": file_content
            }
            return ORJSONResponse(content=data,status_code=500)
    except Exception as e:
        logger.error('text_to_speech error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "text_to_speech internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)


@app.post("/enhance_answer/",response_class=ORJSONResponse,summary='enhance answer',
                                                            description="enhance answer with openai",
                                                            tags=["enhance_answer"])
def fetch_enhance_answer(enhance_request: EnhanceRequest = Body(...)):
    logger.info(f'enhance_request:{enhance_request}')
    original_answer = enhance_request.original_answer
    summarized_answer = enhance_request.summarized_answer
    usage = enhance_request.usage
    summerizer = OpenAI_Summarizer(
        api_key=enhance_request.api_key,
        proxy=enhance_request.summarizer_config.proxy,
        requests_per_minute=enhance_request.summarizer_config.requests_per_minute,
        acquire_mode=enhance_request.summarizer_config.acquire_mode,
        prompt_factor=enhance_request.summarizer_config.prompt_factor,
        summary_prompts=enhance_request.summarizer_config.summary_prompts,
        resummry_prompts=enhance_request.summarizer_config.resummry_prompts,
        split_mode=enhance_request.summarizer_config.split_mode,
        ignore_titles=enhance_request.summarizer_config.ignore_titles,
        num_processes=enhance_request.summarizer_config.num_processes,
        base_url=enhance_request.summarizer_config.base_url,
        model_info={
            "model": enhance_request.summarizer_config.gpt_config.model,
            "temperature": enhance_request.summarizer_config.gpt_config.temperature,
            "max_tokens": enhance_request.summarizer_config.gpt_config.max_tokens,
            "top_p": enhance_request.summarizer_config.gpt_config.top_p,
            "frequency_penalty": enhance_request.summarizer_config.gpt_config.frequency_penalty,
            "presence_penalty": enhance_request.summarizer_config.gpt_config.presence_penalty,
        }
    )

    try:
        enhanced_answer = summerizer.Enhance_Answer(
                                         original_answer= original_answer,
                                         summarized_answer= summarized_answer,
                                         usage=usage)
        # enhanced_answer = usage +'\n'  + summarized_answer
        data = {
            "status": "success",
            "message": {"enhanced_answer": enhanced_answer}
        }
        return ORJSONResponse(content=data,status_code=200)

    except Exception as e:
        logger.error('enhance_answer error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "enhance_answer internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)