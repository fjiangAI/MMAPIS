import os
import os.path as osp
import io
import sys
import time
from datetime import datetime
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES
from MMAPIS.tools import ArxivCrawler,NougatPredictor, get_batch_size,download_pdf,YouDaoTTSConverter,extract_zip_from_bytes,zip_dir_to_bytes,init_logging,avg_score,img2url,is_allowed_file_type,path2url,bytes2io
from MMAPIS.server import Section_Summarizer,Summary_Integrator,img_txt_alignment,Paper_Recommender,Blog_Generator,BroadcastTTSGenerator,Regenerator,MultiModal_QA_Generator
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES,APPLICATION_PROMPTS, SECTION_PROMPTS,INTEGRATE_PROMPTS,TEMPLATE_DIR,TTS_CONFIG
from pathlib import Path
from fastapi.responses import ORJSONResponse,PlainTextResponse,FileResponse,Response,HTMLResponse,RedirectResponse
from fastapi.exceptions import RequestValidationError
from typing_extensions import Annotated
from fastapi import FastAPI,Body,File, UploadFile,Form,Depends
import json
from typing import List,Union,Literal,Dict
from pydantic import BaseModel,Field
import reprlib
import logging
import shutil
from fastapi.templating import Jinja2Templates
import uuid
from fastapi.staticfiles import StaticFiles
import tempfile
from urllib.parse import urljoin
import re
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
import asyncio


logger = init_logging(logger_mode=LOGGER_MODES)
app = FastAPI(title="MMAPIS",description="A Multi-Modal Automated Academic Papers Interpretation System",version="0.2.0")
gpu_semaphore = asyncio.Semaphore(1)

origins = [
    GENERAL_CONFIG['middleware_url'],
    GENERAL_CONFIG['frontend_url'],
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def gpu_lock():
    await gpu_semaphore.acquire()
    try:
        yield
    finally:
        gpu_semaphore.release()

def handle_pdf_content(pdf_content:UploadFile,
                       pdf:str,
                       save_dir:str,
                       temp_file:bool = False):
    if pdf_content:
        content = pdf_content.file.read()
        file_name = Path(pdf_content.filename).stem
        dir_name = os.path.join(save_dir,file_name)
        os.makedirs(dir_name,exist_ok=True)
        temp_time = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf_path = os.path.join(dir_name,f"{file_name}_{temp_time}.pdf")
        try:
            with open(pdf_path,'wb') as f:
                f.write(content)
            return True, pdf_path
        except Exception as e:
            return False, f"Write pdf content to {pdf_path} error: {e}"

    else:
        if pdf is None:
            return False,"No pdf file or pdf url found"
        else:
            flag, pdf_path = download_pdf(
                pdf_url=pdf,
                save_dir=save_dir,
                temp_file= temp_file
            )
            return flag, pdf_path



class ArxivRequest(BaseModel):
    key_word:Union[str,None,List[str]] = Field(None,example="graph neural network",description="search key word")
    searchtype:str = Field('all',example="all",description="search type")
    abstracts:str = Field('show',example="show",description="if show abstracts")
    order:str = Field("-announced_date_first",example="-announced_date_first",description="search order")
    size:int = Field(50,example=50,description="search size")
    max_return:int = Field(10,example=10,description="max number of return items")
    line_length:int = Field(15,example=15,description="display line length of abstracts in the front end")
    return_md: bool = Field(False,example=False,description="if return markdown")
    daily_type: str = Field("cs",example="cs",description="search daily type, only used when key_word is None, i.e. search new submissions")
    model_config = {"json_schema_extra": {
                        "examples": [
                            {
                            "description": "parameters for get arxiv links",
                            "key_word": "graph neural network",
                            "searchtype": "all",
                            "abstracts": "show",
                            "order": "-announced_date_first",
                            "size": 50,
                            "max_return": 10,
                            "line_length": 15,
                            "return_md": False,
                            "daily_type": "cs"
                            }
                        ]
                    }
                }


class gpt_model_config(BaseModel):
    model: str = OPENAI_CONFIG['model_config']['model']
    temperature:float = OPENAI_CONFIG['model_config']['temperature']
    max_tokens: int = OPENAI_CONFIG['model_config']['max_tokens']
    top_p:float = OPENAI_CONFIG['model_config']['top_p']
    frequency_penalty:float = OPENAI_CONFIG['model_config']['frequency_penalty']
    presence_penalty:float = OPENAI_CONFIG['model_config']['presence_penalty']
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for openai model",
                    "model": "gpt3.5-turbo-16k-0613",
                    "temperature": 0.9,
                    "max_tokens": 16385,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            ]
        }
    }



class Summarizer_Config(BaseModel):
    rpm_limit:int = OPENAI_CONFIG['rpm_limit']
    ignore_titles: Union[List, None] = OPENAI_CONFIG['ignore_title']
    prompt_ratio:float = OPENAI_CONFIG['prompt_ratio']
    num_processes:int = OPENAI_CONFIG['num_processes']
    gpt_model_params: Union[gpt_model_config, None] = gpt_model_config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rpm_limit": 3,
                    "ignore_titles": ["references", "appendix"],
                    "prompt_ratio": 0.8,
                    "num_processes": 10,
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

summarizer_config = Summarizer_Config()

class SectionSummaryRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",description="openai api key")
    base_url: str = Field(...,example="https://xxxxxx",description="base url")
    article_text:Union[str,None] = Field(...,example="# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n",description="article text need to be summarized")
    file_name:Union[str,None] = Field(None,example="xxxxxx",description="file name")
    init_grid: int = Field(3,example=3,description="initial grid of section summarizer, 3 means ### [Title]")
    max_grid: int = Field(4,example=4,description="max grid of section summarizer, 4 means #### [Title]")
    summary_prompts: Union[Dict, str] = SECTION_PROMPTS
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for section summarizer",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "artile_text": "# [Title] \n\n ## [Abstract] \n\n ## [Introduction] \n\n ## [Related Work] \n\n ## [Method] \n\n ## [Experiment] \n\n ## [Conclusion] \n\n ## [References] \n\n ## [Appendix] \n\n",
                    "file_name": "xxxxxx",
                    "init_grid": 2,
                    "max_grid": 4,
                    "summarizer_config": {
                        "requests_per_minute": 3,
                        "ignore_titles": ["references", "appendix"],
                        "prompt_factor": 0.8,  # prompt tokens / total tokens
                        "summary_prompts":
                            {
                                "system": "xxxx",
                                "abstract": "xxxx",
                                "introduction": "xxxx",
                                "related_work": "xxxx",
                                "method": "xxxx",
                                "experiment": "xxxx",
                                "conclusion": "xxxx",
                            },
                        "summarizer_params":
                            {
                                "rpm_limit": 3,
                                "ignore_titles": ["references", "appendix"],
                                "prompt_ratio": 0.8,
                                "num_processes": 10,
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
                }
            ]

        }
    }

class DocumentLevelSummaryRequest(BaseModel):
    api_key: str = Field(..., example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(..., example="https://xxxxxx")
    section_summaries: Union[str, List[str]] = Field(..., example="xxxxxx")
    integrate_prompts: dict = INTEGRATE_PROMPTS
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for document level summarizer",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "section_summaries": "xxxxxx",
                    "integrate_prompts":
                        {
                            "integrate": "xxxx",
                            "integrate_system": "xxxx",
                        },
                    "summarizer_params":
                        {
                            "rpm_limit": 3,
                            "ignore_titles": ["references", "appendix"],
                            "prompt_ratio": 0.8,
                            "num_processes": 10,
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


class RecommendationRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://api.ai-gaochao.cn/v1")
    document_level_summary: str = ...
    raw_md_text: str = ...
    score_prompts: dict = APPLICATION_PROMPTS["score_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for recommendation",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "document_level_summary": "xxxxxx",
                    "raw_text": "xxxxxx",
                    "score_prompts":
                        {
                            "score": "xxxx",
                            "score_system": "xxxx",
                        },
                    "summarizer_params":
                        {
                            "rpm_limit": 3,
                            "ignore_titles": ["references", "appendix"],
                            "prompt_ratio": 0.8,
                            "num_processes": 10,
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



class AlignmentRequest(BaseModel):
    text:Union[str,None] = ...
    pdf:Union[str,None] = None
    raw_md_text:Union[str,None] = None
    init_grid: int = ALIGNMENT_CONFIG['init_grid']
    max_grid: int = ALIGNMENT_CONFIG['max_grid']
    img_width: int = ALIGNMENT_CONFIG['img_width']
    threshold: float = ALIGNMENT_CONFIG['threshold']
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for alignment",
                    "text": "xxxxxx",
                    "pdf": "https://arxiv.org/pdf/xxxx.pdf",
                    "init_grid": 2,
                    "max_grid": 4,
                    "img_width": 500,
                    "threshold": 0.7
                }
            ]

        }
    }

class RegenerationRequest(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://api.ai-gaochao.cn/v1")
    document_level_summary: str = ...
    section_summaries: Union[str, List[str]] = ...
    regenerate_prompts: dict = APPLICATION_PROMPTS["regenerate_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for regeneration",
                    "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    "base_url": "https://xxxxxx",
                    "document_level_summary": "xxxxxx",
                    "section_summaries": "xxxxxx",
                    "regenerate_prompts":
                        {
                            "regenerate": "xxxx",
                            "regenerate_system": "xxxx",
                        },
                    "summarizer_params":
                        {
                            "rpm_limit": 3,
                            "ignore_titles": ["references", "appendix"],
                            "prompt_ratio": 0.8,
                            "num_processes": 10,
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    handle the validation error, transform the error message to a more readable string
    """
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


@app.post("/get_links/",response_class=ORJSONResponse,
          summary="crawl information from arxiv",
          description="get basic info(links, titles, abstract, authors) from arxiv",
          tags=["backend","preprocess"])
def fetch_links(link_request: ArxivRequest = Body(...)):
    arxiv_crawler = ArxivCrawler()

    if isinstance(link_request.key_word, list):
        link_request.key_word = " ".join(link_request.key_word)
    try:
        if link_request.key_word is None:
            articles = arxiv_crawler.run_daily_crawler(
                daily_type=link_request.daily_type,
                max_return=link_request.max_return,
                return_md=link_request.return_md,
                line_length=link_request.line_length,
            )
        else:
            articles = arxiv_crawler.run_keyword_crawler(
                key_word=link_request.key_word,
                max_return=link_request.max_return,
                return_md=link_request.return_md,
                searchtype=link_request.searchtype,
                abstracts=link_request.abstracts,
                order=link_request.order,
                line_length=link_request.line_length,
                size=link_request.size,
            )

        data = {
            "status": "success",
            "message":
                [{"pdf_url":article.pdf_url,
                  "title":article.title,
                  "author":article.authors,
                  "abstract":article.abstract}  for article in articles]

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




@app.post("/pdf2md/",response_class=ORJSONResponse,summary='model predict with nougat',
                                                          description="transfer the article format from pdf to txt with nougat model",
                                                          tags=["backend","preprocess"])
async def fetch_predictions(pdf:Union[str,List[str]] = Form(None,example="https://arxiv.org/pdf/xxxx.xxxx",description="pdf url or list of pdf urls, if pdf_content is not None, this param will be ignored"),
                      pdf_name:Union[str,List[str]] = Form(None,example="xxxx.xxxx"),
                      markdown:bool = Form(True,example=True),
                      pdf_content:List[UploadFile] = File(None,description="Multiple files as UploadFile"),
                      gpu_access: asyncio.Semaphore = Depends(gpu_lock)
                    ):
    async with gpu_access:
        nougat_preditor = NougatPredictor(markdown=markdown)
        if pdf_content:
            pdfs = [await pdf.read() for pdf in pdf_content]
            pdf_names = [pdf.filename for pdf in pdf_content] if pdf_name is None else pdf_name
            if isinstance(pdf_names,str):
                pdf_names = [pdf_names]
        else:
            if pdf is None:
                data = {
                    "status": "request error",
                    "message": "No pdf file or pdf url found"
                }
                return ORJSONResponse(content=data, status_code=400)
            else:
                pdfs = pdf if isinstance(pdf, List) else [pdf]
                pdf_names = pdf_name if isinstance(pdf_name, List) else [pdf_name] if pdf_name is not None else [None] * len(pdfs)
                assert len(pdf_names) == len(pdfs),f"pdf_names length {len(pdf_names)} not equal to pdfs length {len(pdfs)}"

        try:
            loop = asyncio.get_event_loop()
            article_ls = await loop.run_in_executor(None, nougat_preditor.pdf2md_text, pdfs, pdf_names)
            # article_ls = nougat_preditor.pdf2md_text(
            #     pdfs=pdfs,
            #     pdf_names=pdf_names)

            data = {
                "status": "success",
                "message": [
                    {"file_name": article.file_name,
                     "text": article.content} for article in article_ls
                ]
            }

            return ORJSONResponse(content=data, status_code=200)
        except Exception as e:
            logger.error('nougat_predict error: %s', e)
            error_message = f"An error occurred: {str(e)}"
            data = {
                "status": "nougat_predict internal error",
                "message": error_message
            }
            return ORJSONResponse(content=data, status_code=500)


@app.post("/alignment/",response_class=Response,summary='align text with image',
                                                          description="align the text with image based on title-like keywords",
                                                          tags=["backend","preprocess"])
def fetch_alignment(
                    text: Union[str, None] = Form(..., example="text", description="text to be aligned"),
                    pdf: Union[str, None] = Form(None, example="https://arxiv.org/pdf/xxxx.xxxx", description="pdf url, if pdf_content is not None, this param will be ignored"),
                    raw_md_text: Union[str, None] = Form(None, example="raw markdown text", description="raw markdown text of pdf, used for more accurate alignment"),
                    init_grid: int = Form(ALIGNMENT_CONFIG['init_grid'], example= 2),
                    max_grid: int = Form(ALIGNMENT_CONFIG['max_grid'], example= 4),
                    img_width: int = Form(ALIGNMENT_CONFIG['img_width'], example= 800),
                    threshold: float = Form(ALIGNMENT_CONFIG['threshold'], example= 0.8),
                    pdf_content:UploadFile = File(None,description="pdf bytes file as UploadFile")):
    flag, pdf_path = handle_pdf_content(pdf_content=pdf_content,
                                        pdf=pdf,
                                        save_dir=GENERAL_CONFIG['save_dir'])
    if not flag:
        data = {
            "status": "fetch pdf error in alignment",
            "message": pdf_path
        }
        return ORJSONResponse(content=data, status_code=400)
    try:
        pdf_path = Path(pdf_path)
        file_path = img_txt_alignment(
                                        text=text,
                                        pdf = pdf_path,
                                        save_dir=GENERAL_CONFIG['app_save_dir'],
                                        file_name=pdf_path.stem,
                                        raw_md_text=raw_md_text,
                                        init_grid=init_grid,
                                        max_grid=max_grid,
                                        img_width=img_width,
                                        threshold=threshold,
                                        temp_file=True
                                    )
        logging.info(f"saving alignment result to {file_path}")
        zip_file = zip_dir_to_bytes(dir_path=osp.dirname(file_path))
        shutil.rmtree(osp.dirname(file_path))
        logger.info(f"remove {osp.dirname(file_path)}")
        return Response(zip_file, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={Path(file_path).stem}.zip"})
    except Exception as e:
        error_message = f"alignment internal error: {str(e)}"
        logger.error(error_message)
        data = {
            "status": "alignment internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)
    finally:
        if osp.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"remove {pdf_path}")
            except Exception as e:
                logging.error(f"remove {pdf_path} error: {e}")


@app.post("/regeneration/",response_class=ORJSONResponse,summary='regenerate the article',
                                                          description="regenerate the article based on the alignment result",
                                                          tags=["backend","application"])
def fetch_regeneration(regeneration_request: RegenerationRequest = Body(...)):
    try:
        # regenerator = Regenerator(api_key=regeneration_request.api_key,
        #                           base_url=regeneration_request.base_url,
        #                           model_config=vars(regeneration_request.summarizer_params.gpt_model_params),
        #                           prompt_ratio=regeneration_request.summarizer_params.prompt_ratio,
        #                           )
        # flag, response = regenerator.regeneration(
        #     document_level_summary=regeneration_request.document_level_summary,
        #     section_summaries=regeneration_request.section_summaries,
        #     regeneration_prompts=regeneration_request.regenerate_prompts,
        #     response_only=True,
        #     reset_messages=True
        # )
        integrator = Summary_Integrator(api_key=regeneration_request.api_key,
                                  base_url=regeneration_request.base_url,
                                  model_config=vars(regeneration_request.summarizer_params.gpt_model_params),
                                  prompt_ratio=regeneration_request.summarizer_params.prompt_ratio,
                                        )
        flag, response = integrator.integrate_summary(section_summaries=regeneration_request.section_summaries,
                                                      integrate_prompts=regeneration_request.regenerate_prompts,
                                                      response_only=True,
                                                      reset_messages=True)
        if flag:
            data = {
                "status": "success",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=200)
        else:
            data = {
                "status": "regeneration error",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        logger.error('regeneration error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "regeneration internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)



@app.post("/section_level_summary/",response_class=ORJSONResponse,summary='summary with openai',
                                                          description="split the article into several groups, summarize each part with openai gpt and integrate the summaries into a whole summary blog",
                                                          tags=["backend","summarization"])
def fetch_section_summaries(summary_request: SectionSummaryRequest = Body(...)):
    try:
        logger.info(f'summary_request: {summary_request}')
        section_summarizer = Section_Summarizer(api_key=summary_request.api_key,
                                                base_url=summary_request.base_url,
                                                model_config=vars(summary_request.summarizer_params.gpt_model_params),
                                                proxy=GENERAL_CONFIG["proxy"],
                                                prompt_ratio= summary_request.summarizer_params.prompt_ratio,
                                                rpm_limit=summary_request.summarizer_params.rpm_limit,
                                                num_processes= OPENAI_CONFIG['num_processes'],
                                                ignore_titles= summary_request.summarizer_params.ignore_titles,
                                                )
        flag, res = section_summarizer.section_summarize(
            article_text=summary_request.article_text,
            file_name=summary_request.file_name,
            summary_prompts=summary_request.summary_prompts,
            init_grid=summary_request.init_grid,
            max_grid=summary_request.max_grid,
        )
        if flag:
            data = {
                "status": "success",
                "message": res
            }
        else:
            data = {
                "status": "request LLM api error",
                "message": res
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





@app.post("/document_level_summary/",response_class=ORJSONResponse,summary= "document level summary",
                                                            description="integrate the original sectional summary",
                                                            tags=["backend","summarization"])
def fetch_document_summary(document_level_request: DocumentLevelSummaryRequest = Body(...)):
    try:
        integrator = Summary_Integrator(api_key=document_level_request.api_key,
                                        base_url=document_level_request.base_url,
                                        model_config=vars(document_level_request.summarizer_params.gpt_model_params),
                                        prompt_ratio=document_level_request.summarizer_params.prompt_ratio,
                                        )
        flag, response = integrator.integrate_summary(section_summaries=document_level_request.section_summaries,
                                                      integrate_prompts=document_level_request.integrate_prompts,
                                                      response_only=True,
                                                      reset_messages=True)

        if flag:
            data = {
                "status": "success",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=200)
        else:
            data = {
                "status": "document_summary generation error",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        logger.error('document_summary error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "document_summary internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)


@app.post("/recommendation_generation/",response_class=ORJSONResponse,summary= "recommendation generation",
                                                            description="generate recommendation based on the document level summary",
                                                            tags=["backend","application"])
def fetch_recommendation(recommendation_request: RecommendationRequest = Body(...)):
    try:
        paper_recommender = Paper_Recommender(
            api_key=recommendation_request.api_key,
            base_url=recommendation_request.base_url,
            model_config=vars(recommendation_request.summarizer_params.gpt_model_params),
            proxy=GENERAL_CONFIG["proxy"],
            prompt_ratio = recommendation_request.summarizer_params.prompt_ratio,
        )
        flag, response = paper_recommender.recommendation_generation(
            document_level_summary=recommendation_request.document_level_summary,
            article=recommendation_request.raw_md_text,
            score_prompts=recommendation_request.score_prompts
        )
        if flag:
            avg_score_ls = [6] * len(response)
            for i, avgscore in enumerate(avg_score_ls):
                response[i]['avg_score'] = avgscore
            print("response",response)
            data = {
                "status": "success",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=200)
        else:

            data = {
                "status": "recommendation generation error",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        logger.error('recommendation_generation error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "recommendation_generation internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)



@app.post("/blog_generation/",response_class=ORJSONResponse,summary= "blog generation",
                                                            description="generate blog based on the document level summary",
                                                            tags=["backend","application"])

def fetch_blog(
               api_key: str = Form(...,example="api_key",description="api_key for openai"),
               base_url: str = Form(...,example="base_url",description="base_url for openai"),
               section_summary: Union[str, List[str]] = Form(..., example="section_summaries"),
               document_level_summary: str = Form(...,example="document_level_summary",description="document level summary of the article"),
               pdf: Union[str, None] = Form(None,example="https://www.example.com/sample.pdf",description="pdf url for the article, if pdf_content is not None, this will be ignored"),
               file_name: Union[str,None] = Form(None,example="blog",description="file name for the pdf"),
               raw_md_text: Union[str,None] = Form(None, example="raw_text",description="raw markdown text for the article pdf, used for more accurate alignment"),
               blog_prompts: str = Form(json.dumps(APPLICATION_PROMPTS["blog_prompts"]),example="'{'blog': 'xxx', 'blog_system': 'xxx'}'"),
               init_grid: int = Form(ALIGNMENT_CONFIG["init_grid"],example= 2),
               max_grid: int = Form(ALIGNMENT_CONFIG["max_grid"],example= 4),
               img_width: int = Form(ALIGNMENT_CONFIG["img_width"],example= 600,description="display width of the image"),
               threshold: float = Form(ALIGNMENT_CONFIG["threshold"],example= 0.8,description="threshold of similarity for alignment"),
               summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
               pdf_content: Union[UploadFile,None] = File(None,description="Multiple files as UploadFile")):
    if pdf_content:
        content = pdf_content.file.read()
        dir_name = os.path.join(GENERAL_CONFIG['save_dir'],Path(pdf_content.filename).stem)
        os.makedirs(dir_name,exist_ok=True)
        temp_time = datetime.now().strftime("%Y%m%d%H%M%S")
        pdf_path = os.path.join(dir_name, f"{temp_time}.pdf")
        with open(pdf_path,'wb') as f:
            f.write(content)
    else:
        if not pdf:
            data = {
                "status": "blog generation error",
                "message": "No pdf content"
            }
            return ORJSONResponse(content=data, status_code=500)
        else:
            flag, pdf_path = download_pdf(
                pdf_url=pdf,
                save_dir=GENERAL_CONFIG['save_dir'],
                temp_file=True
            )
            if not flag:
                data = {
                    "status": "download pdf error",
                    "message": f"download pdf from {pdf} failed"
                }
                return ORJSONResponse(content=data, status_code=400)
    summarizer_params = json.loads(summarizer_params)
    blog_prompts = json.loads(blog_prompts)
    blog_generator = Blog_Generator(
        api_key=api_key,
        base_url=base_url,
        model_config=summarizer_params["gpt_model_params"],
        proxy=GENERAL_CONFIG["proxy"],
        prompt_ratio=summarizer_params["prompt_ratio"],
    )
    try:
        flag, response = blog_generator.blog_generation(
            pdf=pdf_path,
            document_level_summary=document_level_summary,
            section_summaries=section_summary,
            raw_md_text=raw_md_text,
            blog_prompts=blog_prompts,
            init_grid=init_grid,
            max_grid=max_grid,
            img_width=img_width,
            threshold=threshold,
            file_name=file_name,
            temp_file=True,
            save_dir=GENERAL_CONFIG['save_dir']
        )
        path = Path(response)
        if flag:
            logging.info(f"saving blog to {path}")
            bytes_data = zip_dir_to_bytes(dir_path=osp.dirname(path))
            shutil.rmtree(osp.dirname(path))
            shutil.rmtree(osp.dirname(pdf_path))

            files = [("zip_content", bytes2io(bytes_data))]
            data = {
                "file_type": "blog_md",
            }
            response = requests.post(f"{GENERAL_CONFIG['middleware_url']}/upload_zip_file/", files=files, data=data)
            json_info = handle_api_response(response)
            if json_info["status"] == "success":
                markdown_content = json_info["message"]
                data = {
                    "status": "success",
                    "message": markdown_content
                }
                return ORJSONResponse(content=data, status_code=200)
            else:
                data = {
                    "status": "Saving markdown to middleware error",
                    "message": json_info["message"]
                }
                return ORJSONResponse(content=data, status_code=500)
        else:
            data = {
                "status": "blog generation error",
                "message": str(response)
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        logger.error('blog generation error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "blog generation internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)
    finally:
        if osp.exists(pdf_path):
            try:
                os.remove(pdf_path)
                logging.info(f"remove {pdf_path}")
            except Exception as e:
                logging.error(f"remove {pdf_path} error: {e}")



class BroadcastScriptRequest(BaseModel):
    llm_api_key: str
    llm_base_url: str
    document_level_summary: str
    section_summaries: Union[str, List[str]]
    broadcast_prompts: dict = APPLICATION_PROMPTS["broadcast_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()

class TTSRequest(BaseModel):
    text: str = Field(...,example="text",description="text to be converted to speech")
    model: str = Field(TTS_CONFIG['model'],example="youdao")
    api_key: str = Field(TTS_CONFIG['api_key'],example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(TTS_CONFIG['base_url'],example="https://xxxxxx")
    app_secret: str = Field(TTS_CONFIG['app_secret'],example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")



class Multimodal_QA_Request(BaseModel):
    api_key:str = Field(...,example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url: str = Field(...,example="https://xxxxxx")
    user_input: str = Field(...,example="user input")
    document_level_summary: str = Field(...,example="document level summary")
    session_message: List = Field(...,example="session message")
    article: str = Field(...,example="article")
    prompts: dict = Field(APPLICATION_PROMPTS["multimodal_qa"],example="{'qa': 'xxxx', 'qa_system': 'xxxx'}")
    init_grid: int = ALIGNMENT_CONFIG['init_grid']
    max_grid: int = ALIGNMENT_CONFIG['max_grid']
    ignore_titles: Union[List, None] = OPENAI_CONFIG['ignore_title']
    detailed_img: bool = False
    img_width: int = 300
    margin: int = 10
    prompt_ratio: float = 0.8
    summarizer_params: Summarizer_Config = Summarizer_Config()


    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "parameters for multimodal qa",
                    "multimodal_qa_generator": {
                        "description": "parameters for multimodal qa",
                        "api_key": "sk-xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                        "base_url": "https://xxxxxx",
                        "model_config": {
                            "description": "parameters for openai api",
                            "model": "gpt3.5-turbo-16k-0613",
                            "temperature": 0.9,
                            "max_tokens": 16385,
                            "top_p": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                        },
                        "prompt_ratio": 0.8,
                        "num_processes": 10,
                    }
                }
            ]

        }
    }

@app.post("/Multimodal_qa/",response_class=ORJSONResponse,summary= "Multimodal_qa",
                                                description="Multimodal_qa",
                                                tags=["backend","application"])
def Multimodal_qa(multimodal_qa_request:Multimodal_QA_Request = Body(...)):
    multimodal_qa_generator = MultiModal_QA_Generator(
        api_key=multimodal_qa_request.api_key,
        base_url=multimodal_qa_request.base_url,
        model_config=vars(multimodal_qa_request.summarizer_params.gpt_model_params),
        prompt_ratio=multimodal_qa_request.prompt_ratio,
        proxy=GENERAL_CONFIG["proxy"],
    )
    flag, answer = multimodal_qa_generator.chat(
        user_input=multimodal_qa_request.user_input,
        document_level_summary=multimodal_qa_request.document_level_summary,
        session_message=multimodal_qa_request.session_message,
        article=multimodal_qa_request.article,
        prompts=multimodal_qa_request.prompts,
        init_grid=multimodal_qa_request.init_grid,
        max_grid=multimodal_qa_request.max_grid,
        ignore_titles=multimodal_qa_request.ignore_titles,
        response_only=True,
        detailed_img= multimodal_qa_request.detailed_img,
        img_width=multimodal_qa_request.img_width,
        margin=multimodal_qa_request.margin,
    )
    if flag:
        data = {
            "status": "success",
            "message": answer
        }
        return ORJSONResponse(content=data, status_code=200)
    else:
        data = {
            "status": "Multimodal_qa error",
            "message": answer
        }
        return ORJSONResponse(content=data, status_code=500)



@app.post("/broadcast_generation/",response_class=ORJSONResponse,summary= "broadcast generation",
                                                description="generate broadcast based on the document level summary",
                                                tags=["backend","application"])
def Broadcast_script_generation(broadcast_request:BroadcastScriptRequest = Body(...)):
    broadcast_script_generator = BroadcastTTSGenerator(llm_api_key=broadcast_request.llm_api_key,
                                                        llm_base_url=broadcast_request.llm_base_url,
                                                       model_config=vars(broadcast_request.summarizer_params.gpt_model_params),
                                                       prompt_ratio=broadcast_request.summarizer_params.prompt_ratio,
                                                          )
    try:
        flag, response = broadcast_script_generator.broadcast_script_generation(
            document_level_summary=broadcast_request.document_level_summary,
            section_summaries=broadcast_request.section_summaries,
            broadcast_prompts=broadcast_request.broadcast_prompts
        )
        if flag:
            data = {
                "status": "success",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=200)
        else:
            data = {
                "status": "broadcast generation error",
                "message": response
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        logger.error('broadcast generation error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "broadcast generation internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)



@app.post("/tts/",response_class=ORJSONResponse,summary= "text to speech",
                                                description="generate broadcast based on the document level summary",
                                                tags=["backend","application"])
def fetch_tts(tts_request:TTSRequest = Body(...)):
    print("tts_request: ",tts_request.dict())
    if tts_request.model == "youdao":
        broadcast_tts_generator = BroadcastTTSGenerator(
                                                        llm_api_key=None,
                                                        llm_base_url=None,
                                                        tts_base_url=tts_request.base_url,
                                                        tts_api_key=tts_request.api_key,
                                                        app_secret=tts_request.app_secret,
                                                        tts_model= tts_request.model
                                                        )
    elif tts_request.model == "openai":
        tts_request.api_key = OPENAI_CONFIG['api_key']
        tts_request.base_url = OPENAI_CONFIG['base_url']
        broadcast_tts_generator = BroadcastTTSGenerator(
                                                        llm_api_key=tts_request.api_key,
                                                        llm_base_url=tts_request.base_url,
                                                        tts_model= tts_request.model
                                                        )
    else:
        return ORJSONResponse(content={"status": "tts model error", "message": "tts model not supported"}, status_code=400)
    print("broadcast_tts_generator: ",broadcast_tts_generator.tts_model)
    try:
        flag, bytes_data = broadcast_tts_generator.text2speech(
            text=tts_request.text,
            return_bytes=True
        )
        if flag:
            files = [("zip_content", bytes2io(bytes_data))]
            data = {
                "file_type": "mp3",
            }
            response = requests.post(
                GENERAL_CONFIG['middleware_url'] + "/upload_zip_file/",
                files=files, data=data
            )
            print("response: ",response.text)
            json_info = handle_api_response(response)
            if json_info["status"] == "success":
                mp3_url = json_info["message"]
            else:
                data = {
                    "status": "Saving mp3 error",
                    "message": json_info["message"]
                }
                return ORJSONResponse(content=data, status_code=500)
            data = {
                "status": "success",
                "message": mp3_url
            }
            return ORJSONResponse(content=data, status_code=200)

        else:
            data = {
                "status": "tts generation error",
                "message": bytes_data.decode("utf-8") if isinstance(bytes_data, bytes) else bytes_data
            }
            return ORJSONResponse(content=data, status_code=500)
    except Exception as e:
        logger.error('tts generation error: %s', e)
        error_message = f"An error occurred: {str(e)}"
        data = {
            "status": "tts generation internal error",
            "message": error_message
        }
        return ORJSONResponse(content=data,status_code=500)

class RedirectRequest(BaseModel):
    usage: str
    text:str
    bytes_content:bytes = None
    api_key:str = Field("",example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    base_url:str = Field("",example="https://xxxxxx")
    document_level_summary: str = Field("",example="document level summary")
    app_secret: str = Field("",example="xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

@app.post("/app/document_level_summary/",response_class=ORJSONResponse,summary= "document level summary",
                                                            description="generate document level summary",
                                                            tags=["Middleware","summarization"])
async def fetch_app_document_summary(
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        article:str = Form(...,example="article text"),
        pdf: Union[str, None] = Form(None, example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        file_name: str = Form(None, example="blog", description="file name for the pdf"),
        summary_prompts: str = Form(json.dumps(SECTION_PROMPTS), example="'{'system': 'xxx', 'abstract': 'xxx'}'"),
        integrate_prompts: str = Form(json.dumps(INTEGRATE_PROMPTS)),
        init_grid: int = Form(ALIGNMENT_CONFIG["init_grid"], example=2),
        max_grid: int = Form(ALIGNMENT_CONFIG["max_grid"], example=4),
        img_width: int = Form(ALIGNMENT_CONFIG["img_width"], example=400, description="display width of the image"),
        margin: int = Form(ALIGNMENT_CONFIG["margin"], example=10, description="margin of the image"),
        threshold: float = Form(ALIGNMENT_CONFIG["threshold"], example=0.8,
                                description="threshold of similarity for alignment"),
        summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
        pdf_content: UploadFile = File(None, description="Multiple files as UploadFile")):
    flag, pdf_path = handle_pdf_content(pdf_content = pdf_content,pdf= pdf,save_dir= os.path.join(GENERAL_CONFIG['app_save_dir'],"pdf"),temp_file=True)
    if not flag:
        data = {
            "status": "fetch pdf error in document-level summary generation",
            "message": pdf_path
        }
        return ORJSONResponse(content=data, status_code=400)

    summary_prompts = json.loads(summary_prompts)
    integrate_prompts = json.loads(integrate_prompts)
    summarizer_params = json.loads(summarizer_params)

    section_summarizer = Section_Summarizer(api_key=api_key,
                                            base_url=base_url,
                                            model_config=summarizer_params["gpt_model_params"],
                                            proxy=GENERAL_CONFIG["proxy"],
                                            prompt_ratio=summarizer_params["prompt_ratio"],
                                            rpm_limit=summarizer_params["rpm_limit"],
                                            num_processes=OPENAI_CONFIG["num_processes"],
                                            ignore_titles=summarizer_params["ignore_titles"],
                                            )
    flag, res = section_summarizer.section_summarize(
        article_text=article,
        file_name=file_name,
        summary_prompts=summary_prompts,
        init_grid=init_grid,
        max_grid=max_grid,
    )
    if flag:
        section_summmary = res
    else:
        data = {
            "status": "Section-level summary generation error",
            "message": res
        }
        return ORJSONResponse(content=data, status_code=500)

    integrator = Summary_Integrator(api_key=api_key,
                                    base_url=base_url,
                                    model_config=summarizer_params["gpt_model_params"],
                                    prompt_ratio=summarizer_params["prompt_ratio"],
                                    )
    flag, response = integrator.integrate_summary(section_summaries=section_summmary,
                                                  integrate_prompts=integrate_prompts,
                                                  response_only=True,
                                                  reset_messages=True)
    if not flag:
        data = {
            "status": "Document-level summary generation error",
            "message": response
        }
        return ORJSONResponse(content=data, status_code=500)
    else:
        pdf_path = Path(pdf_path)
        document_level_summary = response
        file_path = img_txt_alignment(
            text= document_level_summary,
            pdf= pdf_path,
            save_dir=osp.join(GENERAL_CONFIG['app_save_dir'],"document_alignment",pdf_path.stem),
            file_name=pdf_path.stem,
            raw_md_text= article,
            init_grid=init_grid,
            max_grid=max_grid,
            img_width=img_width,
            margin=margin,
            threshold=threshold,
        )
        dir_path = file_path.parent
        bytes_data = zip_dir_to_bytes(dir_path)
        try:
            shutil.rmtree(dir_path)
            shutil.rmtree(pdf_path.parent)
        except Exception as e:
            logger.error(f"remove {dir_path} & {pdf_path.parent} error: {e}")
        files = [("zip_content",bytes2io(bytes_data))]
        data = {
            "file_type": "blog_md"
        }
        response = requests.post(f"{GENERAL_CONFIG['middleware_url']}/upload_zip_file/",files=files,data=data)
        json_info = handle_api_response(response)
        if json_info["status"] == "success":
            markdown_content = json_info["message"]
        else:
            data = {
                "status": "Saving markdown to middleware error",
                "message": json_info["message"]
            }
            return ORJSONResponse(content=data, status_code=500)

        data = {
            "status": "success",
            "message": {
                "document_level_summary": document_level_summary,
                "section_level_summary": section_summmary,
                "document_level_summary_aligned": markdown_content
            }
        }
        return ORJSONResponse(content=data, status_code=200)



def filter_unsafe_markdown(text:str):
    text = re.sub(r'\$\{\}\^\{(.*?)\}\$', r'<sup>\1</sup>', text)
    return text

def create_new_doc(
        redir_request: RedirectRequest,
):
    temp_path = os.path.join(TEMPLATE_DIR, f"{redir_request.usage}.html")
    save_dir = osp.join(GENERAL_CONFIG['app_save_dir'], redir_request.usage+"_html")
    os.makedirs(save_dir, exist_ok=True)
    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    redir_request.text = filter_unsafe_markdown(redir_request.text)
    if redir_request.usage == "recommend":
        print("redir_request.text: ", redir_request.text,"type: ",type(redir_request.text))
        avg_score_ls = avg_score(score_dir=save_dir)
        redir_request.text = eval(redir_request.text)
        for i,avgscore in enumerate(avg_score_ls):
            redir_request.text[i]["avgscore"] = avgscore
        redir_request.text = str(redir_request.text)
    # elif redir_request.usage == "speech":
    #     html_content = html_content.replace("{{api_key}}", redir_request.api_key).replace("{{base_url}}", redir_request.base_url).replace("{{secret_key}}" , redir_request.app_secret)
    html_content = html_content.replace("{{text}}", redir_request.text)
    bytes_content = redir_request.bytes_content
    if bytes_content:
        html_content = html_content.replace("{{bytes}}", bytes_content.decode("utf-8"))
    if redir_request.usage == "qa":
        redir_request.document_level_summary = filter_unsafe_markdown(redir_request.document_level_summary)
        html_content = html_content.replace("{{document_level_summary}}", redir_request.document_level_summary).replace("{{base_url}}", OPENAI_CONFIG["base_url"]).replace("{{api_key}}", redir_request.api_key)
    if redir_request.usage in ["qa","speech"]:
        html_content = html_content.replace("{{backend_url}}", GENERAL_CONFIG["backend_url"])
    bytes_text = html_content.encode("utf-8")
    files = [("zip_content", bytes_text)]
    data = {
        "file_type": f"{redir_request.usage}_html"
    }
    response = requests.post(f"{GENERAL_CONFIG['middleware_url']}/upload_zip_file/", files=files, data=data)
    json_info = handle_api_response(response)
    if json_info["status"] == "success":
        redirect_url = json_info["message"]
    else:
        data = {
            "status": "Saving html to middleware error",
            "message": json_info["message"]
        }
        return ORJSONResponse(content=data, status_code=500)
    return redirect_url



def handle_api_response(
        response: Union[Response, ORJSONResponse],
):
    if isinstance(response, ORJSONResponse):
        json_info = json.loads(response.body.decode("utf-8"))
        if isinstance(json_info["message"], bytes):
            json_info["message"] = json_info["message"].decode("utf-8")
    elif isinstance(response, Response):
        json_info = {
            "status": "success",
            "message": response.body
        }
    else:
        try:
            json_info = response.json()
        except Exception as e:
            json_info = {
                "status": "Unsupport response error",
                "message": "Unsupport response type"
            }
    return json_info



@app.post("/app/regeneration/",
          response_class=RedirectResponse,
          summary= "regeneration",
          description="generate application based on the document level summary",
          tags=["Middleware","application"])
async def fetch_app(
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        document_level_summary: str = Form(...,example="document_level_summary",description="document level summary of the article"),
        usage:str = Form(...,example="blog",description="usage of the application, choices: ['blog', 'speech', 'regenerate','recommend','qa']"),
        raw_md_text:str = Form(None,example="article text",description="raw markdown text of the article"),
        section_summary: Union[str, List[str]] = Form(None, example="section_summaries"),
        prompts: str = Form(None, example="prompts", description="prompts for the application"),
        pdf: Union[str, None] = Form(None, example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        init_grid: int = Form(ALIGNMENT_CONFIG["init_grid"], example=2),
        max_grid: int = Form(ALIGNMENT_CONFIG["max_grid"], example=4),
        img_width: int = Form(ALIGNMENT_CONFIG["img_width"], example=600, description="display width of the image"),
        threshold: float = Form(ALIGNMENT_CONFIG["threshold"], example=0.8,
                                description="threshold of similarity for alignment"),
        summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
        pdf_content: UploadFile = File(None, description="Multiple files as UploadFile")):
    if usage == "regenerate":
        if not prompts:
            prompts = json.dumps(APPLICATION_PROMPTS["regenerate_prompts"])
        prompts = json.loads(prompts)
        regeneration_request = RegenerationRequest(
            api_key=api_key,
            base_url=base_url,
            document_level_summary=document_level_summary,
            section_summaries=section_summary,
            regenerate_prompts=prompts,
            summarizer_params=json.loads(summarizer_params)
        )
        response = fetch_regeneration(
            regeneration_request=regeneration_request
        )
        json_info = handle_api_response(response)
        if json_info["status"] == "success":
            regenerated_article = json_info["message"]
            regenerated_article = re.sub(r'(?<![\n#])(#+\s+.*?\n+)', r'\n\1', regenerated_article)
            save_dir = osp.join(GENERAL_CONFIG['app_save_dir'],"pdf")
            flag, pdf_path = handle_pdf_content(pdf_content=pdf_content, pdf=pdf,
                                                save_dir=save_dir, temp_file=True)
            if not flag:
                return ORJSONResponse(content={"status": "fetch pdf error in regeneration", "message": pdf_path}, status_code=400)
            pdf_path = Path(pdf_path)
            file_path = img_txt_alignment(
                text=regenerated_article,
                pdf=pdf_path,
                save_dir=osp.join(GENERAL_CONFIG['app_save_dir'],"regenerate_md",pdf_path.stem),
                file_name=pdf_path.stem,
                raw_md_text=raw_md_text,
                init_grid=init_grid,
                max_grid=max_grid,
                img_width=img_width,
                threshold=threshold,
            )
            dir_path = file_path.parent
            bytes_data = zip_dir_to_bytes(dir_path)
            try:
                shutil.rmtree(dir_path)
                shutil.rmtree(pdf_path.parent)
            except Exception as e:
                logger.error(f"remove {dir_path} & {pdf_path.parent} error: {e}")
            files = [("zip_content", bytes2io(bytes_data))]
            data = {
                "file_type": "regenerate_md"
            }
            response = requests.post(f"{GENERAL_CONFIG['middleware_url']}/upload_zip_file/", files=files, data=data)
            json_info = handle_api_response(response)
            if json_info["status"] == "success":
                markdown_content = json_info["message"]
                data = {
                    "status": "success",
                    "message": markdown_content
                }
            else:
                data = {
                    "status": "Saving markdown to middleware error",
                    "message": json_info["message"]
                }
                return ORJSONResponse(content=data, status_code=500)

            return ORJSONResponse(content=data, status_code=200)
        else:
            return ORJSONResponse(content=json_info, status_code=500)




@app.post("/redirect/",response_class=RedirectResponse,summary= "redirect",
                                                description="redirect to the application",
                                                tags=["Middleware","application"])

async def fetch_redirect(
        text: str = Body(..., example="text", description="text to be converted to speech"),
    ):
    redirect_url = f"/test_html/?text={text}"
    return RedirectResponse(url=redirect_url, status_code=303)

@app.post("/test_html/", response_class=HTMLResponse)
async def handle_form_submission(user_name: str = Form(...), text: str = Form(...),raw_md_text: str = Form(...)):
    # Generate the HTML content dynamically
    data = {
        "user_name": user_name,
        "text": len(text),
        "raw_md_text": len(raw_md_text)
    }
    print("data: ",data)
    return ORJSONResponse(content=data, status_code=200)


@app.post("/blog_test/",response_class=ORJSONResponse,summary= "blog generation")
async def generate_content(user_name: str = Body(...), raw_text: str = Body(...)):
    print("request: ",user_name,len(raw_text))
    # 这里可以添加更复杂的逻辑，例如数据库查询、文本生成等
    time.sleep(5)
    generated_content = {
        "status": "success",
        "message": f"Hello, {user_name}!\n\n {raw_text}"
    }
    print("success: ",user_name,len(raw_text))
    return ORJSONResponse(content=generated_content, status_code=200)



@app.post("/speech_test/",response_class=ORJSONResponse,summary= "blog generation")
async def generate_content(user_name: str = Body(...), raw_text: str = Body(...)):
    print("request: ",user_name,len(raw_text))
    # 这里可以添加更复杂的逻辑，例如数据库查询、文本生成等
    time.sleep(5)
    generated_content = {
        "status": "success",
        "message": f"Hello, {user_name}!\n\n {raw_text}"
    }
    print("success: ",user_name,len(raw_text))
    return ORJSONResponse(content=generated_content, status_code=200)


@app.post("/recommend_test/",response_class=ORJSONResponse,summary= "blog generation")
async def generate_content(user_name: str = Body(...), raw_text: str = Body(...)):
    print("request: ",user_name,len(raw_text))
    # 这里可以添加更复杂的逻辑，例如数据库查询、文本生成等
    time.sleep(5)
    generated_content = {
        "status": "success",
        "message": [{'title': 'Clarity of Objectives and Central Theme', 'score': 9, 'comments': 'The paper presents its objectives and central theme with high clarity, specifically focusing on the concept of cyclical learning rates (CLR) for neural network training. The significance and innovation of CLR are well-articulated, and the research questions are clearly defined.', 'avgscore': 8.19}, {'title': 'Appropriateness and Accuracy of Methods', 'score': 8, 'comments': 'The methods used, including the CLR strategy and various experiments conducted across different datasets and architectures, are appropriate and accurately described. They align well with the research objectives. However, a full score is withheld as the paper could benefit from more detailed justification for the chosen methods.', 'avgscore': 8.06}, {'title': 'Authenticity and Accuracy of Data and Findings', 'score': 8, 'comments': "Data and findings appear to be authentic and accurate, with empirical experiments supporting the claims made about CLR's efficacy. The paper makes a strong case for the reliability of its results through comparative analysis.", 'avgscore': 9.88}, {'title': 'Depth of Analysis and Conclusiveness', 'score': 7, 'comments': 'The analysis presented is thorough, leading to well-supported conclusions about the benefits of CLR. However, the paper acknowledges limitations and suggests areas for future research, indicating that while the conclusions are robust, they are not entirely conclusive for all architectures or scenarios.', 'avgscore': 7.56}, {'title': 'Overall Writing Quality', 'score': 8, 'comments': 'The excerpt provided demonstrates clear and concise writing with a well-structured abstract and conclusion. The language is academic and appropriate for the field, explaining complex concepts in an accessible manner. The paper effectively summarizes the research findings and their implications, suggesting further avenues for research. However, without the full paper, it is difficult to judge the consistency of quality throughout the entire document.', 'avgscore': 8.12}, {'title': 'Overall Score', 'score': 8.0, 'avgscore': 7.99}]
    }
    print("success: ",user_name,len(raw_text))
    return ORJSONResponse(content=generated_content, status_code=200)