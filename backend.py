import os
import os.path as osp
import io
import sys
from datetime import datetime
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES
from MMAPIS.tools import ArxivCrawler,NougatPredictor, get_batch_size,download_pdf,YouDaoTTSConverter,extract_zip_from_bytes,zip_dir_to_bytes,init_logging
from MMAPIS.server import Section_Summarizer,Summary_Integrator,img_txt_alignment,Paper_Recommender,Blog_Generator,BroadcastTTSGenerator
from MMAPIS.config.config import GENERAL_CONFIG,APPLICATION_PROMPTS, ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES,APPLICATION_PROMPTS, SECTION_PROMPTS,INTEGRATE_PROMPTS
from pathlib import Path
from fastapi.responses import ORJSONResponse,PlainTextResponse,FileResponse,StreamingResponse,Response
from fastapi.exceptions import RequestValidationError
from typing_extensions import Annotated
from fastapi import FastAPI,Body,File, UploadFile,Form
import json
from typing import List,Union,Literal,Dict
from pydantic import BaseModel,Field
import reprlib
import logging
import shutil






logger = init_logging(logger_mode=LOGGER_MODES)
app = FastAPI(title="MMAPIS",description="A Multi-Modal Automated Academic Papers Interpretation System",version="0.1.0")

class ArxivRequest(BaseModel):
    key_word:Union[str,None,List[str]] = Field(None,example="graph neural network",description="search key word")
    searchtype:str = Field('all',example="all",description="search type")
    abstracts:str = Field('show',example="show",description="if show abstracts")
    order:str = Field("-announced_date_first",example="-announced_date_first",description="search order")
    size:str = Field(50,example=50,description="search size")
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
    rpm_limit:int = 3
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
    raw_text: str = ...
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
          tags=["preprocess"])
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
                                                          tags=["preprocess"])
def fetch_predictions(pdf:Union[str,List[str]] = Form(None,example="https://arxiv.org/pdf/xxxx.xxxx",description="pdf url or list of pdf urls, if pdf_content is not None, this param will be ignored"),
                      pdf_name:Union[str,None] = Form(None,example="xxxx.xxxx"),
                      markdown:bool = Form(True,example=True),
                      pdf_content:List[UploadFile] = File(None,description="Multiple files as UploadFile")):
    nougat_preditor = NougatPredictor(markdown=markdown)
    if pdf_content:
        pdfs = [pdf.file.read() for pdf in pdf_content]
        pdf_names = [pdf.filename for pdf in pdf_content]

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

    try :
        article_ls = nougat_preditor.pdf2md_text(
            pdfs=pdfs,
            pdf_names=pdf_names)

        data = {
            "status": "success",
            "message": {"article_ls": [
                {"file_name": article.file_name,
                 "text": article.content} for article in article_ls
            ]}
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


@app.post("/alignment/",response_class=StreamingResponse,summary='align text with image',
                                                          description="align the text with image based on title-like keywords",
                                                          tags=["preprocess"])
def fetch_alignment(
                    text: Union[str, None] = Form(..., example="text", description="text to be aligned"),
                    pdf: Union[str, None] = Form(None, example="https://arxiv.org/pdf/xxxx.xxxx", description="pdf url, if pdf_content is not None, this param will be ignored"),
                    raw_md_text: Union[str, None] = Form(None, example="raw markdown text", description="raw markdown text of pdf, used for more accurate alignment"),
                    init_grid: int = Form(ALIGNMENT_CONFIG['init_grid'], example= 2),
                    max_grid: int = Form(ALIGNMENT_CONFIG['max_grid'], example= 4),
                    img_width: int = Form(ALIGNMENT_CONFIG['img_width'], example= 800),
                    threshold: float = Form(ALIGNMENT_CONFIG['threshold'], example= 0.8),
                    pdf_content:UploadFile = File(None,description="pdf bytes file as UploadFile")):
    if pdf_content:
        content = pdf_content.file.read()
        dir_name = os.path.join(GENERAL_CONFIG['save_dir'],Path(pdf_content.filename).stem)
        pdf_path = os.path.join(dir_name,pdf_content.filename)
        with open(pdf_path,'wb') as f:
            f.write(content)

    else:
        if pdf is None:
            data = {
                "status": "request error",
                "message": "No pdf file or pdf url found"
            }
            return ORJSONResponse(content=data, status_code=400)
        else:
            flag, pdf_path = download_pdf(
                pdf_url=pdf,
                save_dir=GENERAL_CONFIG['save_dir']
            )
            if not flag:
                data = {
                    "status": "download pdf error",
                    "message": f"download pdf from {pdf} failed"
                }
                return ORJSONResponse(content=data, status_code=400)
    pdf_path = Path(pdf_path)
    file_path = img_txt_alignment(
                                    text=text,
                                    pdf = pdf_path,
                                    save_dir=osp.join(GENERAL_CONFIG['save_dir'],pdf_path.stem),
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
    return StreamingResponse(io.BytesIO(zip_file), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={Path(file_path).stem}.zip"})




@app.post("/section_level_summry/",response_class=ORJSONResponse,summary='summary with openai',
                                                          description="split the article into several groups, summarize each part with openai gpt and integrate the summaries into a whole summary blog",
                                                          tags=["summarization"])
def fetch_section_summaries(summary_request: SectionSummaryRequest = Body(...)):
    try:
        logger.info(f'summary_request: {summary_request}')
        section_summarizer = Section_Summarizer(api_key=summary_request.api_key,
                                                base_url=summary_request.base_url,
                                                model_config=vars(summary_request.summarizer_params.gpt_model_params),
                                                proxy=GENERAL_CONFIG["proxy"],
                                                prompt_ratio= summary_request.summarizer_params.prompt_ratio,
                                                rpm_limit=summary_request.summarizer_params.rpm_limit,
                                                num_processes= summary_request.summarizer_params.num_processes,
                                                ignore_titles= summary_request.summarizer_params.ignore_titles,
                                                )
        res = section_summarizer.section_summarize(
            article_text=summary_request.article_text,
            file_name=summary_request.file_name,
            summary_prompts=summary_request.summary_prompts,
            init_grid=summary_request.init_grid,
            max_grid=summary_request.max_grid,
        )

        data = {
            "status": "success",
            "message": {"section_summary": res}
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


# @app.post("/text2audio/",response_class=FileResponse,summary='text to voice',
#                                                             description="transfer text to voice",
#                                                             tags=["text2voice"])
# def fetch_voice(text_request: TTSRequest = Body(...)):
#     logger.info(f'text:{text_request}')
#     try:
#         youdao_tts = YouDaoTTSConverter(
#             base_url=text_request.base_url,
#             api_key=text_request.api_key,
#             app_secret=text_request.app_secret,
#             proxy=GENERAL_CONFIG["proxy"]
#         )
#
#         flag, file_content = youdao_tts.convert_texts_to_speech(
#             text=text_request.text,
#             num_processes=text_request.num_processes,
#             return_bytes=text_request.return_bytes,
#             save_dir=GENERAL_CONFIG['save_dir'],
#         )
#         if flag:
#             return FileResponse(path=file_content,media_type="audio/mp3")
#         else:
#             data = {
#                 "status": "text_to_speech response type error",
#                 "message": file_content
#             }
#             return ORJSONResponse(content=data,status_code=500)
#     except Exception as e:
#         logger.error('text_to_speech error: %s', e)
#         error_message = f"An error occurred: {str(e)}"
#         data = {
#             "status": "text_to_speech internal error",
#             "message": error_message
#         }
#         return ORJSONResponse(content=data,status_code=500)


@app.post("/document_level_summary/",response_class=ORJSONResponse,summary= "document level summary",
                                                            description="integrate the original sectional summary",
                                                            tags=["summarization"])
def fetch_document_summary(document_level_request: DocumentLevelSummaryRequest = Body(...)):
    try:
        integrator = Summary_Integrator(api_key=document_level_request.api_key,
                                        base_url=document_level_request.base_url,
                                        model_config=vars(document_level_request.summarizer_params.gpt_model_params))
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
                "message": {"document_summary": response}
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
                                                            tags=["application"])
def fetch_recommendation(recommendation_request: RecommendationRequest = Body(...)):
    try:
        paper_recommender = Paper_Recommender(
            api_key=recommendation_request.api_key,
            base_url=recommendation_request.base_url,
            model_config=vars(recommendation_request.summarizer_params.gpt_model_params),
            proxy=GENERAL_CONFIG["proxy"]
        )
        flag, response = paper_recommender.recommendation_generation(
            document_level_summary=recommendation_request.document_level_summary,
            article=recommendation_request.raw_text,
            score_prompts=recommendation_request.score_prompts
        )
        if flag:
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
                                                            tags=["application"])

def fetch_blog(
               api_key: str = Form(...,example="api_key",description="api_key for openai"),
               base_url: str = Form(...,example="base_url",description="base_url for openai"),
               section_summary: Union[str, List[str]] = Form(..., example="section_summaries"),
               document_level_summary: str = Form(...,example="document_level_summary",description="document level summary of the article"),
               pdf: Union[str, None] = Form(None,example="https://www.example.com/sample.pdf",description="pdf url for the article, if pdf_content is not None, this will be ignored"),
               file_name: str = Form(None,example="blog",description="file name for the pdf"),
               raw_md_text: str = Form(None, example="raw_text",description="raw markdown text for the article pdf, used for more accurate alignment"),
               blog_prompts: str = Form(json.dumps(APPLICATION_PROMPTS["blog_prompts"]),example="'{'blog': 'xxx', 'blog_system': 'xxx'}'"),
               init_grid: int = Form(ALIGNMENT_CONFIG["init_grid"],example= 2),
               max_grid: int = Form(ALIGNMENT_CONFIG["max_grid"],example= 4),
               img_width: int = Form(ALIGNMENT_CONFIG["img_width"],example= 600,description="display width of the image"),
               threshold: float = Form(ALIGNMENT_CONFIG["threshold"],example= 0.8,description="threshold of similarity for alignment"),
               summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
               pdf_content: UploadFile = File(None,description="Multiple files as UploadFile")):
    if pdf_content:
        content = pdf_content.file.read()
        dir_name = os.path.join(GENERAL_CONFIG['save_dir'],Path(pdf_content.filename).stem)
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
            temp_file=True
        )
        path = Path(response)
        if flag:
            logging.info(f"saving blog to {path}")
            zip_file = zip_dir_to_bytes(dir_path=osp.dirname(path))
            shutil.rmtree(osp.dirname(path))
            return StreamingResponse(io.BytesIO(zip_file), media_type="application/zip", headers={
                "Content-Disposition": f"attachment; filename={Path(path).stem}.zip"})

        else:
            data = {
                "status": "blog generation error",
                "message": response
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
        os.remove(pdf_path)
        logging.info(f"remove {pdf_path}")



class BroadcastScriptRequest(BaseModel):
    llm_api_key: str
    llm_base_url: str
    document_level_summary: str
    section_summaries: Union[str, List[str]]
    broadcast_prompts: dict = APPLICATION_PROMPTS["broadcast_prompts"]
    summarizer_params: Summarizer_Config = Summarizer_Config()

class TTSRequest(BaseModel):
    tts_api_key: str
    tts_base_url: str
    app_secret: str
    text: str




@app.post("/broadcast_generation/",response_class=ORJSONResponse,summary= "broadcast generation",
                                                description="generate broadcast based on the document level summary",
                                                tags=["application"])
def Broadcast_script_generation(broadcast_request:BroadcastScriptRequest = Body(...)):
    broadcast_script_generator = BroadcastTTSGenerator(llm_api_key=broadcast_request.llm_api_key,
                                                          llm_base_url=broadcast_request.llm_base_url,
                                                          model_config=vars(broadcast_request.summarizer_params.gpt_model_params),
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
                "message": {
                    "broadcast_script": response
                }
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
                                                tags=["application"])
def fetch_tts(tts_request:TTSRequest = Body(...)):
    broadcast_tts_generator = BroadcastTTSGenerator(
                                                    llm_api_key=None,
                                                    llm_base_url=None,
                                                    tts_base_url=tts_request.tts_base_url,
                                                    tts_api_key=tts_request.tts_api_key,
                                                    app_secret=tts_request.app_secret,
                                                    )
    try:
        flag, bytes_data = broadcast_tts_generator.text2speech(
            text=tts_request.text,
            return_bytes=True
        )
        if flag:
            return Response(content=bytes_data, media_type="audio/mp3")

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

