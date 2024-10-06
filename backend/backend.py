import os
import os.path as osp
import io
import sys
import tempfile
import time
from datetime import datetime
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from MMAPIS.backend.data_structure import SectionIMGPaths
from MMAPIS.backend.preprocessing import ArxivCrawler,NougatPredictor, PDFFigureExtractor, Aligner
from MMAPIS.backend.summarization import Summarizer,SectionSummarizer,DocumentSummarizer
from MMAPIS.backend.config.config import GENERAL_CONFIG,ALIGNMENT_CONFIG,OPENAI_CONFIG,LOGGER_MODES,APPLICATION_PROMPTS, SECTION_PROMPTS,DOCUMENT_PROMPTS,TTS_CONFIG
from MMAPIS.backend.downstream import PaperRecommender, Regenerator, BroadcastTTSGenerator, MultiModalQAGenerator, BlogGenerator
from MMAPIS.backend.tools import init_logging,download_pdf,zip_dir_to_bytes, bytes2io, zip_dir_to_bytes
from MMAPIS.backend.tools.utils.server_models import *
from MMAPIS.backend.tools.utils.server_function import *
from pathlib import Path
from fastapi.responses import ORJSONResponse,Response,HTMLResponse,RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI,Body,File, UploadFile,Form,Depends, Request, HTTPException
import json
from typing import List,Union
import logging
import shutil
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import asynccontextmanager
import asyncio
import aiofiles
import requests



init_logging()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_MODES)

app = FastAPI(title="MMAPIS",description="A Multi-Modal Automated Academic Papers Interpretation System",version="1.0")
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




@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles validation errors by transforming the error messages into a more readable format.

    :param request: The incoming HTTP request
    :param exc: The exception raised due to validation errors
    :return: A JSON response with a formatted error message
    """
    logger.error(f'Validation exception handler: {exc}')
    error_infos = exc.errors()

    # Generate and return the error response
    if isinstance(error_infos, list):
        error_response = generate_error_message(error_infos)
        data = {
            "status": "request error",
            "message": error_response
        }
        return ORJSONResponse(content=data, status_code=400)

    # Fallback error response
    data = {
        "status": "request error",
        "message": str(exc)
    }
    return ORJSONResponse(content=data, status_code=400)


@app.post("/get_links/daily_search/", response_class=ORJSONResponse,
          summary="Crawl information from arXiv using daily search",
          description="Fetches links, titles, abstracts, and authors from arXiv based on daily updates.",
          tags=["backend", "preprocess"])
def fetch_links(link_request: ArxivRequest = Body(...)):
    arxiv_crawler = ArxivCrawler()
    try:
        articles = arxiv_crawler.run_daily_crawler(
            daily_type=link_request.daily_type,
            max_return=link_request.max_return,
            return_md=link_request.return_md,
            line_length=link_request.line_length,
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
        return handle_error(e, "fetch daily links")

@app.post("/get_links/keyword_search/", response_class=ORJSONResponse,
          summary="Crawl information from arXiv using keyword search",
          description="Fetches links, titles, abstracts, and authors from arXiv based on keyword search.",
          tags=["backend", "preprocess"])
def fetch_links(link_request: ArxivRequest = Body(...)):
    arxiv_crawler = ArxivCrawler()
    try:
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
        return handle_error(e, "fetch links via keyword")

@app.post("/extract_img/", response_class=ORJSONResponse,
          summary="Extract Images from PDF",
          description="Extracts figures or images from the uploaded PDF file and returns them as a zip archive.",
          tags=["backend", "preprocess"])
async def extract_img(
        pdf:str = Form(None,example="https://arxiv.org/pdf/xxxx.xxxx",description="pdf url"),
        pdf_content:Union[UploadFile,None] = File(None,description="Multiple files as UploadFile")):
    """
    Asynchronously extracts images or figures from the uploaded PDF file. The resulting images are compressed into a
    zip file and returned as a response.
    """
    try:
        if pdf:
            # content = await fetch_pdf_content(pdf)
            content = requests.get(pdf).content
            file_name = Path(pdf).stem
        elif pdf_content:
            content = await pdf_content.read()
            file_name = Path(pdf_content.filename).stem
        else:
            raise HTTPException(status_code=400,
                                detail="No PDF source provided. Please provide either a URL or upload a file.")

        save_dir = GENERAL_CONFIG['save_dir']
        os.makedirs(save_dir,exist_ok=True)
        save_dir = tempfile.mkdtemp(dir=save_dir)
        pdf_path = os.path.join(save_dir, f"{file_name}.pdf")

        async with aiofiles.open(pdf_path, 'wb') as f:
            await f.write(content)
        pdf_parser = PDFFigureExtractor(pdf_path=pdf_path)
        img_paths = await asyncio.to_thread(pdf_parser.extract_save_figures, save_dir=save_dir)
        del pdf_parser
        img_dir = os.path.join(save_dir,"img")
        # Compress the extracted images into a zip file
        zip_file = await asyncio.to_thread(zip_dir_to_bytes, img_dir)

        # Clean up the temporary directory
        await asyncio.to_thread(shutil.rmtree, osp.dirname(pdf_path))

        # Return the zip file as a response
        return Response(zip_file, media_type="application/zip",
                        headers={"Content-Disposition": f"attachment; filename={file_name}.zip"}, status_code=200)
    except Exception as e:
        return handle_error(e, "Image Extraction")



@app.post("/pdf2md/", response_class=ORJSONResponse,
          summary='PDF to Markdown Transformation',
          description="Transform the article format from PDF to Markdown",
          tags=["backend", "preprocess"])
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
            pdf_names = [pdf_names] if isinstance(pdf_names,str) else pdf_names
        else:
            if pdf is None:
                raise HTTPException(status_code=400, detail="No PDF file or PDF URL found")
            else:
                pdfs = pdf if isinstance(pdf, List) else [pdf]
                pdf_names = pdf_name if isinstance(pdf_name, List) else [pdf_name] if pdf_name is not None else [None] * len(pdfs)

        try:
            assert len(pdf_names) == len(
                pdfs), f"pdf_names length {len(pdf_names)} not equal to pdfs length {len(pdfs)}"
            loop = asyncio.get_event_loop()
            article_ls = await loop.run_in_executor(None, nougat_preditor.pdf2md_text, pdfs, pdf_names)
            response_data = {
                "status": "success",
                "message": [{"file_name": article.file_name, "text": article.content} for article in article_ls]
            }

            return ORJSONResponse(content=response_data, status_code=200)
        except Exception as e:
            return handle_error(e, "pdf->md")


@app.post("/alignment/", response_class=Response,
          summary='Align Text with Image',
          description="Align the text with the image based on title-like keywords",
          tags=["backend", "preprocess"])
def fetch_alignment(alignment_request:AlignmentRequest):
    """
    Fetches alignment of text with images based on specified parameters.

    Args:
        alignment_request (AlignmentRequest): Request body containing alignment parameters.

    Returns:
        Response: Alignment result or error message.
    """
    aligner = Aligner()

    # Transform each path dictionary into a SectionIMGPaths object
    alignment_request.img_paths = [SectionIMGPaths(**path_dict) for path_dict in alignment_request.img_paths]
    try:
        res = aligner.align(
                text=alignment_request.text,
                img_paths=alignment_request.img_paths,
                min_grained_level=alignment_request.min_grained_level,
                max_grained_level=alignment_request.max_grained_level,
                img_width=alignment_request.img_width,
                margin=alignment_request.margin,
                threshold=alignment_request.threshold,
                raw_md_text=alignment_request.raw_md_text
            )
        data = {
            "status": "success",
            "message": res
        }
        return ORJSONResponse(content=data, status_code=200)

    except Exception as e:
        return handle_error(e, "alignment")



@app.post("/regeneration/", response_class=ORJSONResponse,
          summary='Regenerate the Article',
          description="Regenerate the article based on the alignment result",
          tags=["backend", "application"])
def fetch_regeneration(regeneration_request: RegenerationRequest = Body(...)):
    try:
        regenerator = Regenerator(api_key=regeneration_request.api_key,
                                  base_url=regeneration_request.base_url,
                                  model_config=vars(regeneration_request.summarizer_params.gpt_model_params),
                                  prompt_ratio=regeneration_request.summarizer_params.prompt_ratio,
                                    )
        flag, response = regenerator.regeneration(section_level_summary=regeneration_request.section_level_summary,
                                                 regeneration_prompts=regeneration_request.regenerate_prompts,
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
        return handle_error(e, "regeneration")



@app.post("/section_level_summary/",response_class=ORJSONResponse,summary='summary with openai',
                                                          description="split the article into several groups, summarize each part with openai gpt and integrate the summaries into a whole summary blog",
                                                          tags=["backend","summarization"])
def fetch_section_level_summary(summary_request: SectionSummaryRequest = Body(...)):
    try:
        section_summarizer = SectionSummarizer(api_key=summary_request.api_key,
                                                base_url=summary_request.base_url,
                                                model_config=vars(summary_request.summarizer_params.gpt_model_params),
                                                proxy=GENERAL_CONFIG["proxy"],
                                                prompt_ratio= summary_request.summarizer_params.prompt_ratio,
                                                rpm_limit=summary_request.summarizer_params.rpm_limit,
                                                num_processes= OPENAI_CONFIG['num_processes'],
                                                ignore_titles= summary_request.summarizer_params.ignore_titles,
                                                )
        flag, res = section_summarizer.section_summarize(
            raw_md_text=summary_request.raw_md_text,
            file_name=summary_request.file_name,
            summary_prompts=summary_request.section_prompts,
            min_grained_level=summary_request.min_grained_level,
            max_grained_level=summary_request.max_grained_level,
        )
        if flag:
            data = {
                "status": "success",
                "message": res
            }
            return ORJSONResponse(content=data, status_code=200)

        else:
            data = {
                "status": "request LLM api error",
                "message": res
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        return handle_error(e, "Section-Level Summarization")





@app.post("/summary/", response_class=ORJSONResponse,
          summary='Summarize document using OpenAI',
          description="Summarizes the entire document by splitting it into sections and using OpenAI GPT, producing a comprehensive blog post.",
          tags=["backend", "summarization"])
def fetch_section_level_summary(summary_request: SummaryRequest = Body(...)):
    summarizer = Summarizer(api_key=summary_request.api_key,
                            base_url=summary_request.base_url,
                            model_config=vars(summary_request.summarizer_params.gpt_model_params),
                            proxy=GENERAL_CONFIG["proxy"],
                            prompt_ratio= summary_request.summarizer_params.prompt_ratio,
                            rpm_limit=summary_request.summarizer_params.rpm_limit,
                            num_processes= OPENAI_CONFIG['num_processes'],
                            ignore_titles= summary_request.summarizer_params.ignore_titles,
                            )
    try:
        flag, res = summarizer.generate_summary(
            text=summary_request.raw_md_text,
            file_name=summary_request.file_name,
            section_prompts=summary_request.section_prompts,
            document_prompts=summary_request.document_prompts,
            min_grained_level=summary_request.min_grained_level,
            max_grained_level=summary_request.max_grained_level,
        )
        if flag:
            data = {
                "status": "success",
                "message": res
            }
            return ORJSONResponse(content=data, status_code=200)
        else:
            data = {
                "status": "request LLM api error",
                "message": res
            }
            return ORJSONResponse(content=data, status_code=500)
    except Exception as e:
        return handle_error(e, "Two-Stage Summarization")



@app.post("/recommendation_generation/",response_class=ORJSONResponse,summary= "recommendation generation",
                                                            description="generate recommendation based on the document level summary",
                                                            tags=["backend","application"])
def fetch_recommendation(recommendation_request: RecommendationRequest = Body(...)):
    try:
        paper_recommender = PaperRecommender(
            api_key=recommendation_request.api_key,
            base_url=recommendation_request.base_url,
            model_config=vars(recommendation_request.summarizer_params.gpt_model_params),
            proxy=GENERAL_CONFIG["proxy"],
            prompt_ratio = recommendation_request.summarizer_params.prompt_ratio,
        )
        flag, response = paper_recommender.recommendation_generation(
            document_level_summary=recommendation_request.document_level_summary,
            raw_md_text=recommendation_request.raw_md_text,
            score_prompts=recommendation_request.score_prompts
        )
        if flag:
            ## TODO: replace the avg_score_ls with the real avg_score
            avg_score_ls = [6] * len(response)
            for i, avgscore in enumerate(avg_score_ls):
                response[i]['avg_score'] = avgscore
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
        return handle_error(e, "Recommendation generation")





@app.post("/blog_generation/", response_class=ORJSONResponse,
          summary="Blog Generation",
          description="Generate blog based on the document level summary",
          tags=["backend", "application"])
def fetch_blog(blog_request:BlogRequest):
    blog_generator = BlogGenerator(
        api_key=blog_request.api_key,
        base_url=blog_request.base_url,
        model_config=vars(blog_request.summarizer_params.gpt_model_params),
        proxy=GENERAL_CONFIG["proxy"],
        prompt_ratio=blog_request.summarizer_params.prompt_ratio,
    )
    try:
        flag, content = blog_generator.blog_generation(
            document_level_summary=blog_request.document_level_summary,
            section_level_summary=blog_request.section_level_summary,
            blog_prompts=blog_request.blog_prompts,
        )
        if flag:
            data = {
                    "status": "success",
                    "message": content
                }
            return ORJSONResponse(content=data, status_code=200)

        else:
            data = {
                "status": "blog generation error",
                "message": str(content)
            }
            return ORJSONResponse(content=data, status_code=500)

    except Exception as e:
        return handle_error(e, "Blog generation")




@app.post("/Multimodal_qa/", response_class=ORJSONResponse,
          summary="Multimodal Question Answering",
          description="Perform multimodal question answering",
          tags=["backend", "application"])
def Multimodal_qa(multimodal_qa_request:Multimodal_QA_Request = Body(...)):
    multimodal_qa_generator = MultiModalQAGenerator(
        api_key=multimodal_qa_request.api_key,
        base_url=multimodal_qa_request.base_url,
        model_config=vars(multimodal_qa_request.summarizer_params.gpt_model_params),
        prompt_ratio=multimodal_qa_request.prompt_ratio,
        proxy=GENERAL_CONFIG["proxy"],
    )
    try:
        flag, answer = multimodal_qa_generator.chat(
            user_input=multimodal_qa_request.user_input,
            document_level_summary=multimodal_qa_request.document_level_summary,
            session_message=multimodal_qa_request.session_message,
            raw_md_text=multimodal_qa_request.raw_md_text,
            prompts=multimodal_qa_request.prompts,
            min_grained_level=multimodal_qa_request.min_grained_level,
            max_grained_level=multimodal_qa_request.max_grained_level,
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
    except Exception as e:
        return handle_error(e, "Multimodal QA")



@app.post("/broadcast_generation/",
          response_class=ORJSONResponse,
          summary="Broadcast Script Generation",
          description="Generate broadcast script based on the document level summary",
          tags=["backend", "application"])
def Broadcast_script_generation(broadcast_request:BroadcastScriptRequest = Body(...)):
    broadcast_script_generator = BroadcastTTSGenerator(llm_api_key=broadcast_request.llm_api_key,
                                                        llm_base_url=broadcast_request.llm_base_url,
                                                       model_config=vars(broadcast_request.summarizer_params.gpt_model_params),
                                                       prompt_ratio=broadcast_request.summarizer_params.prompt_ratio,
                                                          )
    try:
        flag, response = broadcast_script_generator.broadcast_script_generation(
            document_level_summary=broadcast_request.document_level_summary,
            section_level_summary=broadcast_request.section_level_summary,
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
        return handle_error(e, "Broadcast Generation")





@app.post("/tts/youdao/", response_class=ORJSONResponse, summary="Text to Speech (Youdao)",
          description="Generate speech audio from text using Youdao TTS", tags=["backend", "application"])
async def fetch_youdao_tts(tts_request: TTSRequest = Body(...)):
    youdao_tts_generator = BroadcastTTSGenerator(
        llm_api_key=None,
        llm_base_url=None,
        tts_base_url=tts_request.base_url,
        tts_api_key=tts_request.api_key,
        app_secret=tts_request.app_secret,
        tts_model=tts_request.model
    )

    try:
        mp3_bytes = await generate_tts(youdao_tts_generator, tts_request.text)
        return Response(mp3_bytes, media_type="audio/mpeg", status_code=200)

    except Exception as e:
        return handle_error(e, "Youdao TTS Generation")


@app.post("/tts/openai/", response_class=ORJSONResponse, summary="Text to Speech (OpenAI)",
          description="Generate speech audio from text using OpenAI TTS", tags=["backend", "application"])
async def fetch_openai_tts(tts_request: TTSRequest = Body(...)):
    tts_request.api_key = OPENAI_CONFIG['api_key']
    tts_request.base_url = OPENAI_CONFIG['base_url']

    openai_tts_generator = BroadcastTTSGenerator(
        llm_api_key=tts_request.api_key,
        llm_base_url=tts_request.base_url,
        tts_model=tts_request.model
    )

    try:
        mp3_bytes = await generate_tts(openai_tts_generator, tts_request.text)
        return Response(mp3_bytes, media_type="audio/mpeg", status_code=200)

    except Exception as e:
        return handle_error(e, "OpenAI TTS Generation")
