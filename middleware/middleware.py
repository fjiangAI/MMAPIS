import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import uuid
from fastapi import FastAPI,Body,File, UploadFile,Form,Depends, Request, HTTPException
from fastapi.responses import HTMLResponse,ORJSONResponse
from fastapi.staticfiles import StaticFiles
import time
from MMAPIS.middleware.config import GENERAL_CONFIG,TEMPLATE_DIR, APPLICATION_PROMPTS,ALIGNMENT_CONFIG, OPENAI_CONFIG,SECTION_PROMPTS,DOCUMENT_PROMPTS
from MMAPIS.middleware.utils import *
import aiofiles
from fastapi.responses import RedirectResponse
import aiohttp
import json
import requests
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from aiofiles import open as aio_open
from fastapi.exceptions import RequestValidationError




app = FastAPI(title="Static Files Server",
              description="This is a static files server for MMAPIS",
              version="1.0")


origins = [
    GENERAL_CONFIG['backend_url'],
    GENERAL_CONFIG['frontend_url'],
]
backend_url = GENERAL_CONFIG['backend_url']


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles validation errors by transforming the error messages into a more readable format.

    :param request: The incoming HTTP request
    :param exc: The exception raised due to validation errors
    :return: A JSON response with a formatted error message
    """
    logging.error(f'Validation exception handler: {exc}')
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


@app.post("/get_links/",response_class=ORJSONResponse,
          summary="crawl information from arxiv",
          description="get basic info(links, titles, abstract, authors) from arxiv",
          tags=["backend","preprocess"])
async def fetch_links(link_request: ArxivRequest = Body(...)):
    request_url = backend_url + "/get_links/"
    try:
        if link_request.key_word is None:
            request_url = request_url + "daily_search/"
            async with aiohttp.ClientSession() as session:
                async with session.post(request_url, json=dict_filter_none(link_request.dict())) as response:
                    json_info = await handle_async_api_response(response)
        else:
            if isinstance(link_request.key_word, list):
                link_request.key_word = " ".join(link_request.key_word)
            request_url = request_url + "keyword_search/"
            async with aiohttp.ClientSession() as session:
                async with session.post(request_url, json=dict_filter_none(link_request.dict())) as response:
                    json_info = await handle_async_api_response(response)
        if json_info["status"] == "success":
            data = json_info
            return ORJSONResponse(content=data,status_code=200)
    except Exception as e:
        return handle_error(e, "fetch links")




os.makedirs(GENERAL_CONFIG['app_save_dir'], exist_ok=True)
app.mount("/index/", StaticFiles(directory=f"{GENERAL_CONFIG['app_save_dir']}"), name="index")



@app.get("/index/{file_path:path}",response_class=HTMLResponse,summary= "get static file",
                                                description="get static file",
                                                tags=["middleware","file"])
async def get_static_file(file_path: str):
    full_path = osp.join(GENERAL_CONFIG['app_save_dir'], file_path)
    if not is_allowed_file_type(full_path):
        return HTMLResponse(
            "<h1>File type not allowed</h1>",
            status_code=403,
        )
    if not osp.exists(full_path):
        return HTMLResponse(
            "<h1>File not found</h1>",
            status_code=404,
        )

    elif full_path.endswith(".html"):
        async with aio_open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = await f.read()
        return HTMLResponse(content=html_content)
    else:
        return StaticFiles(directory=GENERAL_CONFIG['app_save_dir']).get_response(file_path, None)



@app.post("/pdf2md/{user_id}/{file_id}/", response_class=ORJSONResponse,
          summary="Transform PDF to Markdown",
          description="Converts the content of the provided PDF to Markdown format.",
          tags=["middleware", "preprocess"])
async def fetch_predictions(user_id: str,
                            file_id: str,
                            pdf:List[str] = Form(None,example="https://arxiv.org/pdf/xxxx.xxxx",description="pdf url or list of pdf urls, if pdf_content is not None, this param will be ignored"),
                            pdf_name:Union[str,List[str]] = Form(None,example="xxxx.xxxx"),
                            markdown:bool = Form(True,example=True),
                            pdf_content:List[UploadFile] = File(None,description="Multiple files as UploadFile"),
                        ):
    save_dir = os.path.join(GENERAL_CONFIG['app_save_dir'], user_id, file_id)
    markdown_file_path = os.path.join(save_dir, "raw.md")
    if os.path.exists(markdown_file_path):
        with open(markdown_file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        data = {
            "status": "success",
            "message": [
                {
                    "file_name": pdf_name if pdf_name else "raw.md",
                    "text": markdown_content
                }
            ]
        }
        return ORJSONResponse(content=data, status_code=200, headers={"content-type": "application/json"})

    # Organize and clean input parameters by removing None values
    request_param = dict_filter_none({
        "pdf": pdf,
        "pdf_name": pdf_name,
        "markdown": markdown
    })

    # Define backend URL for processing the PDF to Markdown conversion
    request_url = backend_url + "/pdf2md/"

    # If file contents are provided, read the files and prepare them for upload
    if pdf_content:
        pdf_contents = [await pdf.read() for pdf in pdf_content]
        files = [('pdf_content', bytes2io(pdf_content)) for pdf_content in pdf_contents]
    else:
        if middleware_url in pdf[0]:
            path_map = lambda x: url2path(
                x,
                base_url=urljoin(GENERAL_CONFIG['middleware_url'], "/index/"),
                base_dir=GENERAL_CONFIG['app_save_dir']
            )
            pdf_paths = list(map(path_map, pdf))
            files = [
                BytesIO(await read_file(pdf_path, is_bytes=True))
                for pdf_path in pdf_paths
            ]
        else:
            files = None

    try:
        async with aiohttp.ClientSession() as session:
            json_info = await post_form_request(request_url=request_url, request_param=request_param, files=files)
        if json_info["status"] == "success":
            os.makedirs(save_dir, exist_ok=True)
            message_data = []
            for markdown_file in json_info['message']:
                with open(markdown_file_path, "w", encoding="utf-8") as f:
                    f.write(markdown_file['text'])

                # Prepare the response message with the file's URL
                message_data.append({
                    "file_name": markdown_file['file_name'],
                    "text": path2url(markdown_file_path, base_url=GENERAL_CONFIG['middleware_url'], absolute=True)
                })
            response_data = {
                "status": "success",
                "message": message_data
            }
            return ORJSONResponse(content=response_data, status_code=200, headers={"content-type": "application/json"})
        else:
            return ORJSONResponse(content=json_info, status_code=500, headers={"content-type": "application/json"})
    except Exception as e:
        return handle_error(e, "PDF to Markdown Conversion")


@app.post("/summary/{user_id}/{file_id}/{request_id}/", response_class=ORJSONResponse,
          summary="Generate Document Level Summary",
          description="Generates a document-level summary based on the provided text or PDF.",
          tags=["middleware", "summarization"])
async def fetch_app_document_level_summary(
        user_id: str,
        file_id: str,
        request_id: str,
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        raw_md_text:str = Form(...,example="article text",description="raw markdown text of the article"),
        pdf: Union[str, None] = Form(..., example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        from_middleware: bool = Form(False, example=False, description="whether the pdf is from middleware"),
        file_name: str = Form(None, example="blog", description="file name for the pdf"),
        section_prompts: str = Form(json.dumps(SECTION_PROMPTS), example="'{'system': 'xxx', 'abstract': 'xxx'}'"),
        document_prompts: str = Form(json.dumps(DOCUMENT_PROMPTS)),
        min_grained_level: int = Form(ALIGNMENT_CONFIG["min_grained_level"], example=2),
        max_grained_level: int = Form(ALIGNMENT_CONFIG["max_grained_level"], example=4),
        img_width: int = Form(ALIGNMENT_CONFIG["img_width"], example=400, description="display width of the image"),
        margin: int = Form(ALIGNMENT_CONFIG["margin"], example=10, description="margin of the image"),
        threshold: float = Form(ALIGNMENT_CONFIG["threshold"], example=0.8,
                                description="threshold of similarity for alignment"),
        summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
    ):
    save_dir = prepare_save_directory(user_id, file_id, request_id)

    # Paths for saving summary and alignment outputs
    section_level_summary_path, document_level_summary_path, alignment_dir = get_summary_paths(save_dir)

    existing_files = [f for f in os.listdir(alignment_dir) if f.endswith(".md")]
    if existing_files:
        target_file_path = os.path.join(alignment_dir,existing_files[-1])
        aligned_document_text = await read_file(target_file_path)
        section_level_summary = await read_file(section_level_summary_path)
        document_level_summary = await read_file(document_level_summary_path)
        data = {
            "status": "success",
            "message": {
                "document_level_summary": document_level_summary,
                "section_level_summary": section_level_summary,
                "document_level_summary_aligned": normalize_header(aligned_document_text)
            }
        }
        return ORJSONResponse(content=data, status_code=200)

    # Convert input strings to their JSON equivalents
    section_prompts = json.loads(section_prompts)
    document_prompts = json.loads(document_prompts)
    summarizer_params = json.loads(summarizer_params)
    try:
        if raw_md_text.startswith("http"):
            raw_md_path = url2path(raw_md_text, base_url=urljoin(GENERAL_CONFIG['middleware_url'], "/index/"),
                                   base_dir=GENERAL_CONFIG['app_save_dir'])
            raw_md_text = await read_file(raw_md_path)
        # Fetch document-level summary from backend
        if os.path.exists(document_level_summary_path):
            document_level_summary = await read_file(file_path=document_level_summary_path)
            section_level_summary = await read_file(file_path=section_level_summary_path)
        else:
            # Asynchronous session for backend API calls
            async with aiohttp.ClientSession() as session:
                summary_url = backend_url + "/summary/"
                summary_params = {
                    "api_key": api_key,
                    "base_url": base_url,
                    "raw_md_text": raw_md_text,
                    "file_name": file_name,
                    "min_grained_level": min_grained_level,
                    "max_grained_level": max_grained_level,
                    "section_prompts": section_prompts,
                    "document_prompts":document_prompts,
                    "summarizer_params": summarizer_params
                }
                async with session.post(summary_url, json=summary_params) as response:
                    json_info = await handle_async_api_response(response)
                    if json_info["status"] != "success":
                        raise Exception(f"Two-Stage summary generation error: {json_info['message']}")
                    section_level_summary = json_info["message"]["section_level_summary"]
                    document_level_summary = json_info["message"]["document_level_summary"]
                    await save_file(section_level_summary, section_level_summary_path)
                    await save_file(document_level_summary, document_level_summary_path)
        aligned_document_text = await align_text_with_images(
            text=document_level_summary,
            pdf=pdf,
            save_dir=alignment_dir,
            min_grained_level=min_grained_level,
            max_grained_level=max_grained_level,
            raw_md_text=raw_md_text,
            img_width=img_width,
            threshold=threshold,
            margin=margin,
            from_middleware=from_middleware
        )

        await save_file(aligned_document_text,path=os.path.join(alignment_dir,"aligned_document.md"))
        # Return the generated summaries and alignment
        return ORJSONResponse(content={
            "status": "success",
            "message": {
                "document_level_summary": document_level_summary,
                "section_level_summary": section_level_summary,
                "document_level_summary_aligned": normalize_header(aligned_document_text)
            }
        }, status_code=200)

    except Exception as e:
        return handle_error(e, "Two Stage Summary Generation")



@app.post("/app/regeneration/{user_id}/{file_id}/{request_id}/",response_class=ORJSONResponse,summary= "regenerate document-level summary",
                                                            description="regenerate document-level summary",
                                                            tags=["middleware","summarization"])
async def fetch_app_document_level_summary(
        user_id: str ,
        file_id: str,
        request_id:str,
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        section_level_summary: str = Form(..., example="section summary text"),
        raw_md_text:str = Form(None,example="article text",description="raw markdown text of the article"),
        pdf: Union[str, None] = Form(None, example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        regenerate_prompts: str = Form(json.dumps(DOCUMENT_PROMPTS)),
        min_grained_level: int = Form(ALIGNMENT_CONFIG["min_grained_level"], example=2),
        max_grained_level: int = Form(ALIGNMENT_CONFIG["max_grained_level"], example=4),
        img_width: int = Form(ALIGNMENT_CONFIG["img_width"], example=400, description="display width of the image"),
        margin: int = Form(ALIGNMENT_CONFIG["margin"], example=10, description="margin of the image"),
        threshold: float = Form(ALIGNMENT_CONFIG["threshold"], example=0.8,
                                description="threshold of similarity for alignment"),
        summarizer_params: str = Form(json.dumps(summarizer_config.dict())),
        from_middleware: bool = Form(False, example=False, description="whether the pdf is from middleware"),
        pdf_content: UploadFile = File(None, description="Multiple files as UploadFile"),
    ):
    # Create necessary directories
    save_dir = prepare_save_directory(user_id,file_id,request_id,"aligned_regeneration")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir,f"{uuid.uuid4()}.md")

    # Parse JSON inputs
    regenerate_prompts = json.loads(regenerate_prompts)
    summarizer_params = json.loads(summarizer_params)

    # Prepare parameters for regeneration API call
    regenerate_url = GENERAL_CONFIG["backend_url"] + "/regeneration/"
    regenerate_summary_params = {
        "api_key": api_key,
        "base_url": base_url,
        "section_level_summary": section_level_summary,
        "regenerate_prompts": regenerate_prompts,
        "summarizer_params": summarizer_params
    }

    # Make asynchronous API call for regeneration
    async with aiohttp.ClientSession() as session:
        async with session.post(regenerate_url, json=regenerate_summary_params) as response:
            json_info = await handle_async_api_response(response)
    if json_info["status"] != "success":
        data = {
            "status": "Regeneration error",
            "message": json_info["message"]
        }
        return ORJSONResponse(content=data, status_code=500)

    regeneration_content = json_info["message"]
    aligned_regeneration_content = await align_text_with_images(
                                    text=regeneration_content,
                                    pdf=pdf,
                                    save_dir=save_dir,
                                    raw_md_text=raw_md_text,
                                    min_grained_level=min_grained_level,
                                    max_grained_level=max_grained_level,
                                    margin=margin,
                                    threshold=threshold,
                                    img_width=img_width,
                                    from_middleware=from_middleware
                                )
    await save_file(aligned_regeneration_content, path=file_path)

    data = {
        "status": "success",
        "message": aligned_regeneration_content
    }
    return ORJSONResponse(content=data, status_code=200)


@app.post("/upload_zip_file/{user_id}/{file_id}/",response_class=ORJSONResponse,
                            summary="upload zip file and save to middleware",
                            description="upload zip file and save to middleware",
                            tags=["middleware","file"])
async def create_upload_file(
                       user_id: str,
                       file_id: str,
                       file_type: str = Form(...),
                       temp_file: bool = Form(True, description="whether to save the file to temp file"),
                       zip_content: UploadFile = File(None, description="pdf bytes file as UploadFile"),
    ):
    save_dir = prepare_save_directory(user_id, file_id, file_type)
    os.makedirs(save_dir, exist_ok=True)
    if temp_file:
        save_dir = tempfile.mkdtemp(dir=save_dir)
    if "mp3" in file_type:
        try:
            bytes_data = await zip_content.read()
            millis = int(round(time.time() * 1000))
            mp3_path = os.path.join(save_dir, str(millis) + ".mp3")
            async with aiofiles.open(mp3_path, 'wb') as f:
                await f.write(bytes_data)
            mp3_url = path2url(
                path=mp3_path,
                base_url=GENERAL_CONFIG['middleware_url'],
                absolute=True
            )
        except Exception as e:
            handle_error(e, "Generate mp3 content error")

        data = {
            "status": "success",
            "message": mp3_url
        }
        return ORJSONResponse(content=data, status_code=200)
    elif "pdf" in file_type:
        try:
            bytes_data = await zip_content.read()
            file_name = zip_content.filename if zip_content.filename.endswith(".pdf") else zip_content.filename+".pdf"
            pdf_path = os.path.join(save_dir, file_name)
            async with aiofiles.open(pdf_path, 'wb') as fo:
                await fo.write(bytes_data)
            pdf_url = path2url(
                path=pdf_path,
                base_url=GENERAL_CONFIG['middleware_url'],
                absolute=True
            )
        except Exception as e:
            handle_error(e, "Upload pdf file error")

        data = {
            "status": "success",
            "message": pdf_url
        }
        return ORJSONResponse(content=data, status_code=200)
    else:
        return ORJSONResponse(content={"status": "error", "message": "file type not allowed"}, status_code=403)





@app.post("/app/{user_id}/{file_id}/{request_id}",
          response_class=RedirectResponse,
          summary= "application generation",
          description="generate application based on the document level summary",
          tags=["middleware","application"])
async def fetch_app(
        user_id: str,
        file_id: str,
        request_id:str,
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        raw_md_text: str = Form(None, example="article text", description="raw markdown text of the article"),
        section_level_summary: Union[str, List[str]] = Form(None, example="section_level_summary"),
        document_level_summary: str = Form(...,example="document_level_summary",description="document level summary of the article"),
        usage:str = Form(...,example="blog",description="usage of the application, choices: ['blog', 'speech', 'regenerate','recommend','qa']"),
        pdf: Union[str, None] = Form(..., example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        prompts: str = Form(None, example="prompts", description="prompts for the application"),
        file_name: str = Form(None, example="blog", description="file name for the pdf"),
        min_grained_level: int = Form(ALIGNMENT_CONFIG["min_grained_level"], example=2),
        max_grained_level: int = Form(ALIGNMENT_CONFIG["max_grained_level"], example=4),
        img_width: int = Form(ALIGNMENT_CONFIG["img_width"], example=600, description="display width of the image"),
        threshold: float = Form(ALIGNMENT_CONFIG["threshold"], example=0.8,
                                description="threshold of similarity for alignment"),
        tts_api_key: str = Form(None, example="api_key", description="api_key for openai"),
        tts_base_url: str = Form(None, example="base_url", description="base_url for openai"),
        app_secret: str = Form("", example="app_secret", description="app_secret"),
        summarizer_params: str = Form(json.dumps(summarizer_config.dict()))):
    if not usage in ['blog', 'speech', 'regenerate','recommend','qa']:
        data = {
            "status": "usage error",
            "message": f"usage {usage} not supported"
        }
        return ORJSONResponse(content=data, status_code=400)
    if not summarizer_params:
        summarizer_params = json.dumps(summarizer_config.dict())
    prompts_map = {
        "blog": APPLICATION_PROMPTS["blog_prompts"],
        "speech": APPLICATION_PROMPTS["broadcast_prompts"],
        "recommend": APPLICATION_PROMPTS["score_prompts"],
        "qa": APPLICATION_PROMPTS["multimodal_qa"]
    }
    if not prompts:
        prompts = prompts_map[usage]
    file_path = os.path.join(prepare_save_directory(user_id, file_id, request_id), f"{usage}_{uuid.uuid4()}.html")

    ## Remove carriage return from the document level summary
    document_level_summary = normalize_header(document_level_summary)

    html_content = await generate_application_html(
                        usage=usage,
                        user_id=user_id,
                        file_id=file_id,
                        request_id=request_id,
                        api_key=api_key,
                        base_url=base_url,
                        document_level_summary=document_level_summary,
                        section_level_summary=section_level_summary,
                        raw_md_text=raw_md_text,
                        pdf=pdf,
                        file_name=file_name,
                        min_grained_level=min_grained_level,
                        max_grained_level=max_grained_level,
                        img_width=img_width,
                        threshold=threshold,
                        prompts=prompts,
                        summarizer_params=summarizer_params,
                    )
    await save_file(content=html_content,path=file_path)
    redirect_url = path2url(path=file_path,base_url=GENERAL_CONFIG['middleware_url'],absolute=True)
    return RedirectResponse(url=redirect_url, status_code=303)







@app.post("/recommendation_generation/",response_class=ORJSONResponse,summary= "recommendation generation",
                                                            description="generate recommendation based on the document level summary",
                                                            tags=["backend","application"])
async def fetch_recommendation(recommendation_request: RecommendationRequest = Body(...)):
    try:
        json_params = recommendation_request.dict()
        recommendation_url= backend_url + "/recommendation_generation/"
        async with aiohttp.ClientSession() as session:
            async with session.post(recommendation_url, json=json_params) as response:
                json_info = await handle_async_api_response(response)
        if json_info["status"] == "success":
            return ORJSONResponse(content=json_info, status_code=200)
        else:
            return ORJSONResponse(content=json_info, status_code=500)

    except Exception as e:
        return handle_error(e, "Recommendation Generation")





@app.post("/blog_generation/{user_id}/{file_id}/{request_id}/", response_class=ORJSONResponse,
          summary="Blog Generation",
          description="Generate blog based on the document level summary",
          tags=["backend", "application"])
async def fetch_blog(
        user_id: str,
        file_id: str,
        request_id: str,
        blog_request:BlogRequest):
    save_dir = prepare_save_directory(user_id,file_id,request_id)
    blog_url = backend_url + "/blog_generation/"
    try:
        json_params = blog_request.dict()
        async with aiohttp.ClientSession() as session:
            async with session.post(blog_url, json=json_params) as response:
                json_info = await handle_async_api_response(response)
        if json_info["status"] != "success":
            return ORJSONResponse(content=json_info, status_code=500)
        blog_content = json_info["message"]
        aligned_blog_content = await align_text_with_images(
            text=blog_content,
            pdf=json_params["pdf"],
            save_dir=save_dir,
            raw_md_text=json_params["raw_md_text"],
            min_grained_level=json_params["min_grained_level"],
            max_grained_level=json_params["max_grained_level"],
            margin=json_params["margin"],
            threshold=json_params["threshold"],
            img_width=json_params["img_width"],
            from_middleware=json_params["from_middleware"]
        )
        json_info["message"] = aligned_blog_content
        return ORJSONResponse(content=json_info, status_code=200)
    except Exception as e:
        return handle_error(e, "Blog generation")




@app.post("/Multimodal_qa/", response_class=ORJSONResponse,
          summary="Multimodal Question Answering",
          description="Perform multimodal question answering",
          tags=["backend", "application"])
async def Multimodal_qa(multimodal_qa_request:Multimodal_QA_Request = Body(...)):
    multimodal_qa_url = backend_url + "/Multimodal_qa/"
    try:
        json_params = multimodal_qa_request.dict()
        async with aiohttp.ClientSession() as session:
            async with session.post(multimodal_qa_url, json=json_params) as response:
                json_info = await handle_async_api_response(response)
        if json_info["status"] == "success":
            return ORJSONResponse(content=json_info, status_code=200)
        else:
            return ORJSONResponse(content=json_info, status_code=500)

    except Exception as e:
        return handle_error(e, "Multimodal QA")


@app.post("/broadcast_generation/",
          response_class=ORJSONResponse,
          summary="Broadcast Script Generation",
          description="Generate broadcast script based on the document level summary",
          tags=["backend", "application"])
async def Broadcast_script_generation(broadcast_request:BroadcastScriptRequest = Body(...)):
    broadcast_url = backend_url + "/broadcast_generation/"
    try:
        json_params = broadcast_request.dict()
        async with aiohttp.ClientSession() as session:
            async with session.post(broadcast_url, json=json_params) as response:
                json_info = await handle_async_api_response(response)
        if json_info["status"] == "success":
            return ORJSONResponse(content=json_info, status_code=200)
        else:
            return ORJSONResponse(content=json_info, status_code=500)

    except Exception as e:
        return handle_error(e, "Broadcast Generation")





@app.post("/tts/{user_id}/{file_id}/{request_id}/", response_class=ORJSONResponse, summary="Text to Speech (Youdao)",
          description="Generate speech audio from text using Youdao TTS", tags=["backend", "application"])
async def fetch_youdao_tts(
        user_id:str,
        file_id:str,
        request_id:str,
        tts_request: TTSRequest = Body(...)):
    if tts_request.model == "youdao":
        tts_url = backend_url + "/tts/youdao/"
    else:
        tts_url = backend_url + "/tts/openai/"
    try:
        json_params = tts_request.dict()
        async with aiohttp.ClientSession() as session:
            async with session.post(tts_url, json=json_params) as response:
                json_info = await handle_async_api_response(response,return_str=False)
        if json_info["status"] != "success":
            return ORJSONResponse(content=json_info, status_code=500)
        save_dir = prepare_save_directory(user_id,file_id,request_id)
        mp3_path = os.path.join(save_dir, f"{uuid.uuid4()}.mp3")
        await save_file(content=json_info["message"], path=mp3_path,is_bytes=True)
        mp3_url = path2url(path=mp3_path, base_url=GENERAL_CONFIG['middleware_url'], absolute=True)
        data = {
            "status": "success",
            "message": mp3_url
        }
        return ORJSONResponse(content=data, status_code=200)


    except Exception as e:
        return handle_error(e, "Youdao TTS Generation")
