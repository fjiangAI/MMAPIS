from fastapi import FastAPI,Body,File, UploadFile,Form
from fastapi.responses import HTMLResponse,ORJSONResponse
from fastapi.staticfiles import StaticFiles
import os.path as osp
import tempfile
import re
import os
import time
import zipfile
from io import BytesIO
from urllib.parse import urljoin
from typing import List, Union
from MMAPIS.config.config import GENERAL_CONFIG,TEMPLATE_DIR, APPLICATION_PROMPTS,ALIGNMENT_CONFIG, OPENAI_CONFIG
import aiofiles
import shutil
from fastapi.responses import Response, RedirectResponse
import json
from pydantic import BaseModel
import asyncio
import httpx
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Static Files Server",
              description="This is a static files server for MMAPIS",
              version="0.2")



origins = [
    GENERAL_CONFIG['backend_url'],
    GENERAL_CONFIG['frontend_url'],
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def string_to_upload_file(file_content: str, filename: str) -> UploadFile:
    bytes_stream = BytesIO(file_content.encode())
    return UploadFile(filename=filename, file=bytes_stream)


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

def url2path(url:str,
             base_url:str,
             base_dir:str):
    if url.startswith("http"):
        url = url.replace(base_url,"")
    path = os.path.join(base_dir,url)
    return path


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


def is_allowed_file_type(file_path: str):
    allowed_file_types = GENERAL_CONFIG['allowed_file_types']
    file_type = osp.splitext(file_path)[1]
    return file_type in allowed_file_types









app.mount("/index/", StaticFiles(directory=f"{GENERAL_CONFIG['app_save_dir']}"), name="index")



@app.get("/index/{file_path:path}",response_class=HTMLResponse,summary= "get static file",
                                                description="get static file",
                                                tags=["Middleware","file"])
async def get_static_file(file_path: str):
    full_path = osp.join(GENERAL_CONFIG['app_save_dir'], file_path)
    if not is_allowed_file_type(full_path):
        return HTMLResponse(
            "<h1>File type not allowed</h1>",
            status_code=403,
        )
    elif full_path.endswith(".html"):
        with open(full_path, "r",encoding="utf-8",errors="ignore") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return StaticFiles(directory=GENERAL_CONFIG['app_save_dir']).get_response(file_path)


@app.post("/upload_zip_file/",response_class=ORJSONResponse,
                            summary="upload zip file and save to middleware",
                            description="upload zip file and save to middleware",
                            tags=["Middleware","file"])
async def create_upload_file(
                       file_type: str = Form(...),
                       user_id: str = Form("", description="user id"),
                       file_id: str = Form("", description="file id"),
                       temp_file: bool = Form(True, description="whether to save the file to temp file"),
                       zip_content: UploadFile = File(None, description="pdf bytes file as UploadFile"),

    ):
    if user_id and file_id:
        save_dir = os.path.join(GENERAL_CONFIG['app_save_dir'], user_id, file_id)
    else:
        save_dir = GENERAL_CONFIG['app_save_dir']
    os.makedirs(save_dir, exist_ok=True)
    if temp_file:
        save_dir = os.path.join(save_dir, file_type)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = tempfile.mkdtemp(dir=save_dir)
    if "md" in file_type:
        # markdown file of application
        if temp_file:
            try:
                extract_zip_from_bytes(
                    zip_data= await zip_content.read(),
                    extract_dir=save_dir
                )
                md_file = [f for f in os.listdir(save_dir) if f.endswith(".md")][0]
                md_file_path = os.path.join(save_dir, md_file)
                async with aiofiles.open(md_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    md_content = await f.read()

                markdown_content = re.sub(r'(?<![\n#])(#+\s+.*?\n+)', r'\n\1', md_content)
                markdown_content = img2url(
                    text=markdown_content,
                    base_url=GENERAL_CONFIG['middleware_url'],
                    img_dir=osp.relpath(osp.dirname(md_file_path),GENERAL_CONFIG['app_save_dir'])
                )
            except Exception as e:
                markdown_content = f"Generate {file_type} content error: {e}"
                return ORJSONResponse(content={"status": "error", "message": markdown_content}, status_code=500)
            data = {
                "status": "success",
                "message": markdown_content
            }
            return ORJSONResponse(content=data, status_code=200)
        # markdown file of summarization
        else:
            zip_content = await zip_content.read()
            text_content = zip_content.decode("utf-8")
            markdown_content = re.sub(r'(?<![\n#])(#+\s+.*?\n+)', r'\n\1', text_content)
            file_name = f"{zip_content.filename}_{file_type}.md"
            async with aiofiles.open(os.path.join(save_dir, file_name), "w", encoding="utf-8", errors="ignore") as f:
                await f.write(markdown_content)
            markdown_url = path2url(
                path=os.path.join(save_dir, file_name),
                base_url=GENERAL_CONFIG['middleware_url'],
                absolute=True
            )

            data = {
                "status": "success",
                "message": markdown_url
            }
            return ORJSONResponse(content=data, status_code=200)



    elif "mp3" in file_type:
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
            mp3_url = f"Generate {file_type} content error: {e}"
            return ORJSONResponse(content={"status": "error", "message": mp3_url}, status_code=500)

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
            pdf_url = f"Generate {file_type} content error: {e}"
            return ORJSONResponse(content={"status": "error", "message": pdf_url}, status_code=500)

        data = {
            "status": "success",
            "message": pdf_url
        }
        return ORJSONResponse(content=data, status_code=200)
    else:
        return ORJSONResponse(content={"status": "error", "message": "file type not allowed"}, status_code=403)



@app.post("/app/{user_id}/{file_id}/",
          response_class=RedirectResponse,
          summary= "application generation",
          description="generate application based on the document level summary",
          tags=["Middleware","application"])
async def fetch_app(
        user_id: str,
        file_id: str,
        api_key: str = Form(..., example="api_key", description="api_key for openai"),
        base_url: str = Form(..., example="base_url", description="base_url for openai"),
        document_level_summary: str = Form(...,example="document_level_summary",description="document level summary of the article"),
        usage:str = Form(...,example="blog",description="usage of the application, choices: ['blog', 'speech', 'regenerate','recommend','qa']"),
        pdf: Union[str, None] = Form(..., example="https://www.example.com/sample.pdf",
                                     description="pdf url for the article, if pdf_content is not None, this will be ignored"),
        raw_md_text:str = Form(None,example="article text",description="raw markdown text of the article"),
        section_summary: Union[str, List[str]] = Form(None, example="section_summaries"),
        prompts: str = Form(None, example="prompts", description="prompts for the application"),
        file_name: str = Form(None, example="blog", description="file name for the pdf"),
        init_grid: int = Form(ALIGNMENT_CONFIG["init_grid"], example=2),
        max_grid: int = Form(ALIGNMENT_CONFIG["max_grid"], example=4),
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
    req_backend_url = f"{GENERAL_CONFIG['backend_url']}"
    if not summarizer_params:
        summarizer_params = json.dumps(summarizer_config.dict())
    temp_path = os.path.join(TEMPLATE_DIR, f"{usage}.html")
    if usage == "blog":
        if file_name is None:
            file_name = file_id
        if not prompts:
            prompts = json.dumps(APPLICATION_PROMPTS["blog_prompts"])
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        html_content = html_content.replace("{{backend_url}}" , req_backend_url)
        html_content = html_content.replace("{{api_key}}", api_key).replace("{{base_url}}", base_url)
        html_content = html_content.replace("{{api_key}}", api_key).replace("{{base_url}}", base_url)
        html_content = html_content.replace("{{document_level_summary}}", document_level_summary).replace("{{section_summary}}", section_summary)
        html_content = html_content.replace("{{pdf}}", pdf).replace("{{file_name}}", file_name).replace("{{raw_md_text}}", raw_md_text)
        html_content = html_content.replace("{{init_grid}}", str(init_grid)).replace("{{max_grid}}", str(max_grid)).replace("{{img_width}}", str(img_width)).replace("{{threshold}}", str(threshold))
        html_content = html_content.replace("{{prompts}}", prompts).replace("{{summarizer_params}}", summarizer_params)
        redirect_url =await save_file(
            user_id=user_id,
            file_id=file_id,
            file_type="blog_html",
            html_content=html_content
        )

    elif usage == "speech":
        if not prompts:
            prompts = json.dumps(APPLICATION_PROMPTS["broadcast_prompts"])

        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        html_content = html_content.replace("{{backend_url}}" , req_backend_url)
        html_content = html_content.replace("{{api_key}}", api_key).replace("{{base_url}}", base_url)
        html_content = html_content.replace("{{document_level_summary}}", document_level_summary).replace("{{section_summary}}", section_summary)
        html_content = html_content.replace("{{prompts}}", prompts).replace("{{summarizer_params}}", summarizer_params)

        redirect_url = await save_file(
            user_id=user_id,
            file_id=file_id,
            file_type="speech_html",
            html_content=html_content
        )

    elif usage == "recommend":
        if not prompts:
            prompts = json.dumps(APPLICATION_PROMPTS["score_prompts"])
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        html_content = html_content.replace("{{backend_url}}" , req_backend_url)
        html_content = html_content.replace("{{api_key}}", api_key).replace("{{base_url}}", base_url)
        html_content = html_content.replace("{{document_level_summary}}", document_level_summary).replace("{{raw_md_text}}", raw_md_text)
        html_content = html_content.replace("{{prompts}}", prompts).replace("{{summarizer_params}}", summarizer_params)
        print("send_request")
        redirect_url = await save_file(
            user_id=user_id,
            file_id=file_id,
            file_type="recommend_html",
            html_content=html_content
        )


    else:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        html_content = html_content.replace("{{backend_url}}" , req_backend_url)
        html_content = html_content.replace("{{api_key}}", api_key).replace("{{base_url}}", base_url)
        html_content = html_content.replace("{{document_level_summary}}", document_level_summary).replace("{{raw_md_text}}", raw_md_text)
        redirect_url = await save_file(
            user_id=user_id,
            file_id=file_id,
            file_type="qa_html",
            html_content=html_content
        )


    return RedirectResponse(url=redirect_url, status_code=303)




@app.post("/upload_file/",response_class=ORJSONResponse,
                            summary="upload zip file and save to middleware",
                            description="upload zip file and save to middleware",
                            tags=["Middleware","file"])
def create_upload_file(file_type: str = Form(...),
                        zip_content: UploadFile = File(None, description="pdf bytes file as UploadFile")
     ):
     print("len(zip_content.file.read()):",len(zip_content.file.read()))
     return {
            "file_type": file_type,
            "file_content": zip_content,
            "file_length": len(zip_content.file.read())
     }


async def save_file(user_id: str,
            file_id: str,
            file_type: str,
            html_content: str):
    save_dir = os.path.join(GENERAL_CONFIG['app_save_dir'], user_id, file_id)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, file_type)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = tempfile.mkdtemp(dir=save_dir)
    try:
        html_file_path = os.path.join(save_dir, file_type + ".html")
        async with aiofiles.open(html_file_path, "w", encoding="utf-8", errors="ignore") as f:
            await f.write(html_content)
    except Exception as e:
        html_content = f"Generate {file_type} content error: {e}"
        return html_content
    print("html_file_path:",html_file_path)
    html_url = path2url(
        path=html_file_path,
        base_url=GENERAL_CONFIG['middleware_url'],
        absolute=True
    )
    return html_url

