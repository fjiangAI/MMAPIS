import logging
import os
import aiofiles
import re
from urllib.parse import urljoin
import io
import zipfile
from io import BytesIO
import os.path as osp
from typing import List, Union
from fastapi.responses import ORJSONResponse, Response
from requests.models import Response as ReqResponse
from aiohttp import ClientResponse
import json
import requests
import aiohttp
from MMAPIS.middleware.config import GENERAL_CONFIG, TEMPLATE_DIR

backend_url = GENERAL_CONFIG['backend_url']
middleware_url = GENERAL_CONFIG['middleware_url']


# Improved error message generation function
def generate_error_message(error_infos):
    """
    Generate a human-readable error message based on the list of validation errors.

    :param error_infos: List of validation error details
    :return: Formatted string of error messages
    """
    error_response = "Input parameter errors:\n"
    for i, error_info in enumerate(error_infos):
        error_type = error_info.get('type', 'Unknown type')
        location = error_info['loc'][0]
        param = error_info['loc'][1] if len(error_info['loc']) > 1 else 'Unknown parameter'
        input_value = error_info.get('input', 'None')
        message = error_info.get('msg', 'No message provided')
        error_response += f"Error {i + 1}: type: {error_type}, location: request {location}, param {param}, input: {input_value}, msg: {message}\n"
    return error_response

def handle_error(exception, process_name):
    """
    A unified error handling function that logs and formats error responses.
    This function ensures consistent error messages and logging across different API endpoints.
    """
    logging.error(f'{process_name} error: {exception}')
    error_message = f"An error occurred during {process_name}: {str(exception)}"
    data = {
        "status": f"{process_name} internal error",
        "message": error_message
    }
    return ORJSONResponse(content=data, status_code=500)

def dir2img_paths(img_dir):
    img_paths= []
    for f in os.listdir(img_dir):
        section_name = f.rsplit("_")[0]
        img_paths.append({"img_path":os.path.abspath(os.path.join(img_dir,f)),"section_name":section_name})
    return img_paths

def bytes2io(bytes_data):
    return io.BufferedReader(io.BytesIO(bytes_data))


def prepare_save_directory(user_id, file_id,*args):
    base_dir = GENERAL_CONFIG['app_save_dir']

    # Join the base directory with the provided directory levels
    save_dir = os.path.abspath(os.path.join(base_dir,user_id, file_id, *args))

    # Ensure the directory exists, create it if necessary
    os.makedirs(save_dir, exist_ok=True)

    return save_dir

def get_summary_paths(save_dir):
    """Returns paths for section and document summaries and alignment directory."""
    section_level_summary_path = os.path.join(save_dir, "section_level_summary.md")
    document_level_summary_path = os.path.join(save_dir, "document_level_summary.md")
    alignment_dir = os.path.join(save_dir, "aligned_document")
    os.makedirs(alignment_dir, exist_ok=True)
    return section_level_summary_path, document_level_summary_path, alignment_dir




async def read_file(file_path: str,is_bytes:bool = False):
    """
    Asynchronously reads the content of a file and returns it as a string.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file.
    """
    try:
        if is_bytes:
            async with aiofiles.open(file_path, mode="rb") as f:
                content = await f.read()
        else:
            async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
        return content
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")


def normalize_header(text: str):
    """
    Normalize the given header by replacing correct newlines.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """
    # Remove extra spaces and newlines
    text = text.replace('\r\n', '\n')
    pattern = re.compile(r'(?<!#)(#{2,})(?!#)', re.MULTILINE)
    text = pattern.sub(r'\n\1', text)
    return text

async def save_file(content, path: str, is_bytes: bool = False):
    """
    Asynchronously saves the given content to a file at the specified path.

    Args:
        content: The content to be written to the file. Can be text or binary data.
        path (str): The path to save the file.
        is_bytes (bool): If True, saves content as bytes. Otherwise, saves as text.

    Raises:
        Exception: If any error occurs during the file writing process.
    """
    try:
        # Ensure the directory exists before saving the file
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if is_bytes:
            async with aiofiles.open(path, 'wb') as file:
                await file.write(content)
        else:
            async with aiofiles.open(path, 'w', encoding='utf-8', errors='ignore') as file:
                await file.write(content)
    except Exception as e:
        raise Exception(f"Error saving file to {path}: {e}")

def img2url(text:str,
             base_url:str,
             img_dir:Union[str,None] = None,
             absolute:bool = False):
    """
    Convert image sources in the given text to URLs based on a base URL and an image directory.
    Hello there
    Args:
        text (str): The text containing image sources to be converted.
        base_url (str): The base URL for generating the image URLs.
        img_dir (str): The directory where the images are located.
        absolute (bool, optional): Whether the img dir path is absolute.

    Returns:
        str: The text with updated image URLs.
    """
    source_pattern = re.compile(r'src="(.+?)"', re.DOTALL)
    source_ls = source_pattern.findall(text)
    for source in source_ls:
        if source.startswith("http"):
            continue
        else:
            img_url = path2url(
                path=f"{img_dir}/{source}" if img_dir else source,
                base_url=base_url,
                absolute=absolute
            )
            text = text.replace(source,img_url)
    return text

def path2url(path:str,
             base_url:str,
             absolute:bool = False):
    """
    Convert a file path to a URL based on a base URL and an option for absolute or relative URL.

    Args:
        path (str): The file path to be converted.
        base_url (str): The base URL for generating the URL.
        absolute (bool, optional): Whether to generate an absolute URL. Defaults to False.

    Returns:
        str: The generated URL.
    """
    if path.startswith("."):
        path = path[1:]
    base_url = base_url + "/index/"
    # Input path is relative path
    if not absolute:
        path = path.replace('\\','/')
        img_url = urljoin(base_url,path)
    else:
        path = os.path.relpath(path,os.path.abspath(GENERAL_CONFIG['app_save_dir']))

        path = path.replace('\\','/')
        img_url = urljoin(base_url,path)
    return img_url

def url2path(url:str,
             base_url:str,
             base_dir:str):
    """
    Convert a URL to a local file path based on a base URL and a base directory.

    Args:
        url (str): The URL to be converted.
        base_url (str): The base URL to be removed from the given URL.
        base_dir (str): The base directory to which the remaining part of the URL is appended.

    Returns:
        str: The resulting local file path.
    """
    if url.startswith("http"):
        url = url.replace(base_url,"")
    path = os.path.join(base_dir,url)
    return path


def dict_filter_none(d:dict):
    return {k:v for k,v in d.items() if v and v != 'null'}



async def extract_zip_from_bytes(zip_data,
                           extract_dir):
    """
    Extracts contents from byte data of a ZIP file into a specified directory.

    Parameters:
    - zip_data: The byte data of the ZIP file.
    - extract_dir: The path to the target directory for extraction.
    """
    # Convert the byte data into a BytesIO object for reading by the zipfile module
    zip_bytes_io = BytesIO(zip_data)
    os.makedirs(extract_dir,exist_ok=True)
    # # Use zipfile to read the BytesIO object
    # with zipfile.ZipFile(zip_bytes_io, 'r') as zip_ref:
    #     # Extract to the specified directory
    #     zip_ref.extractall(extract_dir)

    with zipfile.ZipFile(zip_bytes_io, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            try:
                zip_ref.extract(file_info, extract_dir)
            except Exception as e:
                logging.error(f"Error extracting file {file_info.filename}: {e}")

def is_allowed_file_type(file_path: str):
    allowed_file_types = GENERAL_CONFIG['allowed_file_types']
    file_type = osp.splitext(file_path)[1]
    return file_type in allowed_file_types

def handle_api_response(
        response: Union[Response, ORJSONResponse, ReqResponse],
        return_str: bool = True
):
    if response.status_code != 200:
        json_info = {
            "status": "error",
            "message": response.text
        }
    # the ORJSONResponse object returns body as bytes
    else:
        try:
            if return_str:
                msg = eval(response.text)["message"]
            else:
                msg = response.content
        except Exception as e:
            json_info = {
                "status": "Unsupport response error",
                "message": f"Unsupport response type, error: {e}"
            }
            return json_info
        json_info = {
            "status": "success",
            "message": msg
        }
    return json_info

async def handle_async_api_response(
        response: Union[ClientResponse, ReqResponse],
        return_str: bool = True
):
    if isinstance(response, ClientResponse):
        status_code = response.status
        if return_str:
            text = await response.text()
        else:
            content = await response.read()
    else:
        status_code = response.status_code
        if return_str:
            text = response.text
        else:
            content = response.content
    if status_code != 200:
        json_info = {
            "status": "error",
            "message": text if return_str else content
        }
    else:
        try:
            if return_str:
                msg = json.loads(text)["message"]
            else:
                msg = content
        except Exception as e:
            json_info = {
                "status": "Unsupported response error",
                "message": f"Unsupported response type, error: {e}"
            }
            return json_info

        json_info = {
            "status": "success",
            "message": msg
        }
    return json_info



def replace_content(html_content, replacements: dict) -> str:
    """
    Replaces placeholders in the HTML content with actual values.

    Args:
        html_content (str): Original HTML content containing placeholders.
        replacements (dict): A dictionary where the key is the placeholder and the value is the actual replacement.

    Returns:
        str: Modified HTML content with all placeholders replaced.
    """
    for placeholder, value in replacements.items():
        html_content = html_content.replace(f"{{{{{placeholder}}}}}", value)
    return html_content



async def generate_application_html(usage: str,
                                    user_id: str,
                                    file_id: str,
                                    request_id: str,
                                    api_key: str,
                                    base_url: str,
                                    document_level_summary: str,
                                    section_level_summary: str,
                                    raw_md_text: str,
                                    pdf: str,
                                    file_name: str,
                                    min_grained_level: int,
                                    max_grained_level: int,
                                    img_width: int,
                                    threshold: float,
                                    prompts: str,
                                    summarizer_params: str):
    """
    Generates and saves the application HTML content based on usage.

    Args:
        usage (str): Type of application (e.g., 'blog', 'speech').
        api_key (str): API key for external service.
        base_url (str): Base URL for external service.
        user_id (str): User ID.
        file_id (str): File ID.
        request_id (str): Request ID.
        document_level_summary (str): Summary of the document.
        section_level_summary (str): Section-level summaries of the document.
        raw_md_text (str): Raw markdown text of the document.
        pdf (str): PDF URL for the document.
        file_name (str): Name of the PDF file.
        min_grained_level (int): Initial grid value.
        max_grained_level (int): Maximum grid value.
        img_width (int): Image width for display.
        threshold (float): Similarity threshold for alignment.
        prompts (str): Prompts for the application.
        summarizer_params (str): Parameters for the summarizer.

    Returns:
        None: The generated content is saved to the specified file path.
    """
    temp_path = os.path.join(TEMPLATE_DIR, f"{usage}.html")
    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()

    replacements = {
        "backend_url": middleware_url,
        "user_id": user_id,
        "file_id": file_id,
        "request_id": request_id,
        "api_key": api_key,
        "base_url": base_url,
        "document_level_summary": document_level_summary,
        "section_level_summary": section_level_summary,
        "pdf": pdf,
        "file_name": file_name,
        "raw_md_text": raw_md_text,
        "min_grained_level": str(min_grained_level),
        "max_grained_level": str(max_grained_level),
        "img_width": str(img_width),
        "threshold": str(threshold),
        "prompts": prompts,
        "summarizer_params": summarizer_params
    }
    replacements = dict_filter_none(replacements)

    html_content = replace_content(html_content, replacements)
    return html_content


async def post_form_request(request_url, request_param, files=None):
    async with aiohttp.ClientSession() as session:
        if files:
            form_data = aiohttp.FormData()
            for file in files:
                form_data.add_field('pdf_content',
                               file,
                               filename='document.pdf',
                               content_type='application/pdf')

            async with session.post(request_url, data=form_data) as response:
                json_info = await handle_async_api_response(response)
        else:
            async with session.post(request_url, data=request_param) as response:
                json_info = await handle_async_api_response(response)

    return json_info


async def align_text_with_images(
        text:str=...,
        pdf:str=...,
        save_dir:str=...,
        raw_md_text:str=None,
        min_grained_level:int=3,
        max_grained_level:int=4,
        img_width:int=400,
        threshold:float=0.8,
        margin:int=10,
        from_middleware:bool=False
        ):
    """Aligns document text with images extracted from the provided PDF."""
    extract_img_url = backend_url + "/extract_img/"
    async with aiohttp.ClientSession() as session:
        if from_middleware:
            pdf_path = url2path(url=pdf, base_url=GENERAL_CONFIG['middleware_url'] + "/index/",
                                base_dir=GENERAL_CONFIG["app_save_dir"])
            async with aiofiles.open(pdf_path, mode="rb") as f:
                pdf_content = await f.read()

            data = aiohttp.FormData()
            data.add_field('pdf_content',
                           BytesIO(pdf_content),
                           filename='document.pdf',
                           content_type='application/pdf')
            async with session.post(extract_img_url, data=data) as response:
                json_info = await handle_async_api_response(response, return_str=False)
        else:
            async with session.post(extract_img_url, data={"pdf": pdf}) as response:
                json_info = await handle_async_api_response(response, return_str=False)
    if json_info["status"] != "success":
        raise Exception(f"Image extraction error: {json_info['message']}")

    zip_data = json_info["message"]
    await extract_zip_from_bytes(zip_data=zip_data, extract_dir=os.path.join(save_dir, "img"))
    img_paths = dir2img_paths(img_dir=os.path.join(save_dir, "img"))
    alignment_url = backend_url + "/alignment/"
    alignment_params = {
        "text": text,
        "raw_md_text": raw_md_text,
        "min_grained_level": min_grained_level,
        "max_grained_level": max_grained_level,
        "img_width": img_width,
        "threshold": threshold,
        "margin": margin,
        "img_paths": img_paths
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(alignment_url, json=alignment_params) as response:
            json_info = await handle_async_api_response(response)
            if json_info["status"] != "success":
                raise Exception(f"Alignment error: {json_info['message']}")
    aligned_document_text = json_info["message"]
    aligned_document_text = img2url(text=aligned_document_text,
                                    base_url=GENERAL_CONFIG['middleware_url'],
                                    img_dir=None,
                                    absolute=True
                                    )
    return aligned_document_text
