from typing import List, Union
from fastapi.responses import ORJSONResponse, Response
import json
import logging
from fastapi import Body, Depends, HTTPException
from MMAPIS.backend.tools import bytes2io
import httpx
from MMAPIS.backend.config.config import GENERAL_CONFIG
import aiohttp


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

async def fetch_pdf_content(pdf_url: str) -> bytes:
    """
    Asynchronously fetch PDF content from a given URL.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        bytes: The content of the PDF file.

    Raises:
        HTTPException: If the PDF fetch fails.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as response:
            if response.status == 200:
                return await response.read()
            else:
                raise HTTPException(status_code=response.status, detail=f"Failed to fetch PDF from {pdf_url}")




# Helper function for TTS generation
async def generate_tts(generator, text: str):
    flag, bytes_data = generator.text2speech(text=text, return_bytes=True)
    if not flag:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {bytes_data.decode('utf-8') if isinstance(bytes_data, bytes) else bytes_data}")
    return bytes_data



def handle_api_response(response: Union[Response, ORJSONResponse]):
    """
    Handle and parse the API response based on the response type.

    This function processes different types of responses (ORJSONResponse, Response)
    and returns a standardized dictionary with 'status' and 'message' fields.
    If the response is not supported, an appropriate error message is returned.

    Args:
        response: A response object of type `Response` or `ORJSONResponse`.

    Returns:
        A dictionary containing the 'status' and 'message' fields.
    """

    try:
        if isinstance(response, ORJSONResponse):
            json_info = json.loads(response.body.decode("utf-8"))

            if isinstance(json_info.get("message"), bytes):
                json_info["message"] = json_info["message"].decode("utf-8")

        elif isinstance(response, Response):
            json_info = {
                "status": "success" if response.status_code == 200 else "error",
                "message": response.body
            }

        else:
            json_info = response.json()

    except Exception as e:
        json_info = {
            "status": "Unsupported response error",
            "message": str(e) if isinstance(e, Exception) else "Unsupported response type"
        }

    return json_info