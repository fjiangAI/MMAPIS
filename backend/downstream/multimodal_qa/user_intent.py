from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.config.config import  APPLICATION_PROMPTS,OPENAI_CONFIG
from typing import Union, List, Dict
from MMAPIS.backend.tools import num_tokens_from_messages
import logging
from openai import OpenAI
import re
from MMAPIS.backend.config.config import OPENAI_CONFIG
import reprlib

class UserIntent:
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = None,
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        self.user_intentor =  GPTHelper(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)
        self.user_intentor.check_model(model_type="json")


    def get_intend(self,
                   user_input:str,
                   prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                   response_only:bool = True,
                   reset_messages:bool = True):
        # Initialize the message history with system prompts
        self.user_intentor.init_messages("system", [prompts.get('intent_system',''),prompts.get('intent', '')])

        # Replace placeholder in prompt with the user's input
        user_input = prompts.get('intent_input','').replace('{user query}',user_input,1)

        # Call the API to get the response and handle errors if any arise
        flag, content = self.user_intentor.request_json_api(
                                        user_input=user_input,
                                        reset_messages=reset_messages,
                                        response_only=response_only)
        try:
            # Transform the response to a dictionary
            content = eval(content.replace("null", "None"))
            return flag,content
        except Exception as e:
            error_msg = f"transform content{content} to dict error: {e}"
            return False,error_msg

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        return_str = []
        for key, value in self.user_intentor.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"UserIntent({', '.join(return_str)})"





