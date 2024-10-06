from MMAPIS.backend.config.config import CONFIG, APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.config.config import DOCUMENT_PROMPTS,OPENAI_CONFIG
from typing import Union, List
import reprlib


class Regenerator:
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict=None,
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):
        self.regenerator = GPTHelper(api_key=api_key,
                                     base_url=base_url,
                                     model_config=model_config,
                                     proxy=proxy,
                                     prompt_ratio=prompt_ratio,
                                     **kwargs)

    def regeneration(self,
                     section_level_summary: Union[str, List[str]],
                     regeneration_prompts: dict = DOCUMENT_PROMPTS,
                     response_only: bool = True,
                     reset_messages: bool = True,
                     raw_marker: str = "Raw Integrate Content",
                     final_marker: str = "Final Integrate Content",
                     **kwargs):
        if isinstance(section_level_summary, List):
            section_level_summary = '\n'.join(section_level_summary)

        # Prepare system prompts and user input.
        system_messages = [regeneration_prompts.get("integrate_system", ''),regeneration_prompts.get("integrate", '')]
        self.regenerator.init_messages("system", system_messages)
        user_input = regeneration_prompts.get("integrate_input", '').replace('{summary chunk}', section_level_summary, 1)
        flag,content =  self.regenerator.request_text_api(user_input=user_input,
                                                          reset_messages=reset_messages,
                                                          response_only=response_only,
                                                          **kwargs)
        content = self.regenerator.filter_final_response(content,raw_marker = raw_marker,final_marker = final_marker)
        return flag,self.regenerator.format_headers(self.regenerator.clean_math_text(content))

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        return_str = []
        for key, value in self.regenerator.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"Regenerator({', '.join(return_str)})"