from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
import reprlib
from MMAPIS.tools.tts import YouDaoTTSConverter
from MMAPIS.tools.utils import init_logging
import re

class Regenerator(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict={},
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         prompt_ratio=prompt_ratio,
                         **kwargs)

    def regeneration(self,
                      document_level_summary:str,
                      section_summaries:Union[str, List[str]],
                      regeneration_prompts:dict = None,
                      reset_messages:bool = True,
                      response_only:bool = True,
                      raw_marker:str = "",
                      final_marker:str = "New Regenerate Content",
                      **kwargs):
        if isinstance(section_summaries, List):
            section_summaries = '\n'.join(section_summaries)
        system_messages = [regeneration_prompts.get("regenerate_system", ''),regeneration_prompts.get("regenerate", '')]
        self.init_messages("system", system_messages)
        user_input = regeneration_prompts.get("app_input", '').replace('{article}', section_summaries, 1).replace('{generated summary}', document_level_summary, 1)
        flag,content =  self.summarize_text(text=user_input,
                                           reset_messages=reset_messages,
                                           response_only=response_only,
                                           **kwargs)
        content = self.filter_final_response(content,raw_marker = raw_marker,final_marker = final_marker)
        print("content is :",content)
        pattern = re.compile(r'##+.*?\n+')
        st_pattern = pattern.search(content)
        if st_pattern:
            content = content[st_pattern.start():]
        return flag,content

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Regenerator(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg