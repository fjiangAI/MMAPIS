from MMAPIS.backend.config.config import CONFIG, APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.backend.tools.chatgpt import GPTHelper
from typing import Union, List
import reprlib



class BroadcastScriptGenerator():
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict={},
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):
        self.broadcast_generator = GPTHelper(api_key=api_key,
                                             base_url=base_url,
                                             model_config=model_config,
                                             proxy=proxy,
                                             prompt_ratio=prompt_ratio,
                                             **kwargs)

    def broadcast_generation(self,
                          document_level_summary:str,
                          section_level_summary:Union[str, List[str]],
                          broadcast_prompts:dict = None,
                          reset_messages:bool = True,
                          response_only:bool = True,
                          raw_marker:str = "Raw Broadcast Content",
                          final_marker:str = "New Broadcast Content",
                          **kwargs):
        if isinstance(section_level_summary, List):
            section_level_summary = '\n'.join(section_level_summary)
        system_messages = [broadcast_prompts.get("broadcast_system", ''),broadcast_prompts.get("broadcast", '')]
        self.broadcast_generator.init_messages("system", system_messages)
        user_input = broadcast_prompts.get("app_input", '').replace('{article}', section_level_summary, 1).replace('{generated summary}', document_level_summary, 1)
        flag,content =  self.broadcast_generator.request_text_api(user_input=user_input,
                                                                  reset_messages=reset_messages,
                                                                  response_only=response_only,
                                                                  **kwargs)
        content = self.broadcast_generator.filter_final_response(content,raw_marker = raw_marker,final_marker = final_marker)
        return flag,content

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Broadcast_GPT(api_key:{reprlib.repr(self.broadcast_generator.api_key)},base_url:{self.broadcast_generator.base_url},model:{self.broadcast_generator.model}, temperature:{self.broadcast_generator.temperature}, max_tokens:{self.broadcast_generator.max_tokens}, top_p:{self.broadcast_generator.top_p}, frequency_penalty:{self.broadcast_generator.frequency_penalty}, presence_penalty:{self.broadcast_generator.presence_penalty})"
        return msg

