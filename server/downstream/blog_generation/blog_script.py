from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
import reprlib
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS
from MMAPIS.tools.tts import YouDaoTTSConverter

class Blog_Script_Generator(GPT_Helper):
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

    def blog_script_generation(self,
                     document_level_summary: str,
                     section_summaries: Union[str, List[str]],
                     blog_prompts: dict = APPLICATION_PROMPTS["blog_prompts"],
                     reset_messages: bool = True,
                     response_only: bool = True,
                     raw_marker: str = "Raw Blog Content",
                     final_marker: str = "New Blog Content",
                     **kwargs):
        if isinstance(section_summaries, List):
            section_summaries = '\n'.join(section_summaries)

        system_messages = [blog_prompts.get("blog_system", ''),blog_prompts.get("blog", '')]
        self.init_messages("system", system_messages)
        user_input = blog_prompts.get("app_input", '').replace('{article}', section_summaries, 1).replace('{generated summary}', document_level_summary, 1)
        flag,content =  self.summarize_text(text=user_input,
                                   reset_messages=reset_messages,
                                   response_only=response_only,
                                   **kwargs)
        return flag,self.filter_final_response(content,raw_marker= raw_marker,final_marker=final_marker)


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Blog_GPT(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg

