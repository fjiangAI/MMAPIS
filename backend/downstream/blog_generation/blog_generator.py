from MMAPIS.backend.tools.chatgpt import GPTHelper
from typing import Union, List
import reprlib
from MMAPIS.backend.config.config import APPLICATION_PROMPTS

class BlogGenerator():
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict={},
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):
        self.blog_generator = GPTHelper(api_key=api_key,
                                        base_url=base_url,
                                        model_config=model_config,
                                        proxy=proxy,
                                        prompt_ratio=prompt_ratio,
                                        **kwargs)

    def blog_generation(self,
                        document_level_summary: str,
                        section_level_summary: Union[str, List[str]],
                        blog_prompts: dict = APPLICATION_PROMPTS["blog_prompts"],
                        reset_messages: bool = True,
                        response_only: bool = True,
                        raw_marker: str = "Raw Blog Content",
                        final_marker: str = "New Blog Content",
                        **kwargs):
        if isinstance(section_level_summary, List):
            section_level_summary = '\n'.join(section_level_summary)

        system_messages = [blog_prompts.get("blog_system", ''),blog_prompts.get("blog", '')]
        self.blog_generator.init_messages("system", system_messages)
        user_input = blog_prompts.get("app_input", '').replace('{article}', section_level_summary, 1).replace('{generated summary}', document_level_summary, 1)
        flag,content =  self.blog_generator.request_text_api(user_input=user_input,
                                              reset_messages=reset_messages,
                                              response_only=response_only,
                                              **kwargs)
        content = content if content.strip().startswith("# ") else "# Blog Script\n" + content
        content = self.blog_generator.filter_final_response(content,raw_marker= raw_marker,final_marker=final_marker)
        return flag,self.blog_generator.format_headers(self.blog_generator.clean_math_text(content))


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        return_str = []
        for key, value in self.blog_generator.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"BlogGenerator({', '.join(return_str)})"



