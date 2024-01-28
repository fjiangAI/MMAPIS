from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
import reprlib
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS
from MMAPIS.tools.tts import YouDaoTTSConverter

class Blog_Generator(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict={},
                 proxy:dict = None,
                 **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         **kwargs)

    def blog_generation(self,
                     document_level_summary: str,
                     section_summaries: Union[str, List[str]],
                     blog_prompts: dict = None,
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
        print("raw content:",content)
        return flag,self.filter_final_response(content,raw_marker= raw_marker,final_marker=final_marker)


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Blog_GPT(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg

if __name__ == "__main__":

    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    blog_prompts = APPLICATION_PROMPTS["blog_prompts"]
    user_input_path = "../integrate.md"
    with open(user_input_path, 'r') as f:
        user_input = f.read()
    section_summaries_path = "../summary.md"
    with open(section_summaries_path, 'r') as f:
        section_summaries = f.read()
    blog_generator = Blog_Generator(api_key=api_key,
                                    base_url=base_url,
                                    model_config=model_config)
    flag,response = blog_generator.blog_generation(
                                    document_level_summary=user_input,
                                    section_summaries=section_summaries,
                                    blog_prompts=blog_prompts,
                                    reset_messages=True,
                                    response_only=True,
                                    )
    print("flag:",flag)
    print("response:",response)
    with open('blog.md','w') as f:
        f.write(response)