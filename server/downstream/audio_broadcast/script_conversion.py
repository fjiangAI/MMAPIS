from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
import reprlib
from MMAPIS.tools.tts import YouDaoTTSConverter
from MMAPIS.tools.utils import init_logging


class Broadcast_Generator(GPT_Helper):
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

    def broadcast_generation(self,
                          document_level_summary:str,
                          section_summaries:Union[str, List[str]],
                          broadcast_prompts:dict = None,
                          reset_messages:bool = True,
                          response_only:bool = True,
                          raw_marker:str = "Raw Broadcast Content",
                          final_marker:str = "New Broadcast Content",
                          **kwargs):
        if isinstance(section_summaries, List):
            section_summaries = '\n'.join(section_summaries)
        system_messages = [broadcast_prompts.get("broadcast_system", ''),broadcast_prompts.get("broadcast", '')]
        self.init_messages("system", system_messages)
        user_input = broadcast_prompts.get("app_input", '').replace('{article}', section_summaries, 1).replace('{generated summary}', document_level_summary, 1)
        flag,content =  self.summarize_text(text=user_input,
                                           reset_messages=reset_messages,
                                           response_only=response_only,
                                           **kwargs)
        content = self.filter_final_response(content,raw_marker = raw_marker,final_marker = final_marker)
        return flag,content

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """

        msg = f"Broadcast_GPT(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg

if __name__ == "__main__":
    logger = init_logging(logger_mode=LOGGER_MODES)
    user_input_path = "../integrate.md"
    with open(user_input_path, 'r') as f:
        user_input = f.read()

    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]


    section_summaries_path = "../summary.md"
    with open(section_summaries_path, 'r') as f:
        section_summaries = f.read()
    print("section_summaries: ", reprlib.repr(section_summaries))
    broadcast_prompts = APPLICATION_PROMPTS["broadcast_prompts"]
    broadcast_generator = Broadcast_Generator(api_key=api_key, base_url=base_url, model_config=model_config)
    flag,content = broadcast_generator.broadcast_generation(document_level_summary=user_input,
                                                          section_summaries=section_summaries,
                                                          broadcast_prompts=broadcast_prompts,
                                                          reset_messages=True,
                                                          response_only=True,
                                                            )
    print("flag: ", flag)
    print("content: ", content)