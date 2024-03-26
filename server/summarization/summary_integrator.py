from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
from MMAPIS.config.config import CONFIG,INTEGRATE_PROMPTS
import reprlib

class Summary_Integrator(GPT_Helper):
    def __int__(self,
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

    def integrate_summary(self,
                          section_summaries:Union[str, List[str]],
                          integrate_prompts:dict = None,
                          response_only:bool = True,
                          reset_messages:bool = True,
                          raw_marker:str = "Raw Integrate Content",
                          final_marker:str = "Final Integrate Content",
                          **kwargs
                          ):
        if isinstance(section_summaries, List):
            section_summaries = ' '.join(section_summaries)
        self.init_messages("system", [integrate_prompts.get('integrate_system', ''),integrate_prompts.get('integrate', '')])
        user_input = integrate_prompts.get('integrate_input', '').replace('{summary chunk}', section_summaries, 1)

        flag,content =  self.summarize_text(text=user_input,
                                   response_only=response_only,
                                   reset_messages=reset_messages)
        content = self.filter_final_response(content,
                                             raw_marker=raw_marker,
                                             final_marker=final_marker)
        return flag,content

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        msg = f"Summary_Integrator(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg




