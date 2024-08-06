from MMAPIS.tools.chatgpt import GPT_Helper
from typing import Union, List
from MMAPIS.config.config import CONFIG,INTEGRATE_PROMPTS,OPENAI_CONFIG
import reprlib
import logging


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
                          min_length=OPENAI_CONFIG["min_length"],
                          compression_ratio=OPENAI_CONFIG["compression_ratio"],
                          max_regenerate=OPENAI_CONFIG["max_regenerate"],
                          **kwargs
                          ):
        if isinstance(section_summaries, List):
            section_summaries = ' '.join(section_summaries)
        system_prompt = [integrate_prompts.get('integrate_system', ''),integrate_prompts.get('integrate', '')]
        self.init_messages("system", system_prompt)
        user_input = integrate_prompts.get('integrate_input', '').replace('{summary chunk}', section_summaries, 1)
        section_summaries_len = self.count_words(section_summaries)
        flag,content =  self.summarize_text(text=user_input,
                                  system_messages=system_prompt,
                                   response_only=response_only,
                                   reset_messages=reset_messages)
        content = self.filter_final_response(content,
                                             raw_marker=raw_marker,
                                             final_marker=final_marker)
        num_regenerate = 0
        max_content_len = 0
        while not self.count_words(content) > section_summaries_len * compression_ratio and flag:
            logging.info(f"Due to compression ratio, regenerating content. Current length: {self.count_words(content)}|Current compression ratio{self.count_words(content)/section_summaries_len}|requred ratio:{compression_ratio}")
            if self.count_words(content) > min_length or num_regenerate >= max_regenerate:
                break
            self.init_messages("system",
                               [integrate_prompts.get('integrate_system', ''), integrate_prompts.get('integrate', '')])
            flag,regeneration_res = self.summarize_text(text=user_input,
                                                        response_only=True,
                                                        reset_messages=reset_messages)
            regeneration_res = self.filter_final_response(regeneration_res,
                                                            raw_marker=raw_marker,
                                                            final_marker=final_marker)
            if flag:
                num_regenerate += 1
                if self.count_words(regeneration_res) > max_content_len:
                    max_content_len = self.count_words(regeneration_res)
                    content = regeneration_res
            else:
                return flag, regeneration_res

        return flag,self.format_headers(self.clean_math_text(content))

    def count_words(self, text:str):
        return len(text.split())

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        msg = f"Summary_Integrator(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg





