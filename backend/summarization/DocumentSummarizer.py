from MMAPIS.backend.tools.chatgpt import GPTHelper
from typing import Union, List
from MMAPIS.backend.config.config import CONFIG,DOCUMENT_PROMPTS,OPENAI_CONFIG
import reprlib
import logging


class DocumentSummarizer:
    def __init__(self,
                api_key,
                base_url,
                model_config:dict={},
                proxy:dict = None,
                prompt_ratio:float = 0.8,
                **kwargs):
        self.summarizer = GPTHelper(api_key=api_key,
                                    base_url=base_url,
                                    model_config=model_config,
                                    proxy=proxy,
                                    prompt_ratio=prompt_ratio,
                                    **kwargs
                                    )

    def integrate_summary(self,
                          section_level_summary:Union[str, List[str]],
                          document_prompts:dict = DOCUMENT_PROMPTS,
                          response_only:bool = True,
                          reset_messages:bool = True,
                          raw_marker:str = "Raw Integrate Content",
                          final_marker:str = "Final Integrate Content",
                          min_length=OPENAI_CONFIG["min_length"],
                          compression_ratio=OPENAI_CONFIG["compression_ratio"],
                          max_regenerate=OPENAI_CONFIG["max_regenerate"],
                          **kwargs):
        """
        Integrate summaries of document sections.

        This method takes summaries of document sections and attempts to generate an integrated summary that meets
        certain compression ratio requirements. If the generated summary does not meet the requirements, it may regenerate
        the summary up to a specified number of times.

        :param section_level_summary: Summaries of document sections. Can be a string or a list of strings.
        :param document_prompts: Dictionary containing prompts for integration.
        :param response_only: Whether to return only the response from the language model.
        :param reset_messages: Whether to reset messages after each request.
        :param raw_marker: Marker for raw integrated content.
        :param final_marker: Marker for final integrated content.
        :param min_length: Minimum length for the generated summary.
        :param compression_ratio: Desired compression ratio for the integrated summary.
        :param max_regenerate: Maximum number of times to regenerate the summary.
        :param kwargs: Additional keyword arguments.
        :return: A tuple containing a flag indicating success and the integrated summary.
        """
        # If section_level_summary is a list, join it into a single string.
        if isinstance(section_level_summary, List):
            section_level_summary = '\n'.join(section_level_summary)

        # Prepare system prompts and user input.
        system_prompt = [document_prompts.get('integrate_system', ''), document_prompts.get('integrate', '')]
        user_input = document_prompts.get('integrate_input', '').replace('{summary chunk}', section_level_summary, 1)

        # Get the length of the input summaries.
        section_level_summary_len = self.count_words(section_level_summary)

        # Request text from the language model.
        flag, content = self.summarizer.request_text_api(
            user_input=user_input,
            system_messages=system_prompt,
            response_only=response_only,
            reset_messages=reset_messages
        )
        # Filter and format the response.
        content = self.summarizer.filter_final_response(content, raw_marker=raw_marker, final_marker=final_marker)

        return self.handle_regeneration(content, section_level_summary_len, system_prompt, user_input, response_only, reset_messages, min_length, compression_ratio, max_regenerate)


    def count_words(self, text:str):
        return len(text.split())

    def handle_regeneration(self,
                            content,
                            section_level_summary_len,
                            system_prompt,
                            user_input,
                            response_only,
                            reset_messages,
                            min_length,
                            compression_ratio,
                            max_regenerate):
        num_regenerate = 0
        max_content_len = 0
        flag = True
        while not self.count_words(content) > section_level_summary_len * compression_ratio:
            if self.count_words(content) > min_length or num_regenerate >= max_regenerate:
                break
            logging.info(
                f"Due to compression ratio, regenerating content. Current length: {self.count_words(content)}|Current compression ratio: {self.count_words(content) / section_level_summary_len}|Required ratio: {compression_ratio}")
            self.summarizer.init_messages("system", system_prompt)
            flag, regeneration_res = self.summarizer.request_text_api(
                user_input=user_input,
                response_only=response_only,
                reset_messages=reset_messages,
            )
            if not flag:
                return flag, regeneration_res
            else:
                regeneration_res = self.summarizer.filter_final_response(regeneration_res,
                                                                         raw_marker="Raw Integrate Content",
                                                                         final_marker="Final Integrate Content")
                num_regenerate += 1
                cur_len = self.count_words(regeneration_res)
                if  cur_len > max_content_len:
                    max_content_len = cur_len
                    content = regeneration_res

        return flag, self.summarizer.format_headers(self.summarizer.clean_math_text(content))

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        msg = f"Summary_Integrator(api_key:{reprlib.repr(self.summarizer.api_key)},base_url:{self.summarizer.base_url}," \
              f"model:{self.summarizer.model}, temperature:{self.summarizer.temperature}, max_tokens:{self.summarizer.max_tokens}, top_p:{self.summarizer.top_p}, frequency_penalty:{self.summarizer.frequency_penalty}, presence_penalty:{self.summarizer.presence_penalty})"
        return msg





