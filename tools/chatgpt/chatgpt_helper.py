from MMAPIS.tools.chatgpt.llm_helper import LLMSummarizer
import multiprocessing
import json
import requests
import reprlib
import os
import openai
from MMAPIS.tools.utils import num_tokens_from_messages
from typing import Union, List
import logging
from MMAPIS.config.config import CONFIG
import re


class GPT_Helper(LLMSummarizer):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict={},
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):

        super().__init__(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy,prompt_ratio=prompt_ratio, **kwargs)


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        msg = f"GPT_Helper(api_key:{reprlib.repr(self.api_key)},base_url:{self.base_url},model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"
        return msg


    # def init_openai_connect(self,host = None):
    #     openai.api_key = self.api_key
    #     if host:
    #         os.environ["http_proxy"] = host
    #         os.environ["https_proxy"] = host

    def init_messages(self,role:str,content:Union[str,List[str]]):
        # init openai role
        self.messages = []
        if isinstance(content, str):
            content = [content]
        for c in content:
            self.messages.append({'role': role, 'content': c})


    # def request_via_openai(
    #                      self,
    #                      user_input:str,
    #                      proxy:str = None,
    #                      system_messages:Union[str,List[str]] = "",
    #                      reset_messages:bool = True,
    #                      response_only:bool = True):
    #
    #     self.init_openai_connect(proxy)
    #     self.init_messages('system', system_messages)
    #     self.messages.append({'role': 'user', 'content': user_input})
    #     input_tokens = num_tokens_from_messages(self.messages, model=self.model)
    #     if input_tokens > self.max_tokens * self.prompt_ratio:
    #        logging.warning(f'input tokens {input_tokens} is larger than max tokens {self.max_tokens* self.prompt_ratio}, will cut the input')
    #        diff = int(input_tokens - self.max_tokens * self.prompt_ratio)
    #        self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]
    #    # use openai api
    #     response = openai.ChatCompletion.create(
    #         model= self.model,
    #         messages=self.messages,
    #         temperature= self.temperature,
    #         max_tokens= self.max_tokens-input_tokens,
    #         top_p= self.top_p,
    #         frequency_penalty= self.frequency_penalty,
    #         presence_penalty= self.presence_penalty
    #     )
    #
    #     if reset_messages:
    #         # pop user input
    #         self.messages.pop(-1)
    #     else:
    #         # add response to messages
    #         self.messages.append({
    #             'role': response['choices'][0]['message']['role'],
    #             'content': response['choices'][0]['message']['content']
    #         })
    #     if  response_only:
    #         return response['choices'][0]['message']['content']
    #     else:
    #         return self.messages


    def request_via_api(self,
                    user_input:str,
                    system_messages:Union[str,List[str]] = None,
                    response_only:bool = True,
                    reset_messages:bool = True,
                    return_the_same:bool = False,
                    session_messages:Union[str,List[str]] = None,
                    ):
        """

        Args:
            parameters: model info,e.g.
                        parameters = {
                            "model": self.model_name,
                            "messages": messages
                            }
            response_only:boolean, if True, only return response content, else return messages
            reset_messages: boolean, if True, reset messages to system , else will append messages
        Returns:
            flag: boolean, use to

        """
        messages = self.messages.copy()

        if isinstance(system_messages, str):
            system_messages = [system_messages]
        # Default self.messages to session_messages or an empty list
        if session_messages:
            print("copy session_messages")
            messages = session_messages[:]
            if system_messages:
                print("extend system_messages")
                messages.extend([{'role': 'system', 'content': c} for c in system_messages])
        else:
            if system_messages:
                print("init system_messages")
                messages.extend([{'role': 'system', 'content': c} for c in system_messages])

        if return_the_same:
            content = user_input
            return True, content

        else:
            url = self.base_url+"/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            messages.append({'role': 'user', 'content': user_input})
            input_tokens = num_tokens_from_messages(messages, model=self.model)
            token_threshold = self.max_tokens * self.prompt_ratio

            if input_tokens > token_threshold:
                logging.warning(f'input tokens {input_tokens} is larger than max tokens {token_threshold}, will cut the input')
                diff = int(input_tokens - token_threshold)
                messages[-1]['content'] = messages[-1]['content'][:-diff]
            parameters = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens-input_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }

            response,content, flag = self.handle_request(url = url,parameters=parameters,headers=headers)

        if reset_messages:
            messages.pop(-1)
            # self.init_messages('system', self.summary_prompts['system'])
        else:
            # add text of response to messages
            if flag:
                messages.append({
                    'role': response['choices'][0]['message']['role'],
                    'content': response['choices'][0]['message']['content']
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': content
                })
        if response_only:
            return flag, content
        else:
            return flag, messages


    def request_api(self,
                    user_input:str,
                    system_messages:Union[str,List[str]] = None,
                    response_only:bool = True,
                    reset_messages:bool = True,
                    return_the_same:bool = False,
                    session_messages:Union[str,List[str]] = None,
                    ):
        """
        :param user_input:
        :param system_messages:
        :param response_only:
        :param reset_messages:
        :return:
        """
        return self.request_via_api(user_input = user_input,
                                    system_messages= system_messages,
                                    response_only= response_only,
                                    reset_messages= reset_messages,
                                    return_the_same= return_the_same,
                                    session_messages= session_messages,
                                    )



    def summarize_text(self,
                       text:str,
                       system_messages:Union[str,List[str]] = None,
                       response_only:bool = True,
                       reset_messages:bool = True,
                       return_the_same:bool = False,
                       session_messages:Union[str,List[str]] = None,
                       ):
        """
        Use ChatGPT to summarize a given text.

        :param text: The text to be summarized.
        :return: The summarized text.
        """
        # Implement the summarization logic
        # This might involve formatting the prompt in a specific way to ask for a summary

        flag, summarized_text = self.request_api(user_input=text,
                                                 system_messages=system_messages,
                                                 response_only=response_only,
                                                 reset_messages=reset_messages,
                                                 return_the_same=return_the_same,
                                                 session_messages=session_messages,
                                                )
        return flag, summarized_text

    def filter_final_response(self,resps:List[str],raw_marker:str,final_marker:str ):
        """
        filter out the final response
        Args:
            resps: list of response
            raw_marker: marker of raw response
            final_marker: marker of final response

        Returns: final response

        """
        is_list = True
        if isinstance(resps,str):
            resps = [resps]
            is_list = False
        if not isinstance(resps,list):
            raise ValueError(f'resps should be list or str, but got {type(resps)}')

        craft_summary_pattern = re.compile(rf"\[{raw_marker}\]|`{raw_marker}`|{raw_marker}", re.IGNORECASE)
        final_summary_pattern = re.compile(rf"\[{final_marker}\]|`{final_marker}`|{final_marker}", re.IGNORECASE)
        res = []
        for resp in resps:
            final_summary_group = final_summary_pattern.search(resp)
            craft_summary_group = craft_summary_pattern.search(resp)

            if final_summary_group:
                res.append(self.clean_resp(resp[final_summary_group.end():].strip()))
            elif craft_summary_group:
                res.append(self.clean_resp(resp[craft_summary_group.end():].strip()))
            else:
                res.append(self.clean_resp(resp.strip()))

        if not is_list:
            return res[0]

        return res

    @staticmethod
    def clean_resp(raw_text_list: Union[List[str], str]):
        """
        remove the special marker in head of response or the tail of response
        Args:
            raw_text_list: list of raw response

        Returns: list of cleaned response

        """
        is_list = True
        if isinstance(raw_text_list, str):
            raw_text_list = [raw_text_list]
            is_list = False
        if not isinstance(raw_text_list, list):
            raise ValueError(f'raw_text_list should be list or str, but got {type(raw_text_list)}')

        cleaned_str_list = []
        for raw_text in raw_text_list:
            cleaned_str = re.sub(r'^[^a-zA-Z.#]*', '', raw_text)
            cleaned_str = re.sub(r'[^a-zA-Z.#]*$', '', cleaned_str)
            cleaned_str_list.append(cleaned_str)
        if not is_list:
            return cleaned_str_list[0]
        return cleaned_str_list




if __name__ == "__main__":
    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    
    chatgpt = GPT_Helper(api_key=api_key, base_url=base_url, model_config=model_config)
    print(chatgpt)
    flag,summary = chatgpt.summarize_text("This is a test", system_messages="This is a test",reset_messages=False,response_only=False)
    if flag:
        print("summary: ", summary)
    else:
        print("error:",summary)
