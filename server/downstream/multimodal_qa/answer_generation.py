import numpy as np

from MMAPIS.tools import init_logging, GPT_Helper
from MMAPIS.server.summarization import Article
from typing import Tuple, Union, Dict
import requests
import spacy

class MultiModal_QA_Generator(GPT_Helper):
    def __init__(self, api_key, base_url, model_config: dict = {}, proxy: dict = None, **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         **kwargs)

    def QA_Generation(self,
                      prompts: Dict,
                      user_intent:Union[str, Tuple[str]],
                      user_input: str,
                      article: Article,
                      response_only: bool = True,
                      reset_messages: bool = True):
        if isinstance(user_intent, str):
            return True, user_input
        else:
            if user_intent[0] == "figure":
                if len(user_intent) == 2:
                    index = user_intent[1]
                    for section in article.sections:
                        if len(section.img_paths) > index:
                            target_img = section.img_paths[index]
                            break
                        else:
                            index -= len(section.img_paths)
                    self.init_messages("system", prompts.get('qa_system', ''))
                    user_input = prompts.get('qa_input', '').replace('{user query}', user_input, 1)

                else:
                    user_input = prompts.get('qa_input', '').replace('{user query}', user_input, 1).replace('{title list}', str(article.title_list), 1)

            elif user_intent[0] == "text":
                user_input = prompts.get('qa_input', '').replace('{user query}', user_input, 1).replace('{title list}', str(article.title_list), 1)

            self.init_messages("system", [prompts.get('qa_system', ''), prompts.get('qa', '')])
            flag, content = self.request_api(user_input=user_input,
                                             reset_messages=reset_messages,
                                             response_only=response_only)
            return flag, content


    @staticmethod
    def find_target_section(target_title:str,
                            article:Article):
        nlp = spacy.load("en_core_web_lg")
        sim_ls = []
        for id, section in enumerate(article.sections):
            temp_sim_ls = []
            for i, section_name in enumerate(section.subtitles):
                temp_sim_ls.append(nlp(section_name).similarity(nlp(target_title)))
                if target_title in section_name:
                    return section if i == 0 else section.subsubgroup[i-1]
            sim_ls.append(temp_sim_ls)
        arr = np.array(sim_ls)
        max_index = np.unravel_index(np.argmax(arr), arr.shape)
        if max_index[1] == 0:
            return article.sections[max_index[0]]
        else:
            return article.sections[max_index[0]].subsubgroup[max_index[1]-1]






    def img_qa_requeset(self,
                    user_input:str,
                    system_messages:Union[str,List[str]] = "",
                    response_only:bool = True,
                    reset_messages:bool = True,
                    return_the_same:bool = False,
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
        if system_messages:
            self.init_messages('system', system_messages)

        if return_the_same:
            content = user_input
            return True, content

        else:
            url = self.base_url+"/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            self.messages.append({'role': 'user', 'content': user_input})
            input_tokens = num_tokens_from_messages(self.messages, model=self.model)
            parameters = {
                "model": self.model,
                "messages": self.messages,
                "max_tokens": self.max_tokens-input_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }

            response,content, flag = self.handle_request(url = url,parameters=parameters,headers=headers)

        if reset_messages:
            self.messages.pop(-1)
            # self.init_messages('system', self.summary_prompts['system'])
        else:
            # add text of response to messages
            if flag:
                self.messages.append({
                    'role': response['choices'][0]['message']['role'],
                    'content': response['choices'][0]['message']['content']
                })
        if response_only:
            return flag, content
        else:
            return flag, self.messages


