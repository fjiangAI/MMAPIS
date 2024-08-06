from MMAPIS.tools import init_logging,GPT_Helper
from MMAPIS.config.config import  APPLICATION_PROMPTS,OPENAI_CONFIG
from typing import Union, List, Dict
from MMAPIS.tools import num_tokens_from_messages
import logging
from openai import OpenAI
import re
from MMAPIS.config.config import OPENAI_CONFIG

class UserIntent(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {"model": "gpt-3.5-turbo-0125"},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)
        if self.model.startswith("gpt-4") and self.model != "gpt-4-turbo-2024-04-09":
            self.model = "gpt-4-1106-preview"
            logging.warning(f"model {self.model} is not supported, will use gpt-4-1106-preview instead")
        else:
            replaced_model = OPENAI_CONFIG["user_intent_model"]
            logging.warning(f"model {self.model} is not supported, will use {replaced_model} instead")
            self.model = replaced_model


    def request_via_openai(
                         self,
                         user_input:str,
                         system_messages:Union[str,List[str]] = "",
                         reset_messages:bool = True,
                         response_only:bool = True,
                         **kwargs):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        if system_messages:
            self.init_messages('system', system_messages)
        self.messages.append({'role': 'user', 'content': user_input})

        input_tokens = num_tokens_from_messages(self.messages, model=self.model)
        token_threshold = self.max_tokens * self.prompt_ratio
        if input_tokens > token_threshold:
            logging.warning(
                f'input tokens {input_tokens} is larger than max tokens {token_threshold}, will cut the input')
            diff = int(input_tokens - token_threshold)
            self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]
        input_tokens = min(input_tokens, token_threshold)
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=min(self.max_tokens - input_tokens,4096),
                temperature=self.temperature,
                top_p=self.top_p,
            )
            msg = None
            completion = dict(completion)
            choices = completion.get('choices', None)
            if choices:
                msg = choices[0].message.content
                json_pattern = re.compile(r'\{.*\}', re.DOTALL)
                res = json_pattern.search(msg)
                if res:
                    msg = res.group()
                else:
                    return (False, f'Expected JSON response, but got: {msg}')
            else:
                return (False, f'OpenAI API Response Error: no choices in response, got: {completion}')

        except Exception as err:
            return (False, f'OpenAI API error: {err}')
        if reset_messages:
            self.messages.pop(-1)
        else:
            # add text of response to messages
            self.messages.append({
                'role': choices[0].message.role,
                'content': choices[0].message.content
            })
        if response_only:
            return True, msg
        else:
            return True, self.messages


    def request_via_api(self,
                    user_input: str,
                    system_messages: Union[str, List[str]] = "",
                    response_only: bool = True,
                    reset_messages: bool = True,
                    **kwargs):
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


        url = self.base_url + "/chat/completions"


        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.messages.append({'role': 'user', 'content': user_input})
        input_tokens = num_tokens_from_messages(self.messages, model=self.model)
        token_threshold = self.max_tokens * self.prompt_ratio
        if input_tokens > token_threshold:
            logging.warning(
                f'input tokens {input_tokens} is larger than max tokens {token_threshold}, will cut the input')
            diff = int(input_tokens - token_threshold)
            self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]
        print("self.messages: ",self.messages)
        parameters = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": min(self.max_tokens - input_tokens,4096),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format":{"type": "json_object"},
        }

        response, content, flag = self.handle_request(url=url, parameters=parameters, headers=headers)

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

    def request_api(self,
                    user_input:str,
                    system_messages:Union[str,List[str]] = None,
                    response_only:bool = True,
                    reset_messages:bool = True,
                    **kwargs):
        return self.request_via_openai(
            user_input=user_input,
            system_messages=system_messages,
            reset_messages=reset_messages,
            response_only=response_only,
        )


    def get_intend(self,
                   user_input:str,
                   prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                   response_only:bool = True,
                   reset_messages:bool = True):
        self.init_messages("system", [prompts.get('intent_system',''),prompts.get('intent', '')])
        user_input = prompts.get('intent_input','').replace('{user query}',user_input,1)
        flag,content = self.request_api(user_input=user_input,
                                        reset_messages=reset_messages,
                                        response_only=response_only)
        try:
            content = content.replace("null", "None")
            content = eval(content)
            print("intent content:",content)
            return flag,content
        except Exception as e:
            error_msg = f"transform content{content} to dict error: {e}"
            return False,error_msg


if __name__ == "__main__":
    logger = init_logging()
    api_key = OPENAI_CONFIG["api_key"]
    base_url = OPENAI_CONFIG["base_url"]
    model_config = OPENAI_CONFIG["model_config"]

    user_intent = UserIntent(api_key=api_key, base_url=base_url, model_config=model_config,proxy=None)
    user_input = "What is the purpose of the study?"
    flag,content = user_intent.get_intend(user_input=user_input)
    print("flag:",flag,"content:",content)





