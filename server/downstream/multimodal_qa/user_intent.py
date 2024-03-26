from MMAPIS.tools import init_logging,GPT_Helper
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS
from typing import Union, List, Dict
from MMAPIS.tools import num_tokens_from_messages
import logging

class UserIntent(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {"model": "gpt-3.5-turbo-0125"},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)

    def request_api(self,
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
        if self.model != "gpt-3.5-turbo-0125":
            self.model = "gpt-3.5-turbo-0125"
            logging.warning(f"model {self.model} is not supported, will use gpt-3.5-turbo-0125 instead")

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
            return flag,content
        except Exception as e:
            error_msg = f"transform content{content} to dict error: {e}"
            return False,error_msg



if __name__ == "__main__":

    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    qa_prompts = APPLICATION_PROMPTS["multimodal_qa"]
    user_intent = UserIntent(api_key, base_url,prompt_ratio=0.6)
    print("user_intenter: ",user_intent)
    user_input = "what's the Figure4 about?"
    flag,content = user_intent.get_intend(user_input,prompts=qa_prompts)
    print("content: ",content)
    print("-"*100)



