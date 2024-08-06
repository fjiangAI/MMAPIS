from MMAPIS.tools import init_logging,GPT_Helper
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS,OPENAI_CONFIG
from typing import Union, List
from MMAPIS.tools import num_tokens_from_messages
import logging
import base64
from openai import OpenAI
import requests


class Img_QA_Generator(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {"model": "gpt-4-vision-preview"},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)
        if self.model != "gpt-4o":
            replaced_model = OPENAI_CONFIG["img_qa_model"]
            logging.warning(f"model {self.model} is not supported, will use {replaced_model} instead")
            self.model = replaced_model



    @staticmethod
    def encode_image(image_path):
        """
            Function to encode the image
        Args:
            image_path:

        Returns:

        """
        if image_path.startswith("http"):
            # Download the image from the URL
            response = requests.get(image_path)
            if response.status_code == 200:
                # Encode the image content
                return base64.b64encode(response.content).decode('utf-8')
            else:
                raise ValueError(f"Failed to download image from URL: {image_path}")

        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')


    def request_via_api(self,
                        user_input: str,
                        url_lst:List[str],
                        response_only: bool = True,
                        detailed_img:bool = False,
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
        if detailed_img:
            detail = "high"
        else:
            detail = "low"

        url = self.base_url + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            img_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_url)}","detail":f"{detail}"}}
                for img_url in url_lst
            ]
        except Exception as err:
            return False, f"Encode image error: {err}"
        img_content.append({"type": "text", "text": user_input})
        self.messages = [
            {
                "role": "user",
                "content":img_content
            }
        ]
        input_tokens = num_tokens_from_messages(self.messages, model=self.model,detailed_img=detailed_img)
        token_threshold = self.max_tokens * self.prompt_ratio
        if input_tokens > token_threshold:
            logging.warning(
                f'input tokens {input_tokens} is larger than max tokens {token_threshold}, will cut the input')
            diff = int(input_tokens - token_threshold)
            self.messages[-1]['content'] = self.messages[-1]['content'][:-diff]
        input_tokens = min(input_tokens, token_threshold)
        parameters = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": min(self.max_tokens - input_tokens,4096),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        response, content, flag = self.handle_request(url=url, parameters=parameters, headers=headers)


        if flag:
            self.messages.append({
                'role': response['choices'][0]['message']['role'],
                'content': response['choices'][0]['message']['content']
            })
        else:
            self.messages.append({
                'role': 'assistant',
                'content': content
            })
        if response_only:
            return flag, content
        else:
            return flag, self.messages

    def request_via_openai(
            self,
            user_input: str,
            url_lst: List[str],
            response_only: bool = True,
            detailed_img: bool = False,
            **kwargs):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        if detailed_img:
            detail = "high"
        else:
            detail = "low"

        img_content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_url)}", "detail": f"{detail}"}}
            for img_url in url_lst
        ]
        img_content.append({"type": "text", "text": user_input})
        self.messages = [
            {
                "role": "user",
                "content": img_content
            }
        ]
        input_tokens = num_tokens_from_messages(self.messages, model=self.model, detailed_img=detailed_img)
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
                self.messages.append({
                    'role': choices[0].message.role,
                    'content': choices[0].message.content
                })
            else:
                try:
                    msg = completion.message.content
                    self.messages.append({
                        'role': 'assistant',
                        'content': msg
                    })
                except Exception as err:
                    return (False, f'OpenAI API Response Error: no choices in response, got: {completion}')
        except Exception as err:
            return (False, f'OpenAI API error: {err}')

        if response_only:
            return (True, msg)
        else:
            return (True, self.messages)

    def request_img_api(self,
                         user_input: str,
                         url_lst: List[str],
                         response_only: bool = True,
                         detailed_img: bool = False,
                         **kwargs):
        """

        Args:
            user_input:
            url_lst:
            response_only:
            detailed_img:
            **kwargs:

        Returns:

        """
        # flag, messages = self.request_via_api(user_input=user_input, url_lst=url_lst, response_only=response_only,
        #                                       detailed_img=detailed_img, **kwargs)
        flag, messages = self.request_via_openai(user_input=user_input, url_lst=url_lst, response_only=response_only,
                                                  detailed_img=detailed_img, **kwargs)
        return flag, messages




if __name__ == "__main__":
    import os
    init_logging()
    api_key = OPENAI_CONFIG["api_key"]
    base_url = OPENAI_CONFIG["base_url"]
    img_qa_generator = Img_QA_Generator(api_key=api_key, base_url=base_url)
    user_input = "What is in the image?"
    url_lst = "../../../app_res/blog_md/tmp2m41h_9m/img/Model Architecture_0.png"
    print("abs path",os.path.abspath(url_lst),os.path.exists(url_lst))
    url_lst = [os.path.abspath(url_lst)]
    flag, messages = img_qa_generator.request_img_api(user_input=user_input,url_lst=url_lst)
    print("flag", flag)
    print("messages", messages)