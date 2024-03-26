from MMAPIS.tools import init_logging,GPT_Helper
from MMAPIS.config.config import CONFIG, APPLICATION_PROMPTS
from typing import Union, List
from MMAPIS.tools import num_tokens_from_messages
import logging
import base64

class Img_QA_Generator(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {"model": "gpt-4-vision-preview"},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key, base_url, model_config, proxy, prompt_ratio, **kwargs)


    @staticmethod
    def encode_image(image_path):
        """
            Function to encode the image
        Args:
            image_path:

        Returns:

        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def request_img_api(self,
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
        if self.model != "gpt-4-vision-preview":
            self.model = "gpt-4-vision-preview"
            logging.warning(f"model {self.model} is not supported, will use gpt-4-vision-preview instead")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        img_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_url)}","detail":f"{detail}"}}
            for img_url in url_lst
        ]
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


if __name__ == "__main__":

    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    qa_prompts = APPLICATION_PROMPTS["multimodal_qa"]
    user_intent = Img_QA_Generator(api_key, base_url,prompt_ratio=0.6)
    print("user_intenter: ",user_intent)
    user_input = "What's in the picture?"
    title_list = ["Introduction", "Background of QA", "Proposed Method", "Eval & Results", "Conclusion"]
    flag,content = user_intent.request_img_api(
        user_input = user_input,
        url_lst=["./Model Architecture_1.png"]
    )

    print("content: \n",content)
