from MMAPIS.backend.tools.chatgpt.llm_helper import LLMSummarizer
import multiprocessing
import json
import requests
import reprlib
import os
from openai import  OpenAI
from MMAPIS.backend.tools.utils import num_tokens_from_messages,init_logging
from MMAPIS.backend.config import LOGGER_MODES
from typing import Union, List
import logging
from MMAPIS.backend.config.config import CONFIG,OPENAI_CONFIG
import re
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import base64
from typing import Dict, List, Tuple
from PIL import Image
from io import BytesIO

init_logging()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_MODES)

class GPTHelper(LLMSummarizer):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config:dict=None,
                 proxy:dict = None,
                 prompt_ratio:float = 0.8,
                 **kwargs):

        super().__init__(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy,prompt_ratio=prompt_ratio, **kwargs)


    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        return_str = []
        for key, value in self.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"GPTHelper({', '.join(return_str)})"


    # def init_openai_connect(self,host = None):
    #     if host:
    #         os.environ["http_proxy"] = host
    #         os.environ["https_proxy"] = host

    def check_model(self, model_type:str):
        """
        check if the model is supported
        Args:
            model_type: model type

        Returns: bool

        """
        model_map = {
            "json": {
                "gpt-4":"gpt-4o-2024-08-06",
                "default": OPENAI_CONFIG["json_qa_model"],
            },
            "img": {
                "gpt-4":"gpt-4o",
                "default": OPENAI_CONFIG["img_qa_model"],
            },
            "tts": {
                "gpt-4":"tts-1",
                "default": OPENAI_CONFIG["tts_model"],
            }
        }
        if model_type not in model_map:
            return False
        model_choice = model_map[model_type]
        if self.model.startswith("gpt-4") and self.model != model_choice["gpt-4"]:
            logger.warning(f"model {self.model} is not supported, will use {model_choice['gpt-4']} instead")
            self.model = model_choice["gpt-4"]
        elif self.model != model_choice["default"]:
            logger.warning(f"model {self.model} is not supported, will use {model_choice['default']} instead")
            self.model = model_choice["default"]

    def init_messages(self,role:str,content:Union[str,List[str]]):
        # init openai role
        self.messages = []
        if isinstance(content, str):
            content = [content]
        for c in content:
            self.messages.append({'role': role, 'content': c})


    def _prepare_messages(self,
                          user_input: str,
                          system_messages: Union[str, List[str]],
                          session_messages: Union[str, List[str]] = None) -> List[Dict[str, str]]:
        """
        Prepare the message list for the API request.

        Args:
            user_input (str): The user's input message.
            system_messages (Union[str, List[str]]): System messages to be included.
            session_messages (Union[str, List[str]]): Existing session messages.

        Returns:
            List[Dict[str, str]]: Prepared list of messages for the API request.
        """
        messages = session_messages[:] if session_messages else self.messages.copy()

        if isinstance(system_messages, str):
            system_messages = [system_messages]

        if system_messages:
            messages.extend([{'role': 'system', 'content': c} for c in system_messages])

        messages.append({'role': 'user', 'content': user_input})

        return messages

    def _truncate_input(self, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
        """
        Truncate the input if it exceeds the token threshold.

        Args:
            messages (List[Dict[str, str]]): The list of messages.

        Returns:
            Tuple[List[Dict[str, str]], int]: Truncated messages and input token count.
        """
        input_tokens = num_tokens_from_messages(messages, model=self.model)
        token_threshold = int(self.max_tokens * self.prompt_ratio)

        if input_tokens > token_threshold:
            logger.warning(
                f'Input tokens ({input_tokens}) exceed max tokens ({token_threshold}). Truncating input.')
            diff = input_tokens - token_threshold
            messages[-1]['content'] = messages[-1]['content'][:-diff]
            input_tokens = token_threshold

        return messages, input_tokens


    def _process_response(self,
                          response: Dict,
                          messages: List[Dict[str, str]],
                          reset_messages: bool,
                          response_only: bool,
                          request_by_openai:bool = True,
                          ) -> Tuple[bool, Union[str, List[Dict[str, str]]]]:
        """
        Process the API response and update messages if necessary.

        Args:
            response (Dict): The API response.
            messages (List[Dict[str, str]]): The current message list.
            reset_messages (bool): Whether to reset messages after the response.
            response_only (bool): Whether to return only the response content.

        Returns:
            Tuple[bool, Union[str, List[Dict[str, str]]]]: Success flag and either the response content or updated messages.
        """

        if request_by_openai:
            role = response['choices'][0].message.role
            msg = response['choices'][0].message.content
        else:
            role = response['choices'][0]['message']['role']
            msg = response['choices'][0]['message']['content']

        if reset_messages:
            messages.pop(-1)
        else:
            messages.append({
                'role': role,
                'content': msg
            })

        if response_only:
            return True, msg
        else:
            return True, messages

    def request_via_openai(self, user_input: str, system_messages: Union[str, List[str]] = "",
                           reset_messages: bool = True, response_only: bool = True,
                           return_the_same: bool = False, session_messages: Union[str, List[str]] = None) -> Tuple[
        bool, Union[str, List[Dict[str, str]]]]:
        """
        Send a request to the OpenAI API using the OpenAI client library.

        Args:
            user_input (str): The user's input message.
            system_messages (Union[str, List[str]]): System messages to be included.
            reset_messages (bool): Whether to reset messages after the response.
            response_only (bool): Whether to return only the response content.
            return_the_same (bool): Whether to return the input without processing.
            session_messages (Union[str, List[str]]): Existing session messages.

        Returns:
            Tuple[bool, Union[str, List[Dict[str, str]]]]: Success flag and either the response content or updated messages.
        """
        import time
        if return_the_same:
            return True, user_input

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # Copy the messages to avoid modifying the original list when multiple requests are made
        messages = self._prepare_messages(user_input, system_messages, session_messages)
        messages, input_tokens = self._truncate_input(messages)
        logger.debug(f"request_via_openai: messages(input_tokens: {input_tokens}): {messages}")
        logger.debug("--" * 20)
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=min(self.max_tokens - input_tokens, self.max_output_tokens),
                temperature=self.temperature,
                top_p=self.top_p,
            )
            completion_dict = dict(completion)
            choices = completion_dict.get('choices', None)
            if not choices:
                return False, f'OpenAI API Response Error: no choices in response, got: {completion_dict}'

            return self._process_response(completion_dict,
                                          messages,
                                          reset_messages,
                                          response_only)

        except Exception as err:
            return False, f'OpenAI request error: {err}'


    def request_via_api(self, user_input: str, system_messages: Union[str, List[str]] = None,
                        response_only: bool = True, reset_messages: bool = True,
                        return_the_same: bool = False, session_messages: Union[str, List[str]] = None) -> Tuple[bool, Union[str, List[Dict[str, str]]]]:
        """
        Send a request to the OpenAI API using a custom API call.

        Args:
            user_input (str): The user's input message.
            system_messages (Union[str, List[str]]): System messages to be included.
            response_only (bool): Whether to return only the response content.
            reset_messages (bool): Whether to reset messages after the response.
            return_the_same (bool): Whether to return the input without processing.
            session_messages (Union[str, List[str]]): Existing session messages.

        Returns:
            Tuple[bool, Union[str, List[Dict[str, str]]]]: Success flag and either the response content or updated messages.
        """
        if return_the_same:
            return True, user_input
        messages = self._prepare_messages(user_input, system_messages, session_messages)
        messages, input_tokens = self._truncate_input(messages)

        url = f"{self.base_url}/chat/completions/"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        parameters = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_tokens - input_tokens, self.max_output_tokens),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        response, content, flag = self.handle_request(url=url, parameters=parameters, headers=headers)

        if not flag:
            if response_only:
                return flag, f'OpenAI request error: {content}'
            else:
                messages.append({'role': 'assistant', 'content': content})
            return flag, messages

        return self._process_response(response,
                                      messages,
                                      reset_messages,
                                      response_only,
                                      request_by_openai=False)



    def request_text_api(self,
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
        # return self.request_via_api(user_input = user_input,
        #                             system_messages= system_messages,
        #                             response_only= response_only,
        #                             reset_messages= reset_messages,
        #                             return_the_same= return_the_same,
        #                             session_messages= session_messages,
        #                             )
        return self.request_via_openai(user_input = user_input,
                                    system_messages= system_messages,
                                    response_only= response_only,
                                    reset_messages= reset_messages,
                                    return_the_same= return_the_same,
                                    session_messages= session_messages,
                                    )

    def request_json_api(self, user_input: str,
                         system_messages: Union[str, List[str]] = "",
                         reset_messages: bool = True,
                         response_only: bool = True,
                         **kwargs) -> Tuple[bool, Union[str, List[Dict[str, str]]]]:
        """
        Send a request to the OpenAI API and extract JSON from the response.

        Args:
            user_input (str): The user's input message.
            system_messages (Union[str, List[str]]): System messages to be included.
            reset_messages (bool): Whether to reset messages after the response.
            response_only (bool): Whether to return only the response content.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[bool, Union[str, List[Dict[str, str]]]]: Success flag and either the JSON response or updated messages.
        """
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = self._prepare_messages(user_input, system_messages)
        messages, input_tokens = self._truncate_input(messages)
        logger.debug(f"request_json_api: messages(input_tokens: {input_tokens}): {messages}")
        logger.debug("--" * 20)
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=min(self.max_tokens - input_tokens, self.max_output_tokens),
                temperature=self.temperature,
                top_p=self.top_p,
            )
            completion_dict = dict(completion)
            choices = completion_dict.get('choices', None)

            if not choices:
                return False, f'OpenAI API Response Error: no choices in response, got: {completion_dict}'

            msg = choices[0].message.content
            json_pattern = re.compile(r'\{.*\}', re.DOTALL)
            res = json_pattern.search(msg)

            if not res:
                return False, f'Expected JSON response, but got: {msg}'

            completion_dict['choices'][0].message.content = res.group()
            return self._process_response(completion_dict,
                                          messages,
                                          reset_messages,
                                          response_only)

        except Exception as err:
            return False, f'OpenAI API error: {err}'

    def request_img_api(self, user_input: str,
                        url_lst: List[str],
                        response_only: bool = True,
                        detailed_img: bool = False,
                        **kwargs) -> Tuple[bool, Union[str, List[Dict[str, str]]]]:
        """
        Send a request to the OpenAI API with image content.

        Args:
            user_input (str): The user's input message.
            url_lst (List[str]): List of image source to be processed, including url and path.
            response_only (bool): Whether to return only the response content.
            detailed_img (bool): Whether to use high-detail image processing.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[bool, Union[str, List[Dict[str, str]]]]: Success flag and either the response content or updated messages.
        """
        detail = "high" if detailed_img else "low"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            img_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img_url)}", "detail": detail}}
                for img_url in url_lst
            ]
        except Exception as err:
            return False, f"Encode image error: {err}"

        img_content.append({"type": "text", "text": user_input})
        messages = [{"role": "user", "content": img_content}]
        messages, input_tokens = self._truncate_input(messages)

        parameters = {
            "model": self.model,
            "messages": messages,
            "max_tokens": min(self.max_tokens - input_tokens, self.max_output_tokens),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


        response, content, flag = self.handle_request(url=url, parameters=parameters, headers=headers)

        if flag:
            return self._process_response(response,
                                          messages,
                                          reset_messages=True,
                                          response_only = response_only,
                                          request_by_openai= False)
        else:
            if response_only:
                return flag, content
            else:
                messages.append({'role': 'assistant', 'content': content})
                return flag, messages

    @staticmethod
    def resize_image(image: Image.Image, low_res: bool = True) -> Image.Image:
        """
        Resize an image according to the low and high resolution requirements.

        :param image: PIL Image object to be resized.
        :param low_res: If True, resize for low resolution (512x512),
                        otherwise for high resolution (less than 768x1024).
        :return: Resized PIL Image object.
        """
        if low_res:
            max_size = (512, 512)
        else:
            max_size = (768, 2000)

        # Only resize if the image exceeds the required dimensions
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        return image

    def encode_image(self,image_path: str, low_res: bool = True) -> str:
        """
        Encodes an image file into a base64 string.

        This method handles both local files and remote URLs. For remote URLs, it downloads
        the image before encoding it. For local files, it reads the file directly and resizes if necessary.

        :param image_path: Path to the image file or URL of the image.
        :param low_res: If True, resize for low resolution (512x512), otherwise for high resolution.
        :return: Base64 encoded string representation of the image.
        """
        if image_path.startswith("http"):
            # Download the image from the URL
            response = requests.get(image_path)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
            else:
                raise ValueError(f"Failed to download image from URL: {image_path}")
        else:
            image = Image.open(image_path)

        # Resize the image according to the resolution requirements
        image = self.resize_image(image, low_res=low_res)

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def filter_final_response(self, resps: List[str], raw_marker: str, final_marker: str) -> str:
        """
        Filters the final response from a list of responses based on markers.

        This method searches for specific markers within each response to identify the final
        and raw summaries. It then extracts the text following these markers and returns the
        cleaned up text. If no markers are found, it returns the entire response cleaned up.

        :param resps: A list of responses or a single string response.
        :param raw_marker: Marker for identifying the start of a raw summary.
        :param final_marker: Marker for identifying the start of a final summary.
        :return: The final response as a string or a list of strings.
        """
        # Ensure input is a list; convert to list if it's a string
        is_list = isinstance(resps, list)
        if isinstance(resps, str):
            resps = [resps]
        elif not is_list:
            raise ValueError(f'resps should be list or str, but got {type(resps)}')

        # Compile regex patterns for both markers
        craft_summary_pattern = re.compile(rf"\[{raw_marker}\]|`{raw_marker}`|{raw_marker}", re.IGNORECASE)
        final_summary_pattern = re.compile(rf"\[{final_marker}\]|`{final_marker}`|{final_marker}", re.IGNORECASE)

        # Process each response
        filtered_responses = []
        for resp in resps:
            final_summary_match = final_summary_pattern.search(resp)
            craft_summary_match = craft_summary_pattern.search(resp)

            # Extract the relevant part of the response after the marker
            if final_summary_match:
                filtered_responses.append(self.clean_resp(resp[final_summary_match.end():].strip()))
            elif craft_summary_match:
                filtered_responses.append(self.clean_resp(resp[craft_summary_match.end():].strip()))
            else:
                filtered_responses.append(self.clean_resp(resp.strip()))

        # Return the result in the appropriate format
        return filtered_responses if is_list else filtered_responses[0]

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


    def multi_request(self,
                      article_texts: Union[str, List[str]] = None,
                      system_messages: Union[str, List[str]] = None,
                      num_processes: int = 2,
                      response_only: bool = True,
                      reset_messages: bool = True):
        with Pool(processes=num_processes) as pool:
            chat_func = partial(self.request_api,
                                response_only=response_only,
                                reset_messages=reset_messages)
            article_texts = [article_texts] if isinstance(article_texts, str) else article_texts
            flag = False
            if system_messages and isinstance(system_messages[0], List):
                if not len(article_texts) == len(system_messages):
                    raise ValueError(f"Length of article_texts {len(article_texts)} and system_messages {len(system_messages)} should be the same")
                flag = True
            article_texts = tqdm(article_texts, position=0, leave=True)
            article_texts.set_description(
                f"Processing {len(article_texts)} articles with {self.model} model")
            try:
                results = [
                    pool.apply_async(chat_func,
                                     kwds={'system_messages':system_messages[i] if flag else system_messages,
                                           'user_input': article_text,
                                           })
                    for i, article_text in enumerate(article_texts)
                ]
            except Exception as e:
                error_msg = f"Multi chat processing failed with error {e}"
                logger.error(error_msg)
                return False, error_msg
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        success = all([r[0] for r in results])
        if success:
            results = [r[1] for r in results]
            return success, results
        else:
            for i, result in enumerate(results):
                if not result[0]:
                    logger.error(f"Failed to summarize section {i} with error {result[1]}")
                    return success, result[1]

    def clean_math_text(self,text):
        """
        Clean math text
        Args:
            text:  text with math formulas in latex

        Returns:
            text with formatted math formulas in markdown
        """
        markdown_text = self.latex_to_markdown(text)
        formatted_text = self.format_markdown_formulas(markdown_text)
        return formatted_text

    @staticmethod
    def latex_to_markdown(text: str):
        """
        Convert LaTeX math formulas to markdown math formulas.
        """
        math_pattern = re.compile(r"(\\\(.*?\\\))|(\\\[.*?\\\])", re.DOTALL)

        def replace_math(match):
            # match.group(0)
            # match.group(1) -> \(...\)
            # match.group(2) -> \[...\]
            if match.group(1):  # if it is \(...\)
                return '$' + match.group(1)[2:-2] + '$'

            elif match.group(2):  # if it is \[...\]
                return '$$' + match.group(2)[2:-2] + '$$'

        return re.sub(math_pattern, replace_math, text)

    @staticmethod
    def format_markdown_formulas(markdown_text):
        def replace_formula(match):
            block_formula, inline_formula = match.groups()
            if block_formula:
                # for block formulas, add newlines and remove existing newlines
                res = block_formula.replace('\n', '')
                return f"\n$${res}$$\n"
            elif inline_formula:
                # for inline formulas, add spaces
                res = inline_formula.replace('\n', '')
                return f" ${res}$ "

        ## find all math formulas
        pattern = re.compile(r'\$\$([^$]*?)\$\$|\$([^$]*?)\$', re.DOTALL)

        ## replace the math formulas
        formatted_text = pattern.sub(replace_formula, markdown_text)

        return formatted_text

    @staticmethod
    def format_headers(text: str):
        # Use regex to match markdown headings and ensure they are preceded by a newline
        pattern = re.compile(r'(?<![\n#])\n?(#+\s+.*?)(?=\n|$)')
        modified_text = pattern.sub(r'\n\n\1', text)
        # Remove any title patterns that may have been left over
        title_pattern = re.compile(r'(?<![\n#])\n?#\s+.*?\n+')
        match = title_pattern.search(modified_text)
        if match:
            modified_text = modified_text[match.end():]
        return modified_text



