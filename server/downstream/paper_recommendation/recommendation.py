from MMAPIS.tools import  GPT_Helper,init_logging,num_tokens_from_messages
from MMAPIS.config.config import CONFIG,APPLICATION_PROMPTS,LOGGER_MODES
from MMAPIS.server.summarization import Article
from typing import Union,List
import logging

class Paper_Recommender(GPT_Helper):
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = {},
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        super().__init__(api_key=api_key,
                         base_url=base_url,
                         model_config=model_config,
                         proxy=proxy,
                         prompt_ratio=prompt_ratio,
                         **kwargs)

    def request_api(self,
                    user_input: str,
                    system_messages: Union[str, List[str]] = None,
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
            else:
                self.messages.append({
                    'role': 'assistant',
                    'content': content
                })
        if response_only:
            return flag, content
        else:
            return flag, self.messages

    def recommendation_generation(self,
                                  document_level_summary: str,
                                  article: Union[str, List[str], Article],
                                  score_prompts: dict,
                                  reset_messages: bool = True,
                                  response_only: bool = True,
                                  **kwargs):

        if isinstance(article, List):
            article_segment = "\n".join([article[0],article[-1]])

        else:
            if isinstance(article, Article):
                article_segment = [article.sections[0].title_text, article.sections[-1].title_text]
            else:
                article = Article(article)
                article_segment = [article.sections[0].title_text, article.sections[-1].title_text]
            article_segment = "\n".join(article_segment)
        score_system = [score_prompts.get('score_system', ''), score_prompts.get('score', '')]
        score_prompt = score_prompts.get('score_input', '').replace('{article}', document_level_summary, 1).replace(
            '{paper excerpt}', article_segment, 1)

        self.init_messages("system",score_system)
        flag, result = self.request_api(score_prompt,
                                        reset_messages=reset_messages,
                                        response_only=response_only,
                                        **kwargs)
        if flag:
            result = eval(result)["output"]
        return flag, result

    def __repr__(self):
        return f"Paper_Recommender(api_key={self.api_key},base_url={self.base_url},proxy={self.proxy}),model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"









