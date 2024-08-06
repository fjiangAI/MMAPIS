from MMAPIS.tools import  GPT_Helper,init_logging,num_tokens_from_messages
from MMAPIS.config.config import CONFIG,APPLICATION_PROMPTS,LOGGER_MODES,OPENAI_CONFIG
from MMAPIS.server.summarization import Article
from typing import Union,List
import logging
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
import re


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
        if self.model.startswith("gpt-4") and self.model != "gpt-4-turbo-2024-04-09":
            self.model = "gpt-4o-mini"
            logging.warning(f"model {self.model} is not supported, will use gpt-4o-mini instead")
        else:
            replaced_model = OPENAI_CONFIG["recommendation_model"]
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
    def request_api(self,
                    user_input:str,
                    system_messages:Union[str,List[str]] = None,
                    response_only:bool = True,
                    reset_messages:bool = True,
                    **kwargs
                    ):
        return self.request_via_openai(
            user_input=user_input,
            system_messages=system_messages,
            response_only=response_only,
            reset_messages=reset_messages,
        )

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
        score_prompt = score_prompts.get('score_input', '').replace('{article}', document_level_summary, 1)
        score_system_writing = [score_prompts.get('score_system_writing', ''), score_prompts.get('score_writing', '')]
        score_input_writing = score_prompts.get('score_input_writing', '').replace('{paper excerpt}', article_segment, 1)
        flag_4,result_4 = self.request_api(
            response_only=response_only,
            reset_messages=reset_messages,
            user_input=score_prompt,
            system_messages=score_system
        )
        flag_1,result_1 = self.request_api(
            response_only=response_only,
            reset_messages=reset_messages,
            user_input=score_input_writing,
            system_messages=score_system_writing
        )
        if flag_4 and flag_1:
            result_4 = eval(result_4)["output"]
            result_1 = eval(result_1)["output"]
            result_4.extend(result_1)
            result = result_4
            score_ls = [r["score"] for r in result]
            result.append({"title": "Overall Score", "score": sum(score_ls) / len(score_ls)})
            return True,result
        else:
            if not flag_4:
                return flag_4,result_4
            else:
                return flag_1,result_1

        # flag, result = self.multi_request(
        #     response_only=response_only,
        #     reset_messages=reset_messages,
        #     article_texts=[score_prompt,score_input_writing],
        #     system_messages=[score_system,score_system_writing]
        # )
        # print("flag:",flag,"result:",result)
        # return flag, result

    def multi_request(self,response_only: bool = True,reset_messages: bool = True,article_texts: Union[str, List[str]] = None,system_messages: Union[str, List[str]] = None):
        with Pool(processes=2) as pool:
            chat_func = partial(self.request_api,
                                response_only=response_only,
                                reset_messages=reset_messages)
            article_texts = [article_texts] if isinstance(article_texts, str) else article_texts
            if system_messages and isinstance(system_messages[0], List):
                if not len(article_texts) == len(system_messages):
                    raise ValueError(f"Length of article_texts {len(article_texts)} and system_messages {len(system_messages)} should be the same")
            article_texts = tqdm(article_texts, position=0, leave=True)
            article_texts.set_description(
                f"Processing {len(article_texts)} articles with {self.model} model")
            try:
                results = [
                    pool.apply_async(chat_func,
                                     kwds={'system_messages': system_messages[i],
                                           'user_input': article_text,
                                           })
                    for i, article_text in enumerate(article_texts)
                ]
            except Exception as e:
                error_msg = f"Multi chat processing failed with error {e}"
                logging.error(error_msg)
                return False, error_msg
            pool.close()
            pool.join()
            results = [p.get() for p in results]
        success = all([r[0] for r in results])
        if success:
            results = [eval(r[1])["output"] for r in results]
            results = results[0].extend(results[1])
            score_ls = [result["score"] for result in results]
            results.append({"title":"Overall Score","score":sum(score_ls)/len(score_ls)})
            return success, results
        else:
            for i, result in enumerate(results):
                if not result[0]:
                    logging.error(f"Failed to summarize section {i} with error {result[1]}")
                    return success, result[1]

    def __repr__(self):
        return f"Paper_Recommender(api_key={self.api_key},base_url={self.base_url},proxy={self.proxy}),model:{self.model}, temperature:{self.temperature}, max_tokens:{self.max_tokens}, top_p:{self.top_p}, frequency_penalty:{self.frequency_penalty}, presence_penalty:{self.presence_penalty})"








