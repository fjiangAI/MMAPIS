from MMAPIS.backend.tools.chatgpt.chatgpt_helper import  GPTHelper
from MMAPIS.backend.config.config import CONFIG,APPLICATION_PROMPTS,LOGGER_MODES,OPENAI_CONFIG
from MMAPIS.backend.data_structure import Article
from typing import Union,List
import logging
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
import re
import concurrent.futures
from typing import Dict, Tuple
import reprlib



class PaperRecommender:
    def __init__(self,
                 api_key,
                 base_url,
                 model_config: dict = None,
                 proxy: dict = None,
                 prompt_ratio: float = 0.8,
                 **kwargs):
        self.recommendation_generator = GPTHelper(api_key=api_key,
                                         base_url=base_url,
                                         model_config=model_config,
                                         proxy=proxy,
                                         prompt_ratio=prompt_ratio,
                                         **kwargs)
        self.recommendation_generator.check_model(model_type="json")

    @staticmethod
    def _prepare_article_segment(article: Union[str, List[str], Article]) -> str:
        """
        Prepares a string containing the first and last section titles of the article.

        :param article: The article in string, list, or Article object format
        :return: A concatenated string of the first and last sections of the article
        """
        if isinstance(article, list):
            # If the article is a list of sections, select the first and last
            return "\n".join([article[0], article[-1]])
        else:
            if isinstance(article, Article):
                # If the article is an Article object, extract the section titles
                return "\n".join([article.sections[0].title_text, article.sections[-1].title_text])
            else:
                # If the article is a string, convert it into an Article object and then extract sections
                article = Article(article)
                return "\n".join([article.sections[0].title_text, article.sections[-1].title_text])

    def recommendation_generation(self,
                                  document_level_summary: str,
                                  raw_md_text: Union[str, List[str], Article],
                                  score_prompts: dict,
                                  reset_messages: bool = True,
                                  response_only: bool = True,
                                  **kwargs):

        # Prepare a segment of the article by selecting its first and last sections
        article_segment = self._prepare_article_segment(raw_md_text)

        # Prepare scoring prompts
        content_score_params = self._prepare_content_score_params(document_level_summary, score_prompts,
                                                                  reset_messages=reset_messages, response_only=response_only)
        writing_score_params = self._prepare_writing_score_params(article_segment, score_prompts,
                                                                    reset_messages=reset_messages, response_only=response_only)

        # Conduct concurrent API requests for different scoring metrics
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_content = executor.submit(self._request_api, **content_score_params)
            future_writing = executor.submit(self._request_api, **writing_score_params)

        # Retrieve and process the results
        content_flag, content_result = future_content.result()
        writing_flag, writing_result = future_writing.result()

        if content_flag and writing_flag:
            try:
                return self._process_successful_results(content_result, writing_result)
            except Exception as e:
                return False, f"Error processing JSON response: {str(e)}"
        else:
            return self._handle_failed_requests(content_flag, content_result, writing_flag, writing_result)

    def _process_successful_results(self, content_result: str, writing_result: str) -> Tuple[
        bool, List[Dict[str, Union[str, float]]]]:
        """Process and combine successful API results."""
        content_scores = eval(content_result)["output"]
        writing_scores = eval(writing_result)["output"]
        all_scores = content_scores + writing_scores

        score_values = [score["score"] for score in all_scores]
        overall_score = sum(score_values) / len(score_values)
        all_scores.append({"title": "Overall Score", "score": overall_score})

        return True, all_scores

    def _handle_failed_requests(self, content_flag: bool, content_result: str,
                                writing_flag: bool, writing_result: str) -> Tuple[bool, str]:
        """Handle scenarios where one or both API requests failed."""
        if not content_flag:
            return content_flag, content_result
        else:
            return writing_flag, writing_result

    def _request_api(self, **kwargs) -> Tuple[bool, str]:
        """Wrapper method for API requests to handle potential exceptions."""
        try:
            return self.recommendation_generator.request_json_api(**kwargs)
        except Exception as e:
            return False, f"API request failed: {str(e)}"


    def _prepare_content_score_params(self,
                                      document_level_summary: str,
                                      score_prompts: Dict[str, str],
                                      reset_messages: bool = True,
                                      response_only: bool = True) -> Dict:
        """Prepare parameters for content scoring API request."""
        return {
            'user_input': score_prompts.get('score_input', '').replace('{article}', document_level_summary, 1),
            'system_messages': [score_prompts.get('score_system', ''), score_prompts.get('score', '')],
            'reset_messages': reset_messages,
            'response_only': response_only
        }

    def _prepare_writing_score_params(self,
                                      article_segment: str,
                                      score_prompts: Dict[str, str],
                                      reset_messages: bool = True,
                                      response_only: bool = True
                                      ) -> Dict:
        """Prepare parameters for writing style scoring API request."""
        return {
            'user_input': score_prompts.get('score_input_writing', '').replace('{paper excerpt}', article_segment, 1),
            'system_messages': [score_prompts.get('score_system_writing', ''), score_prompts.get('score_writing', '')],
            'reset_messages': reset_messages,
            'response_only': response_only
        }


    def __repr__(self):
        return_str = []
        for key, value in self.recommendation_generator.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"PaperRecommender({', '.join(return_str)})"







