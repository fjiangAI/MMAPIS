import logging

import numpy as np
from MMAPIS.backend.tools.utils import num_tokens_from_messages,img_to_html,strip_title
from MMAPIS.backend.tools.chatgpt import GPTHelper
from MMAPIS.backend.downstream.multimodal_qa.user_intent import UserIntent
from MMAPIS.backend.downstream.multimodal_qa.img_qa_generation import ImgQAGenerator
from MMAPIS.backend.config.config import  APPLICATION_PROMPTS,ALIGNMENT_CONFIG, OPENAI_CONFIG,GENERAL_CONFIG
from MMAPIS.backend.data_structure import Article,Section
from typing import Tuple, Union, Dict, List
import requests
import spacy
import re
import os
from typing import Iterable
import reprlib

class MultiModalQAGenerator:
    def __init__(self, api_key, base_url, model_config: dict = None, proxy: dict = None,prompt_ratio = 0.8, **kwargs):
        self.user_intent = UserIntent(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.img_qa_generator = ImgQAGenerator(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.text_qa_generator = GPTHelper(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.general_qa_generator = GPTHelper(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.session_message = []
        self.nlp = spacy.load("en_core_web_lg")

    def QA_Generation(self,
                      user_intent: Dict,
                      document_level_summary: str,
                      raw_md_text: str = None,
                      prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                      min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level'],
                      max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level'],
                      ignore_titles=OPENAI_CONFIG['ignore_title'],
                      response_only: bool = True,
                      session_message: List[str] = None,
                      detailed_img: bool = False,
                      img_width: int = 200,
                      margin: int = 10
                      ) -> Tuple[bool, str]:
        """
        Generate question-answer pairs based on user intent and document content.

        Args:
            user_intent (Dict): User's intent for the query.
            document_level_summary (str): Summary of the document.
            raw_md_text (str, optional): Full original markdown text.
            prompts (Dict): Prompts for QA generation.
            min_grained_level (int): Minimum granularity level for text processing.
            max_grained_level (int): Maximum granularity level for text processing.
            ignore_titles (List): Titles to ignore in processing.
            response_only (bool): Whether to return only the response.
            session_message (List[str], optional): Previous session messages.
            detailed_img (bool): Whether to provide detailed image analysis.
            img_width (int): Width of images in output.
            margin (int): Margin for image display.

        Returns:
            Tuple[bool, str]: Success flag and generated answer or error message.
        """
        if not isinstance(user_intent, Dict):
            return False, f"User intent parse error, expected dict, but got {type(user_intent)}"

        document_article = self._prepare_document_article(document_level_summary,
                                                          min_grained_level,
                                                          max_grained_level,
                                                          ignore_titles)
        raw_article = self._prepare_document_article(raw_md_text,
                                                     min_grained_level,
                                                     max_grained_level,
                                                     ignore_titles) if raw_md_text else None
        session_message = self._prepare_session_message(session_message, prompts, document_level_summary)

        user_type = user_intent.get('type')
        index = user_intent.get('index')
        user_input = user_intent.get('user_input', '')

        if user_type == 'text':
            return self._handle_text_query(index,
                                           user_input,
                                           document_article,
                                           raw_article,
                                           prompts,
                                           session_message,
                                           response_only)
        elif user_type == 'img':
            return self._handle_img_query(index,
                                          user_input,
                                          document_level_summary=document_level_summary,
                                          document_article=document_article,
                                          detailed_img = detailed_img,
                                          img_width = img_width,
                                          margin = margin,
                                          response_only=response_only)
        else:
            return self._handle_general_query(user_input, session_message, response_only)



    def chat(self,
             user_input: str,
             document_level_summary: str,
             session_message: List[str] = None,
             raw_md_text: str = None,
             prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
             min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level'],
             max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level'],
             ignore_titles = OPENAI_CONFIG['ignore_title'],
             response_only: bool = True,
             detailed_img:bool = False,
             img_width:int = 400,
             margin:int = 10
             ):
        if session_message:
            self.session_message = session_message
        flag, user_intent = self.user_intent.get_intend(user_input,prompts=prompts,response_only=response_only)
        logging.debug(f"User intent: {user_intent}")
        flag, answer = self.QA_Generation(
            user_intent=user_intent,
            document_level_summary=document_level_summary,
            raw_md_text=raw_md_text,
            prompts=prompts,
            min_grained_level=min_grained_level,
            max_grained_level=max_grained_level,
            ignore_titles=ignore_titles,
            response_only=response_only,
            session_message=self.session_message,
            detailed_img=detailed_img,
            img_width=img_width,
            margin=margin
        )
        logging.debug(f"Answer: \n{answer}")
        answer = self.general_qa_generator.clean_math_text(answer)
        self.session_message.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
        logging.debug(f"Session message: {self.session_message}")
        return flag, answer



    def reset_session(self):
        self.session_message = []

    def multi_round_chat(self,
                         document_level_summary: str,
                         raw_md_text: str = None,
                         max_round: int = 10,
                         prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                         min_grained_level: int = ALIGNMENT_CONFIG['min_grained_level'],
                         max_grained_level: int = ALIGNMENT_CONFIG['max_grained_level'],
                         ignore_titles = OPENAI_CONFIG['ignore_title'],
                         response_only: bool = True,
                         ):
        round_idx = 0
        while round_idx < max_round:
            user_input = input("User:\n")
            if user_input == "exit":
                break
            self.chat(
                user_input=user_input,
                document_level_summary=document_level_summary,
                raw_md_text=raw_md_text,
                prompts=prompts,
                min_grained_level=min_grained_level,
                max_grained_level=max_grained_level,
                ignore_titles=ignore_titles,
                response_only=response_only,
            )
            logging.debug("-"*25+"Round "+str(round_idx+1) +"end"+"-"*25+"\n")
            round_idx += 1


    def clear_session(self,session_message:List[str] = None):
        token_threshold = self.general_qa_generator.max_tokens*self.general_qa_generator.prompt_ratio
        if num_tokens_from_messages(session_message) < token_threshold:
            pass
        else:
            while num_tokens_from_messages(session_message) > token_threshold:
                for i, message in enumerate(session_message):
                    if message['role'] == 'system':
                        continue
                    else:
                        session_message.pop(i)



    def find_target_section(self,
                            title: str,
                            article: Article) -> Section:
        """Find a target section in the article based on the title."""
        target_title = strip_title(title.lower())
        sim_ls = []

        # Iterate through each section in the article to compare titles
        for section in article.sections:
            temp_sim_ls = []
            for i, section_name in enumerate(section.section_titles):
                section_name = strip_title(section_name.lower())
                temp_sim_ls.append(self.nlp(section_name).similarity(self.nlp(target_title)))
                if target_title in section_name:
                    return section if i == 0 else section.children_list[i - 1]
            sim_ls.append(temp_sim_ls)
        # If no sections were found or similarity list is empty, return None
        if not sim_ls:
            return None
        max_index = self.find_max_index(sim_ls)
        if max_index[1] == 0:
            return article.sections[max_index[0]]
        else:
            return article.sections[max_index[0]].children_list[max_index[1] - 1]

    @staticmethod
    def find_max_index( arr:List[List[float]]) -> Tuple[int,int]:
        max_num = -1
        max_index = (-1,-1)
        for i, ls in enumerate(arr):
            for j, num in enumerate(ls):
                # i: section index, j: title index
                if num > max_num:
                    max_num = num
                    max_index = (i,j)
        return max_index




    @staticmethod
    def url_to_path(url_lst:List[str])->List[str]:
        for i in range(len(url_lst)):
            if not url_lst[i].startswith("http"):
                url_lst[i] = os.path.abspath(url_lst[i])
        return url_lst


    @staticmethod
    def find_img_url(text: str)->List[str]:
        # Find all image URLs in the text
        div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
        img_pattern = re.compile(r"s?rc=\"(.*?)\"", re.DOTALL)
        divs = div_pattern.findall(text)
        img_urls = []
        for div in divs:
            img_urls.extend(img_pattern.findall(div))
        return img_urls

    def _prepare_document_article(self, document_level_summary: str, min_grained_level: int, max_grained_level: int, ignore_titles: List[str]) -> Article:
        """Prepare the document article for processing."""
        return Article(document_level_summary, min_grained_level=min_grained_level, max_grained_level=max_grained_level, ignore_title=ignore_titles)


    def _prepare_session_message(self, session_message: List[str], prompts: Dict, document_level_summary: str) -> List[str]:
        """Prepare the session message with system prompts if necessary."""
        system_prompt = [
            {"role": "system", "content": prompts.get('text_system', '')},
            {"role": "system", "content": f"Context: {self._clean_img_url(document_level_summary)}"}
        ]

        if session_message:
            if session_message[0]['role'] != 'system':
                session_message = system_prompt + session_message
            self.clear_session(session_message)
        else:
            session_message = system_prompt

        return session_message

    def _handle_text_query(self, index,
                           user_input: str,
                           document_article: Article,
                           article: Article,
                           prompts: Dict,
                           session_message: List[str], response_only: bool) -> Tuple[bool, str]:
        """Handle text-based queries."""
        target_title = self._get_target_titles(index, document_article.sections)
        if not target_title:
            return self._handle_general_query(user_input, session_message, response_only)

        context = self._get_context_from_titles(target_title, article or document_article)
        if not context:
            return False, f"Target section \"{target_title}\" not found in the document."
        user_input = prompts.get('text_input', '').replace('{context}', context).replace('{user query}', user_input)

        flag, answer = self.text_qa_generator.request_text_api(
            user_input=user_input,
            response_only=response_only,
            session_messages=session_message
        )
        return flag, answer

    def _handle_img_query(self, index,
                          user_input: str,
                          document_level_summary: str,
                          document_article: Article,
                          detailed_img: bool,
                          img_width: int,
                          margin: int,
                          response_only: bool) -> Tuple[bool, str]:
        """Handle image-based queries."""
        target_img, target_url = self._get_target_images(index, document_level_summary=document_level_summary, document_article=document_article)
        if not target_img:
            return False, target_url

        qa_system = "## Role:\nYou are specialized in interpreting and providing detailed explanations based on academic paper images and their context. Your task is to answer user questions with high accuracy and clarity, ensuring that your responses are formatted in markdown."
        qa_input = f"{qa_system}\n\n## user query:\n{user_input}"

        flag, answer = self.img_qa_generator.request_img_api(
            user_input=qa_input,
            url_lst=target_img,
            detailed_img=detailed_img,
            response_only=True
        )

        if flag:
            answer = img_to_html(img_path_ls=target_url, img_height=img_width, margin=margin) + "\n\n- **Answer**:\n" + answer

        return flag, answer

    def _handle_general_query(self, user_input: str, session_message: List[str], response_only: bool) -> Tuple[bool, str]:
        """Handle general queries that don't fall into text or image categories."""
        flag, answer = self.general_qa_generator.request_text_api(
            user_input=user_input,
            session_messages=session_message,
            response_only=response_only
        )
        return flag, answer

    def _get_target_titles(self, index, sections: List) -> List[str]:
        """Get target titles based on the provided index."""
        if isinstance(index, int):
            return [sections[index].title]
        elif isinstance(index, List):
            return [sections[i].title for i in index]
        return []

    def _get_context_from_titles(self, target_titles: List[str], article: Article) -> str:
        """Extract context from the article based on target titles."""
        contexts = []
        for title in target_titles:
            target_section = self.find_target_section(title, article=article)
            if target_section:
                contexts.append(self._clean_img_url(target_section.title_text))
        return "\n".join(contexts)

    def _get_target_images(self, index,
                           document_level_summary: str,
                           document_article: Article
                           ):
        """Get target images and their URLs based on the provided index."""

        if isinstance(index, int):
            url_ls = self.find_img_url(document_level_summary)
            img_paths = self.url_to_path(url_ls)
            return [img_paths[index]], [url_ls[index]]
        elif isinstance(index, Iterable):
            # If the index is a list of indices, meaning that the user asked for img in a specific section
            if isinstance(index[0], int):
                section = document_article.sections[index[0]]
                section_url_ls = self.find_img_url(section.title_text)
                section_img_paths = self.url_to_path(section_url_ls)
                try:
                    return [section_img_paths[index[1]]], [section_url_ls[index[1]]]
                except IndexError:
                    return None, f"Section \"{section.title}\" img query index out of range, expect index < {len(section_img_paths)}, but got {index[1]}"

            # If the index is a list of lists of indices, meaning that the user asked for img in multiple sections
            elif isinstance(index[0], Iterable):
                target_img, target_url = [], []
                for i in index:
                    section = document_article.sections[i[0]]
                    section_url_ls = self.find_img_url(section.title_text)
                    section_img_paths = self.url_to_path(section_url_ls)
                    try:
                        target_img.append(section_img_paths[i[1]])
                        target_url.append(section_url_ls[i[1]])
                    except IndexError:
                        return None, f"Section \"{section.title}\" img query index out of range, expect index < {len(section_img_paths)}, but got {i[1]}"
                return target_img, target_url
        return None, f"Your query is not specific enough, detected index type is {type(index)}, please ask with a specific index."

    @staticmethod
    def _clean_img_url(text: str) -> str:
        div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
        return re.sub(div_pattern, '', text)

    def __repr__(self):
        """
        print the basic info of OpenAI_Summarizer
        :return: str
        """
        return_str = []
        for key, value in self.user_intent.__dict__.items():
            if value:
                return_str.append(f"{key}: {reprlib.repr(value)}")
        return f"MultiModalQAGenerator({', '.join(return_str)})"