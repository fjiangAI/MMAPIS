import numpy as np
from MMAPIS.tools import num_tokens_from_messages,img_to_html,strip_title
from MMAPIS.server.downstream.multimodal_qa.user_intent import UserIntent
from MMAPIS.server.downstream.multimodal_qa.img_qa_generation import Img_QA_Generator
from MMAPIS.config.config import  APPLICATION_PROMPTS,ALIGNMENT_CONFIG, OPENAI_CONFIG,GENERAL_CONFIG
from MMAPIS.tools import init_logging, GPT_Helper
from MMAPIS.server.summarization import Article
from typing import Tuple, Union, Dict, List
import requests
import spacy
import re
import os

class MultiModal_QA_Generator():
    def __init__(self, api_key, base_url, model_config: dict = {}, proxy: dict = None,prompt_ratio = 0.8, **kwargs):
        self.user_intent = UserIntent(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.img_qa_generator = Img_QA_Generator(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.text_qa_generator = GPT_Helper(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.general_qa_generator = GPT_Helper(api_key=api_key, base_url=base_url, model_config=model_config, proxy=proxy, prompt_ratio=prompt_ratio, **kwargs)
        self.session_message = []

    def QA_Generation(self,
                      user_intent:Dict,
                      document_level_summary: str,
                      article: str = None,
                      prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                      init_grid: int = ALIGNMENT_CONFIG['init_grid'],
                      max_grid: int = ALIGNMENT_CONFIG['max_grid'],
                      ignore_titles = OPENAI_CONFIG['ignore_title'],
                      response_only: bool = True,
                      session_message: List[str] = None,
                      detailed_img:bool = False,
                      img_width:int = 400,
                      margin:int = 10
                      ):
        if not isinstance(user_intent, Dict):
            return False, f"user_intent parse error, expect dict, but got {type(user_intent)}"
        div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
        system_prompt =  [{"role": "system", "content": prompts.get('text_system', None)},{"role": "system", "content": f"`Context: {re.sub(div_pattern,'',document_level_summary)}`"}]
        if session_message:
            if session_message[0]['role'] != 'system':
                session_message = system_prompt + session_message
            self.clear_session(session_message)
        else:
            session_message = system_prompt
        user_type = user_intent.get('type', None)
        index = user_intent.get('index', None)
        user_input = user_intent.get('user_input', None)
        ## TODO: add a default user input
        # if user_type is None:
        #     div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
        #     document_level_summary = div_pattern.sub("", document_level_summary)
        document_article = Article(document_level_summary,grid=init_grid,max_grid=max_grid,ignore_title=ignore_titles)
        sections = document_article.sections
        print("imgs: ", self.find_img_url(document_level_summary))
        print("document_article: ", document_level_summary)
        if user_type == 'text':
            if isinstance(index, int):
                target_title = sections[index].title
            elif isinstance(index, List):
                target_title = []
                for i in index:
                    target_title.append(sections[i].title)
            elif index is None:
                flag, answer = self.general_qa_generator.summarize_text(
                    text=user_input,
                    session_messages=session_message,
                    response_only=response_only
                )
                if flag:
                    answer = "- Context: No specific chapter detected, answer with general context:\n"+answer
                return flag, answer
            else:
                flag, answer = False, f"user intent parse error in text query, expect int(for specific chapter) or list(for multiple chapters), but got {type(index)}"
                return flag, answer
            if not isinstance(target_title, List):
                target_title = [target_title]
            context = []
            if article is None:
                article = document_level_summary
            else:
                article = Article(article,grid=init_grid,max_grid=max_grid,ignore_title=ignore_titles)
            for title in target_title:
                target_section = self.find_target_section(title,article=article)
                if target_section is None:
                    return False, f"can't find target section: `{title}`, due to article parse error"
                else:
                    context.append(target_section.title_text)
            context = "\n".join(context)
            system_prompt = prompts.get('text_system', None)
            user_input = prompts.get('text_input', None).replace('{context}', context).replace('{user query}', user_input)
            flag, answer = self.text_qa_generator.summarize_text(
                text=user_input,
                system_messages=system_prompt,
                response_only=response_only
            )
            if flag:
                answer = f"- Context:\n Chapter: {target_title}\n- Answer:\n{answer}"
        elif user_type == 'img':
            if isinstance(index, int):
                url_ls = self.find_img_url(document_level_summary)
                img_paths = self.url_to_path(url_ls)
                try:
                    target_url = url_ls[index]
                    target_img = img_paths[index]
                except:
                    flag, answer = False, f"img query index out of range, expect image index < {len(img_paths)}, but got {index}"
                    return flag, answer
            elif isinstance(index, List):
                if isinstance(index[0], int):
                    target_section = sections[index[0]]
                    print("target_section: ", target_section.title)
                    url_ls = self.find_img_url(target_section.title_text)
                    img_paths = self.url_to_path(url_ls)
                    try:
                        target_url = url_ls[index[1]]
                        target_img = img_paths[index[1]]
                    except:
                        flag, answer = False, f"img query index out of range, expect index < {len(img_paths)}, but got {index[1]}"
                        return flag, answer
                elif isinstance(index[0], List):
                    target_img = []
                    target_url = []
                    for i in index:
                        target_section = sections[i[0]]
                        print("target_section: ", target_section.title)
                        url_ls = self.find_img_url(target_section.title_text)
                        img_paths = self.url_to_path(url_ls)
                        try:
                            target_url.append(url_ls[i[1]])
                            target_img.append(img_paths[i[1]])
                        except:
                            flag, answer = False, f"img query index out of range, expect index < {len(img_paths)}, but got {i[1]}"
                            return flag, answer
                else:
                    ##  TODO: add possible default target img
                    pass

            elif index is None:
                flag, answer = self.general_qa_generator.summarize_text(
                    text=user_input,
                    session_messages=session_message,
                    response_only=response_only
                )
                if flag:
                    answer = "- Context: No specific image index detected, answer with general context:\n"+answer
                return flag, answer
            else:
                flag, answer = False, f"user intent parse error in img query, expect index be int(for specific figure) or list(for figure in specific chapter), but got {type(index)}"
                return flag, answer
            if not isinstance(target_img, List):
                target_img = [target_img]
            flag, answer = self.img_qa_generator.request_img_api(user_input=user_input, url_lst=target_img,detailed_img=detailed_img,response_only=True)
            if isinstance(target_url,str):
                target_url = [target_url]
            answer = "- Context:\n" + img_to_html(img_path_ls=target_url,img_height=img_width,margin=margin)+"\n- Answer:\n"+answer
        else:
            flag, answer = self.general_qa_generator.summarize_text(
                text=user_input,
                session_messages=session_message,
                response_only=response_only
            )
            if flag:
                answer = "- Context: Detect as general query, answer with general context:\n- Answer:\n"+answer
            return flag, answer
        return flag, answer


    def chat(self,
             user_input: str,
             document_level_summary: str,
             session_message: List[str] = None,
             article: str = None,
             prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
             init_grid: int = ALIGNMENT_CONFIG['init_grid'],
             max_grid: int = ALIGNMENT_CONFIG['max_grid'],
             ignore_titles = OPENAI_CONFIG['ignore_title'],
             response_only: bool = True,
             detailed_img:bool = False,
             img_width:int = 400,
             margin:int = 10
             ):
        if session_message:
            self.session_message = session_message
        flag, user_intent = self.user_intent.get_intend(user_input,prompts=prompts,response_only=response_only)
        print("user_intent: ", user_intent)
        flag, answer = self.QA_Generation(
            user_intent=user_intent,
            document_level_summary=document_level_summary,
            article=article,
            prompts=prompts,
            init_grid=init_grid,
            max_grid=max_grid,
            ignore_titles=ignore_titles,
            response_only=response_only,
            session_message=self.session_message,
            detailed_img=detailed_img,
            img_width=img_width,
            margin=margin
        )
        # if flag:
        self.session_message.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
        print("Assistant: ", answer)
        print("session_message: ", self.session_message)
        return flag, answer



    def reset_session(self):
        self.session_message = []

    def multi_round_chat(self,
                         document_level_summary: str,
                         article: str = None,
                         max_round: int = 10,
                         prompts: Dict = APPLICATION_PROMPTS["multimodal_qa"],
                         init_grid: int = ALIGNMENT_CONFIG['init_grid'],
                         max_grid: int = ALIGNMENT_CONFIG['max_grid'],
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
                article=article,
                prompts=prompts,
                init_grid=init_grid,
                max_grid=max_grid,
                ignore_titles=ignore_titles,
                response_only=response_only,
            )
            print("-"*25+"Round "+str(round_idx+1) +"end"+"-"*25+"\n")
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
                        del_msg = session_message.pop(i)
                        print("-"*25+"Delete Message"+"-"*25)
                        print(del_msg)
                        print("-"*25+"Delete Message"+"-"*25)
                        break




    def find_target_section(self,
                            target_title:str,
                            article:Union[str,Article],
                            init_grid:int = ALIGNMENT_CONFIG['init_grid'],
                            max_grid:int = ALIGNMENT_CONFIG['max_grid'],
                            ignore_titles = OPENAI_CONFIG['ignore_title']):
        print("target_title: ", target_title)
        target_title = strip_title(target_title.lower())
        if isinstance(article, str):
            article = Article(article,grid=init_grid,max_grid=max_grid,ignore_title=ignore_titles)
        nlp = spacy.load("en_core_web_lg")
        sim_ls = []
        for id, section in enumerate(article.sections):
            print("section: ", section)
            temp_sim_ls = []
            for i, section_name in enumerate(section.subtitles):
                temp_sim_ls.append(nlp(section_name).similarity(nlp(target_title)))
                if target_title in section_name.lower():
                    return section if i == 0 else section.subsubgroup[i-1]
            sim_ls.append(temp_sim_ls)
        if len(sim_ls) == 0:
            return None
        max_index = self.find_max_index(sim_ls)
        if max_index[1] == 0:
            return article.sections[max_index[0]]
        else:
            return article.sections[max_index[0]].subsubgroup[max_index[1]-1]

    @staticmethod
    def find_max_index( arr:List[List[float]]) -> Tuple[int,int]:
        max_num = -1
        max_index = (-1,-1)
        for i, ls in enumerate(arr):
            for j, num in enumerate(ls):
                if num > max_num:
                    max_num = num
                    max_index = (i,j)
        return max_index

    # @staticmethod
    # def find_img_path(text: str):
    #     div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
    #     img_pattern = re.compile(r"s?rc=\"(.*?)\"", re.DOTALL)
    #     divs = div_pattern.findall(text)
    #     img_paths = []
    #     img_urls = []
    #     for div in divs:
    #         img_paths.extend(img_pattern.findall(div))
    #     for i, img_path in enumerate(img_paths):
    #         if img_path.startswith('http'):
    #             img_urls.append(img_path)
    #             img_paths[i] = os.path.join(GENERAL_CONFIG["app_save_dir"],img_path.split("index/")[1])
    #     print("img_paths: ", img_paths)
    #     return img_paths, img_urls


    @staticmethod
    def url_to_path(url_lst:List[str])->List[str]:
        img_paths = []
        for i, url in enumerate(url_lst):
            img_paths.append(os.path.join(GENERAL_CONFIG["app_save_dir"],url.split("index/")[1]))
        print("img_paths: ", img_paths)
        return img_paths


    @staticmethod
    def find_img_url(text: str)->List[str]:
        div_pattern = re.compile(r"(<div.*?>.*?</div>\n+</div>)", re.DOTALL)
        img_pattern = re.compile(r"s?rc=\"(.*?)\"", re.DOTALL)
        divs = div_pattern.findall(text)
        img_urls = []
        for div in divs:
            img_urls.extend(img_pattern.findall(div))
        print("img_urls: ", img_urls)
        return img_urls


if __name__ == "__main__":
    answer_generation = MultiModal_QA_Generator(api_key=OPENAI_CONFIG["api_key"],
                                                base_url=OPENAI_CONFIG["base_url"],
                                                model_config= OPENAI_CONFIG["model_config"],
                                                proxy=None,
                                                prompt_ratio=0.8)
    raw_path = "./raw.md"
    document_level_summary_path = "./test.md"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(document_level_summary_path, "r", encoding="utf-8") as f:
        document_level_summary = f.read()

    answer_generation.multi_round_chat(
        document_level_summary=document_level_summary,
        article=raw,
        max_round=10,
        prompts=APPLICATION_PROMPTS["multimodal_qa"],
        init_grid=ALIGNMENT_CONFIG['init_grid'],
        max_grid=ALIGNMENT_CONFIG['max_grid'],
        ignore_titles=OPENAI_CONFIG['ignore_title'],
        response_only=True,
    )
