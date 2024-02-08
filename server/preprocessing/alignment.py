from MMAPIS.tools import PDFFigureExtractor,SectionIMGPaths
from MMAPIS.server.summarization import  Section_Summarizer,Article,sectional_split,subgroup,Summary_Integrator
from MMAPIS.tools.utils import strip_title
from typing import Union, List,Tuple
from MMAPIS.config.config import GENERAL_CONFIG, OPENAI_CONFIG, SECTION_PROMPTS,INTEGRATE_PROMPTS, ALIGNMENT_CONFIG
import os
import spacy
from pathlib import Path
import time
from fastapi import UploadFile

def section_level_alignment(
    article:Article,
    pdf_path:str,
    save_dir:str = None,
    img_width:int = 600,
    save:bool = True
)-> Tuple[List[SectionIMGPaths],Article]:
    """
      :param pdf_path: str
      :return: List[subgroup]
      """
    if not save_dir:
        save_dir = os.path.dirname(pdf_path)
    pdf_extractor = PDFFigureExtractor(pdf_path)
    img_paths = pdf_extractor.extract_save_figures(save_dir=save_dir)
    fail_match_img_paths = []
    for section_img_paths in img_paths:
        img_section_name = section_img_paths.section_name
        flag = False

        for i, section in enumerate(article.sections):
            section_names = section.subtitles
            for section_name in section_names:
                section_name = strip_title(section_name)
                if img_section_name.lower() in section_name.lower():
                    flag = True
                    section_img_paths.parent = strip_title(section.parent)
                    section.add_img_paths([path for path in section_img_paths.img_path])
                    break
            if flag:
                break
        if flag:
            continue
        else:
            fail_match_img_paths += [path for path in section_img_paths.img_path]
    article.raw_document = article.extra_info + '\n' + img_ls_to_md(fail_match_img_paths,img_width=img_width) + '\n' + '\n'.join(['#'*section.rank + ' '+ strip_title(section.title) + '\n'+ img_ls_to_md(section.img_paths,img_width=img_width) +'\n\n' +section.text for section in article.sections])
    article.section_interpretation = article.extra_info + '\n' + img_ls_to_md(fail_match_img_paths,img_width=img_width) + '\n' + '\n'.join(['#'*section.rank + ' '+ strip_title(section.title) + '\n'+ img_ls_to_md(section.img_paths,img_width=img_width) +'\n\n' +section.summary for section in article.sections])
    if save:
        raw_docuemnt_file_name = f'{article.file_name}_aligned_raw.md'
        section_interpretation_file_name = f'{article.file_name}_aligned_section.md'
        raw_docuemnt_file_path = os.path.join(save_dir,raw_docuemnt_file_name)
        section_interpretation_file_path = os.path.join(save_dir,section_interpretation_file_name)
        with open(raw_docuemnt_file_path,'w',encoding='utf-8') as f:
            f.write(article.raw_document.strip())
        with open(section_interpretation_file_path,'w',encoding='utf-8') as f:
            f.write(article.section_interpretation.strip())
    return img_paths,article



def document_level_alignment(
    article:Article,
    document_level_summary:Union[str,Article],
    img_paths:List[SectionIMGPaths],
    save_dir:str = None,
    threshold:float = 0.9,
    init_grid:int = 3,
    max_grid:int = 4,
    img_width:int = 600,
    save:bool = True
):
    if isinstance(document_level_summary,str):
        document_level_summary = Article(article = document_level_summary,file_name='article_level_summary',grid=init_grid,max_grid=max_grid)
    article.document_summary = '\n'.join([section.title_text for section in document_level_summary.sections])
    nlp = spacy.load('en_core_web_lg')
    fail_match_img_paths = []

    for section_img_paths in img_paths:
        img_section_name = section_img_paths.section_name
        img_parent_name = section_img_paths.parent
        similarities = []
        flag = False
        for i, section in enumerate(document_level_summary.sections[::-1]):
            section_name = strip_title(section.title)
            if img_parent_name:
                temp_similarities = max([nlp(section_name).similarity(nlp(img_section_name)),nlp(section_name).similarity(nlp(img_parent_name))])
            else:
                fail_match_img_paths += [path for path in section_img_paths.img_path]
                flag = True
                break
            similarities.append(temp_similarities)
            if img_section_name.lower() in section_name.lower() or img_parent_name.lower() == section_name.lower():
                flag = True
                section.add_img_paths([path for path in section_img_paths.img_path])
                break
        if similarities:
            max_similarity = max(similarities)
        if flag:
            continue
        elif max_similarity > threshold and section_img_paths.img_path:
            max_similarity_index = similarities.index(max_similarity)
            document_level_summary.sections[max_similarity_index].add_img_paths([path for path in
                                                                  section_img_paths.img_path])
        else:
            fail_match_img_paths += [path for path in section_img_paths.img_path]
    if document_level_summary.extra_info:
        extra_info = document_level_summary.extra_info
    else:
        extra_info = article.extra_info
    article.document_interpretation = extra_info + '\n' + img_ls_to_md(fail_match_img_paths,img_width=img_width) + '\n\n' + '\n'.join(
        ['#' * section.rank + ' ' + strip_title(section.title) + '\n' + img_ls_to_md(section.img_paths,img_width=img_width)
         + '\n\n' + section.text for section in document_level_summary.sections])
    if save:
        file_name = f'{article.file_name}_aligned_document.md'
        file_path = os.path.join(save_dir,file_name)
        with open(file_path,'w',encoding='utf-8') as f:
            f.write(article.document_interpretation.strip())
    return img_paths,article

def img_txt_alignment(
        text:str,
        pdf:Union[str,Path,UploadFile],
        save_dir:str = None,
        file_name:str = None,
        raw_md_text:str = None,
        init_grid:int = 3,
        max_grid:int = 4,
        img_width:int = 600,
        threshold:float = 0.9
):
    if isinstance(pdf,str):
        pdf = Path(pdf)
    elif isinstance(pdf,UploadFile):
        pdf_name = pdf.filename
        save_dir = Path(save_dir) / Path(pdf_name).stem
        pdf_path = save_dir / pdf_name
        with open(pdf_path,'wb') as f:
            content = pdf.read()
            f.write(content)
        pdf = Path(pdf_path)
    if not save_dir:
        save_dir = pdf.parent
    res_article = Article(article = text,file_name = pdf.stem,grid=init_grid,max_grid=max_grid)
    pdf_extractor = PDFFigureExtractor(pdf_path = pdf)
    img_paths = pdf_extractor.extract_save_figures(save_dir=save_dir)
    if raw_md_text:
        article = Article(article = raw_md_text,file_name = f'{pdf.stem}_raw_md',grid=init_grid,max_grid=max_grid)
        for section_img_paths in img_paths:
            img_section_name = section_img_paths.section_name
            if img_section_name.startswith("unknown_section_"):
                continue
            flag = False
            for i, section in enumerate(article.sections):
                section_names = section.subtitles
                for section_name in section_names:
                    section_name = strip_title(section_name)
                    if img_section_name.lower() in section_name.lower():
                        flag = True
                        section_img_paths.parent = strip_title(section.parent)
                        break
                if flag:
                    break
        nlp = spacy.load('en_core_web_lg')
        fail_match_img_paths = []
        for section_img_paths in img_paths:
            img_section_name = section_img_paths.section_name
            img_parent_name = section_img_paths.parent
            similarities = []
            flag = False
            for i, section in enumerate(res_article.sections[::-1]):
                section_name = strip_title(section.title)
                if img_parent_name:
                    temp_similarities = max([nlp(section_name).similarity(nlp(img_section_name)),
                                                  nlp(section_name).similarity(nlp(img_parent_name))])
                    similarities.append(temp_similarities)
                else:
                    fail_match_img_paths += [path for path in section_img_paths.img_path]
                    flag = True
                    break
                if img_section_name.lower() in section_name.lower() or img_parent_name.lower() in section_name.lower():
                    flag = True
                    section.add_img_paths([path for path in section_img_paths.img_path])
                    break
            if similarities:
                max_similarity = max(similarities)
            if flag:
                continue
            elif max_similarity > threshold and section_img_paths.img_path:
                max_similarity_index = similarities.index(max_similarity)
                res_article.sections[max_similarity_index].add_img_paths([path for path in
                                                                                     section_img_paths.img_path])
            else:
                fail_match_img_paths += [path for path in section_img_paths.img_path]
        extra_info = article.extra_info


    else:
        nlp = spacy.load('en_core_web_lg')
        fail_match_img_paths = []
        for section_img_paths in img_paths:
            img_section_name = section_img_paths.section_name
            if img_section_name.startswith("unknown_section_"):
                fail_match_img_paths += [path for path in section_img_paths.img_path]
                continue
            similarities = []
            flag = False
            for i, section in enumerate(res_article.sections):
                section_name = strip_title(section.title)
                if img_section_name.lower() in section_name.lower():
                    flag = True
                    section.add_img_paths([path for path in section_img_paths.img_path])
                    break
                else:
                    temp_similarities = nlp(section_name).similarity(nlp(img_section_name))
                    similarities.append(temp_similarities)
            if similarities:
                max_similarity = max(similarities)
            if flag:
                continue
            elif max_similarity > threshold and section_img_paths.img_path:
                max_similarity_index = similarities.index(max_similarity)
                res_article.sections[max_similarity_index].add_img_paths([path for path in
                                                                          section_img_paths.img_path])
            else:
                fail_match_img_paths += [path for path in section_img_paths.img_path]
        extra_info = res_article.extra_info


    res = extra_info + '\n' + img_ls_to_md(fail_match_img_paths,img_width=img_width) + '\n' + '\n'.join(
        ['#' * section.rank + ' ' + strip_title(section.title) + '\n' + img_ls_to_md(section.img_paths,img_width=img_width)
         + '\n' + section.text for section in res_article.sections])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    if file_name:
        file_path = Path(save_dir) / f'{file_name}.md'
        with open(file_path,'w') as f:
            f.write(res)
    else:
        file_name = f'{pdf.stem}.md'
        file_path = Path(save_dir) / file_name
        with open(file_path,'w') as f:
            f.write(res)
    return file_path







def img_ls_to_md(img_ls:Union[List[str],str],
                 img_width:int = 600)->str:
    if isinstance(img_ls,str):
        img_ls = [img_ls]
    if not img_ls:
        return ""
    if len(img_ls) == 1:
        prefix = '<div style="display: flex; justify-content: center; overflow-x: auto;align-items: flex-start; padding: 5px 5px;">'
    else:
        prefix = '<div style="display: flex; justify-content:flex-start; overflow-x: auto;align-items: flex-start; padding: 5px 5px;">'
    suffix = "</div>"

    img_md = '\n'.join([f'<img src="{img}" width="{img_width}" style="display:inline-block; margin-right: 10px;" />' for img in img_ls])
    return prefix + "\n" + img_md + "\n" + suffix



if __name__ == "__main__":
    file_path = "./chen/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.md"
    pdf_path = "./chen/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf"
    document = "./chen/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper_blog.md"
    save_dir = './24_27'
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(document, 'r', encoding='utf-8') as f:
        document = f.read()
    file_path = img_txt_alignment(
        text = document,
        pdf = pdf_path,
        save_dir = save_dir,
        init_grid=3,
        raw_md_text=text,
    )













