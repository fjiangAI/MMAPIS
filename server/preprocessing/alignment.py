from MMAPIS.tools import PDFFigureExtractor,SectionIMGPaths
from MMAPIS.server.summarization import  Section_Summarizer,Article,sectional_split,subgroup,Summary_Integrator
from MMAPIS.tools.utils import strip_title
from typing import Union, List,Tuple
import time
from MMAPIS.config.config import CONFIG, SECTION_PROMPTS,INTEGRATE_PROMPTS
import os
import spacy


def section_level_alignment(
    article:Article,
    pdf_path:str,
    save_dir:str = None,
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
                    section.add_img_paths([f'![img]({path})' for path in section_img_paths.img_path])
                    break
            if flag:
                break
        if flag:
            continue
        else:
            fail_match_img_paths += [f'![img]({path})' for path in section_img_paths.img_path]
    article.raw_document = article.extra_info + '\n' + '\n\n'.join(fail_match_img_paths) + '\n\n' + '\n'.join(['#'*section.rank + ' '+ strip_title(section.title) + '\n'+ '\n\n'.join(section.img_paths) +'\n\n' +section.text for section in article.sections])
    article.section_interpretation = article.extra_info + '\n' + '\n\n'.join(fail_match_img_paths) + '\n\n' + '\n'.join(['#'*section.rank + ' '+ strip_title(section.title) + '\n'+ '\n\n'.join(section.img_paths) +'\n\n' +section.summary for section in article.sections])
    save_t = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    raw_docuemnt_file_name = f'{article.file_name}_{save_t}_raw.md'
    section_interpretation_file_name = f'{article.file_name}_{save_t}_section_interpretation.md'

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
    file_name:str='document_interpretation',
    save_dir:str = None,
    threshold:float = 0.9,
    init_grid:int = 3,
    max_grid:int = 4,
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
            temp_similarities = []
            if img_parent_name:
                temp_similarities.append(max([nlp(section_name).similarity(nlp(img_section_name)),nlp(section_name).similarity(nlp(img_parent_name))]))
            else:
                temp_similarities.append(nlp(section_name).similarity(nlp(img_section_name)))
            similarities.append(max(temp_similarities))
            if not img_parent_name:
                fail_match_img_paths += [f'![img]({path})' for path in section_img_paths.img_path]
                flag = True
                break
            elif img_section_name.lower()  in section_name.lower() or img_parent_name.lower() in section_name.lower():
                flag = True
                section.add_img_paths([f'![img]({path})' for path in section_img_paths.img_path])
                break
        max_similarity = max(similarities)
        if flag:
            continue
        elif max_similarity > threshold and section_img_paths.img_path:
            max_similarity_index = similarities.index(max_similarity)
            document_level_summary.sections[max_similarity_index].add_img_paths([f'![img]({path})' for path in
                                                                  section_img_paths.img_path])
        else:
            fail_match_img_paths += [f'![img]({path})' for path in section_img_paths.img_path]
    if document_level_summary.extra_info:
        extra_info = document_level_summary.extra_info
    else:
        extra_info = article.extra_info
    article.document_interpretation = extra_info + '\n' + '\n\n'.join(
        fail_match_img_paths) + '\n\n' + '\n'.join(['#' * section.rank + ' ' + strip_title(
        section.title) + '\n' + '\n\n'.join(section.img_paths) + '\n\n' + section.text for section in
                                                    document_level_summary.sections])

    save_t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = f'{article.file_name}_{save_t}_{file_name}.md'
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'w',encoding='utf-8') as f:
        f.write(article.document_interpretation.strip())
    return img_paths,article




def document_level_split(
        document_level_summary:str,
):
    pass


if __name__ == "__main__":
    file_path = "./chen/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.mmd"
    pdf_path = "./chen/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf"
    save_dir = './23_128'
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    integrate_prompts = INTEGRATE_PROMPTS
    api_key = CONFIG["openai"]["api_key"]
    base_url = CONFIG["openai"]["base_url"]
    model_config = CONFIG["openai"]["model_config"]
    section_summarizer = Section_Summarizer(
                                        api_key=api_key,
                                        base_url=base_url,
                                        model_config=model_config,
                                        proxy=CONFIG["arxiv"]["proxy"],
                                        prompt_ratio=CONFIG["openai"]["prompt_ratio"],
                                        rpm_limit=CONFIG["openai"]["rpm_limit"],
                                        num_processes=CONFIG["openai"]["num_processes"],
                                        ignore_titles=CONFIG["openai"]["ignore_title"],
                                        )
    article = section_summarizer.section_summarize(
        article_text=text,
        file_name="test",
        summary_prompts=SECTION_PROMPTS,
        init_grid=3,
        max_grid=4)
    print("article:",article.sections)

    img_paths,article = section_level_alignment(
        article=article,
        pdf_path=pdf_path,
        save_dir= save_dir,
    )
    for img in img_paths:
        print('section_name:',img.section_name,'parent:',img.parent)
        print('img_path:',img.img_path)
        print('*'*50)

    integrator = Summary_Integrator(api_key=api_key, base_url=base_url, model_config=model_config)
    print("integrator: ", integrator)
    print("article.section_summary:",article.section_summary)
    flag, response = integrator.integrate_summary(
        section_summaries=article.section_summary,
        integrate_prompts=integrate_prompts,
        response_only=True,
        reset_messages=True)


    img_paths,article = document_level_alignment(
        article=article,
        document_level_summary=response,
        img_paths=img_paths,
        save_dir=save_dir,
        threshold=0.9,
        init_grid=2,
        max_grid=4,
    )













