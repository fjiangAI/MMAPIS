import re
import logging
from MMAPIS.server.summarization.section_class import Article
from MMAPIS.config.config import LOGGER_MODES
from MMAPIS.tools.utils import init_logging,strip_title

logger = init_logging(logger_mode=LOGGER_MODES)

def sectional_split(text,
                    file_name,
                    init_grid = 3,
                    ignore_title = None,
                    max_tokens=16385,
                    max_grid = 4):
    """
    Splits text into chunks of at most max_tokens tokens.
    :param text: str
    :param max_tokens: int
    :return: list of str
    """

    grid = init_grid
    # groups_info : [title,text,length]
    article = Article(text,file_name=file_name,grid=grid,ignore_title=ignore_title,max_grid=max_grid)
    while any([section.length > max_tokens for section in article.sections]):
        grid += 1
        article = Article(text,file_name=file_name,grid=grid,ignore_title=ignore_title)
        # set max grid(The most granular heading level) to 4
        if grid >= max_grid:
            logging.info(f'grid is up to {max_grid},chunking into {max_tokens} tokens')
            break
    logging.info(f'finish split in split grid:{grid}(i.e. {grid*"#"}), with {len(article)} sections,including {len(article.tables)} tables and {len(article.figures)} figures, max section length:{max([section.length for section in article.sections])}')
    logging.debug(f'details:\n{article}\n{article.tables}')
    # in case grid > 2, return subsection_titles
    return article



def assgin_prompts(
                   summary_prompts:dict,
                   query_title:str,
                   replace:bool = True
                   ):

    for key in summary_prompts.keys():
        if key in query_title.lower():
            break

    if 'general' in key:
        title_pattern = re.compile(r"#+\s*(.*)")
        query_title = re.match(title_pattern,query_title).group(1)
        if replace:
            return query_title,summary_prompts[key].replace('[title_to_replace]',query_title)
        else:
            return key,summary_prompts[key].replace('[title_to_replace]',query_title)
    else:
        if replace:
            query_title = strip_title(query_title)
            return query_title,summary_prompts.get(key,'')
        else:
            return key,summary_prompts.get(key,'')

if '__main__' == __name__:
    file_path = "Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.mmd"
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print("read file")
    init_grid = 2
    ignore_titles = ['references']
    article = sectional_split(text,file_path,init_grid,ignore_titles)
    print(article)
    print(article.grid)
    print("title:",article.titles)
    print("author:",article.authors)
    print("affiliations:",article.affiliations)
    print("sections:",article.sections)