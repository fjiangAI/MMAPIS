import re
import logging
import sys
from typing import List, Union
import tiktoken
import spacy
from functools import partial
import reprlib



def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613"):
    """
        Returns the number of tokens used by a list of messages.
        Args:
            messages: A list of messages. Each message is:
                dict (with keys "role" and "content".)
                string (in which case the role is assumed to be "user".)
                list (already normalized into a list of dicts.)
            model: The model to use. Defaults to "gpt-3.5-turbo-16k-0613".
        Returns:
            The number of tokens used by the messages.
    """
    # try:
    #     encoding = tiktoken.encoding_for_model(model)
    try:
        map_dict = {
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-4": "cl100k_base",
            "text-embedding-ada": "cl100k_base",
            "text-davinci": "p50k_base",
            "Codex": "p50k_base",
            "davinci": "p50k_base",
        }
        for key in map_dict.keys():
            if key in model:
                encode_model = map_dict[key]
                break
        encoding = tiktoken.get_encoding(encode_model)
    except KeyError:
        logging.error(f"model {model} not found,load default model: cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in "gpt-3.5-turbo-16k-0613":  # note: future models may deviate from this
        if isinstance(messages, dict):
            messages = [messages]
        elif isinstance(messages, list):
            pass
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
        See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")



class subtitle():
    def __init__(self, title: str):
        self._title = title
        self._update_grouptag()

    def __str__(self):
        return f"subtitle:{self._title}"
    def __repr__(self):
        return f"subtitle({self._title})"

    @property
    def title(self):
        return self._title

    @property
    def tag(self):
        return self._grouptag

    def _update_grouptag(self):
        self._grouptag = self._title.count('#')
        if self._grouptag == 0:
            logging.error(f'no # in subtitle: {self._title}, parser error')
            self._grouptag = None

    def __len__(self):
        return num_tokens_from_messages(self._title)



class subtext():
    def __init__(self, text: str):
        self._text = text

    def __str__(self):
        return f"subtext:{reprlib.repr(self._text)}"
    def __repr__(self):
        return f"subtext({reprlib.repr(self._text)})"

    def __len__(self):
        return num_tokens_from_messages(self._text)

    @property
    def text(self):
        return self._text




class subgroup():
    def __init__(self, title: str, text: str):
        self._text = subtext(text)
        self._title = subtitle(title)

    def __len__(self):
        return len(self._text)

    def __str__(self):
        return f"{str(self._title):<45} {str(self._text):<40} length of subtext:{len(self._text)}"

    def __repr__(self):
        return f"subgroup(title={repr(self._title)},text={repr(self._text)})"
    def group_tag(self):
        return self._title.tag

    @property
    def title(self):
        return self._title.title

    @property
    def text(self):
        return self._text.text

class Table():
    def __init__(self,
                 table:str,
                 caption:str):
        self._table = table
        self._caption = caption

    @property
    def table(self):
        return self._table

    @property
    def caption(self):
        return self._caption

    def __str__(self):
        return f"table:{reprlib.repr(self._table):<45} caption:{reprlib.repr(self._caption):<40}"

    def __repr__(self):
        return f"Table(table={reprlib.repr(self._table)},caption={reprlib.repr(self._caption)})"


class Tables():
    def __init__(self,tables:List[Table]):
        self._tables = tables

    def __str__(self):
        msg = 'Tables:\n'
        for i in self._tables:
            msg += f'{i}\n'
        return msg

    def __repr__(self):
        return f"Tables(tables={reprlib.repr(self._tables)})"

    def __getitem__(self, item):
        return self._tables[item]

    def __len__(self):
        return len(self._tables)

    def __iter__(self):
        return iter(self._tables)

    def __next__(self):
        pass

class Article():
    def __init__(self, article: str,grid: int = 2,ignore_title:list = None,file_name:str = None):
        self.article = re.sub(r'#+\s+Abstract', '## Abstract', article,1)
        self._tables = self.get_tables()
        self._figures = self.get_figures()
        self._file_name = file_name
        self._ignore_title = ignore_title
        self.groups = self.get_subgroups(grid)
        self._titles, self._authors,self._affiliations = self.get_title_contributors()


    def get_tables(self):
        """
        get table from article, including table description and table content(latex format)
        :return: list of Table
        """
        table_pattern = r"(\\begin{table}.*?\\end{table})\W+(.*?)\n"
        table_pattern = re.compile(table_pattern, re.DOTALL)
        matches = re.finditer(table_pattern, self.article)
        tables = []
        for match in matches:
            self.article = self.article.replace(match.group(0), '')
            tables.append(Table(match.group(1), match.group(2)))
        return Tables(tables)

    def get_figures(self):
        """
        get figure dicrption from article
        :return: list of str
        """
        figure_caption_pattern = r"\n(Figu?r?e?\.?\W?\d+[:|\.].*?)(?=\n|$)"
        figure_pattern = re.compile(figure_caption_pattern, re.DOTALL)
        matches = re.finditer(figure_pattern, self.article)
        figure_captions = []
        for match in matches:
            self.article = self.article.replace(match.group(1), '')
            figure_captions.append(subtext(match.group(1)))
        return figure_captions


    def get_title_contributors(self):
        title_pattern = re.compile(r'(#\s+.*?)\n+(.*?)#+', re.DOTALL)
        res = re.match(title_pattern, self.article)
        if res is None:
            logging.error(f'article :{self._file_name},Title not found, parser error')
            return None,None,None
            # raise ValueError('No title and authors found,parser error')
        titles, contributors = res.group(1).strip(), res.group(2).strip()

        # transfer authors to markdown format
        contributors = re.sub(r'\\+\(\{\}\^\{\\*([^\\]*?)\\*\}\\+\)(\\+\(\{\}\^\{([^\\]*?)\}\\+\))?',r'<sup>\1,\3</sup>', contributors)
        contributors = re.sub(r'>([^,]*?),<',r'>\1<',contributors)
        contributors = re.sub(r'\n', r'<br>', contributors)

        authors, affiliations = self.get_authors_institution(contributors)

        if authors is not None:
            logging.info(f'article :{self._file_name},parse author and affiliation with latex format successfully')
            return titles, self.format_transfer(authors), self.format_transfer(affiliations)
        elif contributors:
            contributors_list = contributors.split('<br><br>')
            authors, affiliations = self.filter_with_NER(contributors_list,NER=['PERSON','ORG'])
            logging.info(f'article :{self._file_name},parse author and affiliation with NER successfully')
            return titles, self.format_transfer(''.join(authors)),self.format_transfer(''.join(affiliations))
        else:
            logging.error(f'article :{self._file_name},contributors is None, parser error')
            return titles,contributors, None



    def format_transfer(self,text):
        format_dic = {
            r'>dagger<': '>&dagger;<',
            r'>ddagger<': '>&ddagger;<',
            r'>S<': '>&sect;<',
            r'>P<': '>&para;<',
            r'>clubsuit<': '>&clubs;<',
            r'>diamondsuit<': '>&diams;<',
            r'>heartsuit<': '>&hearts;<',
            r'>spadesuit<': '>&spades;<',
            r'>flat<': '>&flat;<',
            r'>natural<': '>&natural;<',
            r'>sharp<': '>&sharp;<',
        }

        for key, value in format_dic.items():
            pattern = re.compile(rf'{key}', re.IGNORECASE)
            text = re.sub(pattern, f'{value}', text)

        return text


    def get_authors_institution(self,contributors):
        split_pattern = re.compile(r'<br><br>(?=<sup>)|affiliation', re.DOTALL)
        try :
            author, affiliation = re.split(split_pattern, contributors, 1)
        except ValueError:
            return None,None
        # transfer author and affiliation to markdown format
        author_pattern = re.compile(r'([^<>]*?<sup>.*?</sup>)')
        authors = re.findall(author_pattern, author)
        affiliation_pattern = re.compile(r'(<sup>.*?</sup>.*?)(?=<sup>|\n+|$)')
        affiliations = re.findall(affiliation_pattern, affiliation)
        authors,affiliations = self.list_strip(authors),self.list_strip(affiliations)
        return '    '.join(self.filter_with_NER(authors,NER='PERSON')), '   '.join(self.filter_with_NER(affiliations,NER='ORG'))

    @staticmethod
    def list_strip(l: List[str]):
        return [i.strip() for i in l]

    @staticmethod
    def filter_with_NER(text: List[str], NER: Union[str, List[str]] = 'PERSON'):
        assert NER in ['ORG', 'PERSON', ['ORG', 'PERSON'], ['PERSON', 'ORG']], logging.error(
            f'NER must be ORG or PERSON or ["ORG", "PERSON"], but got {NER}')
        if NER == 'PERSON':
            pure_name_pattern = re.compile(r'([^<>]*?)<sup>.*?</sup>')
        else:
            pure_name_pattern = re.compile(r'<sup>.*?</sup>([^<>]*?)')

        def filter_func(doc, NER=NER):
            for ent in doc.ents:
                if ent.label_ != NER:
                    return False
            return True

        nlp = spacy.load("en_core_web_sm")
        # input text is filtered by regular expression, so it format is fixed
        if isinstance(NER, str):
            pure_text = [re.search(pure_name_pattern, i).group(1) for i in text]
            docs = [nlp(i) for i in pure_text]
            return [text[i] for i in range(len(text)) if filter_func(docs[i])]
        # input text is not formatted
        else:
            filter_func = partial(filter_func, NER=NER[0])
            docs = [nlp(i) for i in text]
            list1 = []
            list2 = []
            for index, i in enumerate(docs):
                if filter_func(i):
                    list1.append(text[index])
                else:
                    list2.append(text[index])
            return list1, list2


    def is_ignore_title(self,title):
        '''
        check if title is in ignore list
        Args:
            title: str, subtitle result of regular expression
            self._ignore_title: list, ignore title list.e.g. ['appendix','reference']
        Returns:


        '''
        if self._ignore_title is None:
            return False
        for i in self._ignore_title:
            if i in title.strip().lower():
                return True
        return False

    def get_subgroups(self, grid = 2):
        sub_title_pattern = re.compile(r'\n+(#{{1,{}}}\s+.*?)\n+(.*?)(?=\n\n#{{1,{}}}\s+|$)'.format(grid, grid), re.DOTALL)
        sub_title = re.findall(sub_title_pattern, self.article)
        if sub_title:
            # i[0]:title,i[1]:text
            return [subgroup(i[0], i[1]) for i in sub_title if i[0].strip() != '' and not self.is_ignore_title(i[0])]
        else:
            logging.warning('sub_title is empty,parser error')
            return [subgroup('# sub_title is empty,parser error',self.article)]

    @property
    def titles(self):
        return self._titles

    @property
    def authors(self):
        return self._authors

    @property
    def affiliations(self):
        return self._affiliations

    @property
    def tables(self):
        return self._tables

    @property
    def figures(self):
        return self._figures

    def __str__(self):
        res_msg = f"file_name:{self._file_name}，after split:\n"
        for i in self.groups:
            res_msg += f"{str(i)}\n"
        return res_msg


    def __repr__(self):
        return f"Article({reprlib.repr(self.article)},grid={self._grid},ignore_title={self._ignore_title})"

    def __getitem__(self, item):
        return self.groups[item]

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __next__(self):
        pass

    def iter_groups(self):
        return iter([[i.title,i.text] for i in self.groups])





def half_split(chunk):
    chunk,num_tokens = chunk
    new_chunks = []
    len_chunk = len(chunk)
    mid_point = len_chunk // 2
    # Find next space
    while chunk[mid_point] != " ":
        mid_point += 1
    ## roughly assume that each char is 1 token
    new_chunks += [(chunk[:mid_point],num_tokens//2 + mid_point-len_chunk//2),(chunk[mid_point:],num_tokens//2 + len_chunk//2 - mid_point)]
    return new_chunks


def split2pieces(text,file_name, max_tokens=16385,mode = 'group',init_grid = 2,ignore_title = None):
    """
    Splits text into chunks of at most max_tokens tokens.
    :param text: str
    :param max_tokens: int
    :return: list of str
    """
    assert mode in ['half','group'],f'mode should be half or group,while {mode} is given'
    ## [text,len(text)]
    chunks = [(text,num_tokens_from_messages(text))]

    # Split chunks in half until they are small enough
    if mode == 'half':
        while any([i[1] > max_tokens for i in chunks]):
            new_chunks = []
            for chunk in chunks:
                if chunk[1] < max_tokens:
                    new_chunks.append(chunk)
                    continue
                else:
                    new_chunks += half_split(chunk)
            chunks = new_chunks
        if chunks:
            logging.info('finish split,max length of part:',max([i[1] for i in chunks]))
        else:
            logging.warning('chunks is empty')
        # keep return format same as group_split
        return None,None,None,[chunk[0] for chunk in chunks]

    elif mode == 'group':
        grid = init_grid
        # groups_info : [title,text,length]
        article = Article(text,file_name=file_name,grid=grid,ignore_title=ignore_title)

        groups_info = [[i.title,i.text,len(i)] for i in article.groups]
        titles,authors,affiliations = article.titles,article.authors,article.affiliations
        while any([i[-1] > max_tokens for i in groups_info]):
            grid += 1
            # subtitle_tag : [(title,group_tag)]
            # subtext_tag : [text,length,tag]
            article = Article(text,file_name=file_name,grid=grid,ignore_title=ignore_title)
            groups_info = [[i.title,i.text,len(i)] for i in article.groups]
            # set max grid(The most granular heading level) to 4
            if grid == 4:
                groups_info = [[i[0],i[1][:max_tokens],max_tokens] for i in groups_info]
                logging.info(f'grid:{grid} ,cut text to max_tokens:{max_tokens}')
                break

        logging.info(f'finish split with {len(groups_info)} parts,including {len(article.tables)}tables, split grid:{grid}(i.e. {grid*"#"}),max length of part:{max([i[-1] for i in groups_info])},details:\n{article}\n{article.tables}')
        return titles,authors,affiliations,groups_info,article.tables

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
    return key,summary_prompts.get(key,'')

