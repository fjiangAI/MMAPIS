import re
import logging
import sys
from typing import List, Union
import tiktoken
import spacy
from functools import partial
import reprlib
from MMAPIS.tools.utils import num_tokens_from_messages
from collections.abc import Iterable


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
        self._grouptag = self._title.split(' ')[0].count('#')
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
    def __init__(self, title: str,
                 text: str,
                 subsubgroups: Union[List['subgroup'], 'subgroup'] = None,
                 parent: 'subgroup.title' = '',
                 summary:str = '',
                 img_paths:Union[List[str],str] = None
                 ):
        self._subtext = subtext(text)
        self._title = subtitle(title)
        self._rank = self._title.tag
        self._subsubgroups = self.get_subsubgroup(subsubgroups)
        self._text = self.get_text()
        self._title_text = self._title.title + '\n' + self._text.text
        self._length = len(self)
        self._subtitles = [self.title] + [i.title for i in self._subsubgroups] if self._subsubgroups else [self.title]
        self.parent = parent
        self.summary = summary
        self._img_paths = img_paths if isinstance(img_paths, List) else [img_paths] if img_paths else []


    def __str__(self):
            return f"{str(self._title):<45} {str(self._text):<40} length of subtext:{self._length},rank:{self._rank}"

    def __repr__(self):
        return f"subgroup(title={reprlib.repr(self._title)},text={reprlib.repr(self._text)},subsubgroup={(self._subsubgroups)},parent={self.parent},summary={reprlib.repr(self.summary)},img_paths={reprlib.repr(self._img_paths)})"

    def __len__(self):
        return len(self._title_text)

    def get_subsubgroup(self,subsubgroup)->List['subgroup']:
        if not subsubgroup:
            return []
        if not isinstance(subsubgroup,Iterable):
            return [subsubgroup]
        return subsubgroup

    def get_text(self)->subtext:
        if not self._subsubgroups:
            return self._subtext
        else:
            return subtext(self._subtext.text + '\n'.join([i.title_text for i in self._subsubgroups]))

    def add_img_paths(self,img_paths):
        if isinstance(img_paths,List):
            self._img_paths += img_paths
        else:
            self._img_paths.append(img_paths)



    def group_tag(self):
        return self._title.tag

    @property
    def title(self):
        return self._title.title

    @property
    def text(self):
        return self._text.text

    @property
    def length(self):
        return self._length

    @property
    def title_text(self):
        return self._title_text

    @property
    def rank(self):
        return self._rank

    @property
    def subsubgroup(self):
        return self._subsubgroups

    @property
    def subtext(self):
        return self._subtext

    @property
    def subtitles(self):
        return self._subtitles



    @property
    def img_paths(self):
        return self._img_paths




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
    def __init__(self, article: str,
                 grid: int = 3,
                 max_grid: int = 4,
                 ignore_title:list = ['appendix','reference','acknowledg'],
                 file_name:str = None,
                 raw_document:str = '',
                 section_summary:str = '',
                 section_interpretation:str = '',
                 document_summary:str = '',
                 document_interpretation:str = '',
                ):
        """
        :param article: str, article content
        :param grid: int, the grid of sections
        :param max_grid: int, the max grid of sections
        :param ignore_title: list, ignore title list.e.g. ['appendix','reference']
        :param file_name: str, file name
        :param extra_info: str, extra information
        :param section_summary: str, section summary, plain text only
        :param section_interpretation: str, section interpretation, contains image path
        :param document_summary: str, document summary, plain text only
        :param document_interpretation: str, document interpretation, contains image path
        """
        self.article = re.sub(r'#+\s+Abstract', '## Abstract', article,1)
        self._tables,self._figures = self.get_tables(),self.get_figures()   # remove table and figure from article
        self._file_name = file_name
        self._ignore_title = ignore_title
        self.sections = self.get_sections(grid,max_grid)
        self._titles, self._authors,self._affiliations = self.get_title_contributors()
        self._grid = grid
        self.raw_document = raw_document
        self.section_summary = section_summary
        self.document_summary = document_summary
        self.section_interpretation = section_interpretation
        self.document_interpretation = document_interpretation


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

        authors, affiliations = self.split_contributors(contributors)

        if authors is not None:
            logging.info(f'article :{self._file_name},parse author and affiliation with latex format successfully')
            return titles, self.format_transfer(authors), self.format_transfer(affiliations)
        elif contributors:
            contributors_list = contributors.split('<br><br>')
            authors, affiliations = self.filter_with_NER(contributors_list,NER=['PERSON','ORG'])
            logging.info(f'article :{self._file_name},parse author and affiliation with NER successfully')
            return titles, self.format_transfer(' '.join(authors)),self.format_transfer(' '.join(affiliations))
        else:
            logging.error(f'article :{self._file_name},contributors is None, parser error')
            return titles,contributors, ''



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


    def split_contributors(self,contributors):
        """
        split authors and affiliations from contributors
        :param contributors:
        :return:
        """
        split_pattern = re.compile(r'<br><br>(?=<sup>)|affiliation', re.IGNORECASE)
        try:
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

    def get_sections(self, grid = 2,maxgrid = 4):
        """
        get sections from article
        First, split article into maxgrid sections, then resort sections with grid-level
        :param grid: int, the grid of sections
        :param maxgrid: int, the max grid of sections
        :return: list of subgroup
        """
        #         sub_title_pattern = re.compile(r'\n+(#{{1,{}}}\s+.*?)\n+(.*?)(?=\n\n#{{1,{}}}\s+|$)'.format(maxgrid, maxgrid), re.DOTALL)
        sub_title_pattern = re.compile(r'\n+(#{{1,{}}}\s+.*?)\n+(.*?)(?=\n+#{{1,{}}}\s+|$)'.format(maxgrid, maxgrid), re.DOTALL)
        sub_setions = re.findall(sub_title_pattern, self.article)
        if sub_setions:
            # i[0]:title,i[1]:text
            sections = []
            for sub_setion in sub_setions:
                subtitle, subtext = sub_setion
                if subtitle.strip() != '' and not self.is_ignore_title(subtitle):
                    if subtext.strip().startswith('#'):
                        subtext = "\n"+subtext.strip()
                        matches = re.search(sub_title_pattern, subtext)
                        if matches:
                            sections.append(subgroup(subtitle, ''))  # no text
                            sections.append(subgroup(matches.group(1), matches.group(2)))
                        else:
                            sections.append(subgroup(subtitle, subtext))
                    else:
                        sections.append(subgroup(subtitle, subtext))
            sections = self.resort_sections(sections,grid)
            return sections
        else:
            logging.warning('no subtitle found,parser error')
            return [subgroup('# sub_title is empty,parser error',self.article)]

    @staticmethod
    def resort_sections(sections:List[subgroup],
                       grid:int):
        """
        resort sections with grid
        :param sections:
        :param grid:
        :return:
        """
        res = []
        length = len(sections)
        i = j = 0
        parent = ''
        while i < length:
            if sections[i].rank == 2:
                parent = sections[i].title
            if sections[i].rank < grid:
                res.append(subgroup(title=sections[i].title,
                                    text=sections[i].text,
                                    parent=parent))
                i += 1
                j = i
            elif sections[i].rank == grid:
                j = i + 1
                while j < length:
                    if sections[j].rank <= grid:
                        break
                    sections[j].parent = parent
                    j += 1
                res.append(subgroup(title=sections[i].title,
                                    text=sections[i].text,
                                    subsubgroups=sections[i + 1:j],
                                    parent=parent))
                i = j

            else:
                j = i + 1
                while j < length:
                    if sections[j].rank <= grid:
                        break
                    sections[j].parent = parent
                    j += 1

                res.append(subgroup(title=sections[i].title,
                                    text=sections[i].text,
                                    subsubgroups=sections[i + 1:j],
                                    parent=parent))
                    # sections[i].title, '\n'.join([sections[k].title_text for k in range(i, j)])))
                i = j

        return res

    @staticmethod
    def judge_na(text):
        if text is None:
            return True
        elif text.strip() == '':
            return True
        else:
            return False

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

    @property
    def file_name(self):
        return self._file_name

    @property
    def grid(self):
        return self._grid

    @property
    def extra_info(self):
        titles = '' if self.judge_na(self._titles) else self._titles.replace('<br>','\n') + "\n"
        authors = '' if self.judge_na(self._authors) else "- Authors: " + self._authors.replace('<br>','\n') + "\n"
        affiliations = '' if self.judge_na(self._affiliations ) else "- Affiliations: " + self._affiliations.replace('<br>','\n') + "\n"
        self._extra_info = '\n'.join([titles, authors, affiliations])
        return self._extra_info

    def __str__(self):
        res_msg = f"Article name:{self._file_name}，after split:\n"
        for i in self.sections:
            res_msg += f"{str(i)}\n"
        return res_msg


    def __repr__(self):
        return f"Article({reprlib.repr(self.article)},grid={self._grid},ignore_title={reprlib.repr(self._ignore_title)},file_name={reprlib.repr(self._file_name)},extra_info={reprlib.repr(self.extra_info)},section_summaries={reprlib.repr(self.section_summaries)},docu_level_summary={reprlib.repr(self.docu_level_summary)})"

    def __getitem__(self, item):
        return self.sections[item]

    def __len__(self):
        return len(self.sections)

    def __iter__(self):
        return iter(self.sections)

    def __next__(self):
        pass

    def iter_sections(self):
        return iter([[i.title,i.text] for i in self.sections])


if __name__ == "__main__":
    text = "In this paper, we propose a novel method for image captioning with verb-specific semantic roles."
    title = "## 1. Introduction"
    group = subgroup(
        title= title,
        text= text
    )
    print(type(group))
    print(isinstance(group,Iterable))