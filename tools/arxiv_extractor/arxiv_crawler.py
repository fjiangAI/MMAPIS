import logging
import os.path as osp
import os
from typing import Union
import re
from bs4 import BeautifulSoup
import pandas as pd
from MMAPIS.config.config import LOGGER_MODES, GENERAL_CONFIG,ARXIV_CONFIG
from MMAPIS.tools.utils import init_logging
from MMAPIS.tools.arxiv_extractor.crawler_base import CrawlerBase,ArxivArticle as Article
import reprlib
from typing import Union
from tqdm import tqdm




class ArxivCrawler(CrawlerBase):
    def __init__(self,
                 proxy=GENERAL_CONFIG['proxy'],
                 headers=GENERAL_CONFIG['headers'],
                 max_retry=3,
                 timeout=1000,
                 download: bool = ARXIV_CONFIG['download'],
                 save_dir: str = GENERAL_CONFIG['save_dir'],
                 **kwargs):
        # Initialization, if needed (e.g., setting base URLs, API keys)
        super().__init__(proxy,headers,max_retry,timeout,download,save_dir,**kwargs)

    def __repr__(self):
        return f"ArxivCrawler(proxy:{reprlib.repr(self.proxy)},headers:{reprlib.repr(self.headers)},max_retry:{self.max_retry},timeout:{self.timeout})"

    def get_keyword_data(self,
                         key_word: Union[str, None] = None,
                         searchtype='all',
                         abstracts='show',
                         order='-announced_date_first',
                         size=50,
                        ):
        if not key_word:
            return self.get_daily_data()

        assert searchtype in ['all', 'title', 'abstract', 'author', 'comment',
                              'journal_ref', 'subject_class', 'report_num', 'id_list'], \
            f"searchtype:{searchtype} does not in one of ['all','title','abstract','author'," \
            f"'comment','journal_ref','subject_class','report_num','id_list']"
        assert abstracts in ['show', 'hide'], f"abstracts:{abstracts} does not in one of ['show','hide']"
        assert order in ['-announced_date_first', 'submitted_date',
                         '-submitted_date', 'announced_date_first', ''], \
            f"order:{order} does not in one of ['-announced_date_first'," \
            f"'submitted_date','-submitted_date','announced_date_first','']"
        assert size in [25, 50, 100, 200], f"size:{size} does not in one of [25,50,100,200]"

        # transfer key_word to url format
        if key_word.startswith('http'):
            url = key_word
        else:
            key_word = key_word.replace(' ', '+')
            url = f'https://arxiv.org/search/?query={key_word}&searchtype={searchtype}' \
                  f'&abstracts={abstracts}&order={order}&size={size}'
        logging.info(f"fetching {url}\nkew_word:{key_word}")
        self.key_word = key_word

        try:
            response = self.get_response(url)
        except Exception as e:
            logging.error(f'All {self.max_retry} retries failed, the error message is: {str(e)}')
            raise e
        return response

    def parse_keyword_data(self,
                           response,
                           max_return=10,
                           line_length=15,
                           return_md=True,
                        ):
        soup = BeautifulSoup(response.text, 'html.parser')
        # get the information of the papers
        content = soup.find_all('p', {'class': 'list-title is-inline-block'}, limit=2 * max_return)
        links = []
        for i in range(len(content)):
            print("content[i]:",content[i])
            link = content[i].find_all('a', string=re.compile("pdf"))
            if link:
                links.append(link[0]['href'])
            else:
                logging.warning(f"Could not find pdf link for {content[i].find('a')['href']}")
                links.append(None)
        title_text = soup.find_all("p", attrs={"class": "title is-5 mathjax"}, limit=2 * max_return)
        abs_text = soup.find_all("p", attrs={"class": "abstract mathjax"}, limit=2 * max_return)
        author_text = soup.find_all("p", attrs={"class": "authors"}, limit=2 * max_return)
        articles = []
        for i, (title_t,abs_t,author_t) in enumerate(zip(title_text, abs_text, author_text)):
            title_t, abs_t, author_t = str(title_t), str(abs_t), str(author_t)
            title, abstract, author = self.extra_pure_text(title_t, abs_t, author_t, line_length=line_length, return_md= return_md)
            if return_md:
                title = '[' + title + '](' + links[i] + ')' if links[i] is not None else title
            article = Article(title=title, abstract=abstract, authors=author, pdf_url=links[i])
            articles.append(article)
        articles = self.remove_na_duplicates(articles)
        num_links = len(links)
        if num_links < max_return:
            logging.info(f'found links({num_links}) < max_return({max_return}) with keywork {self.key_word},return all {num_links} links')
        else:
            logging.info(f'found links({num_links}) > max_return({max_return}) with keywork {self.key_word},return {max_return} links')
            num_links = max_return

        return articles[:num_links]




    def get_daily_data(self, daily_type:str = 'cs'):
        type_l = ['cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'eess', 'econ', 'astro-ph', 'cond-mat', 'gr-qc',
                  'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nucl-ex', 'nucl-th', 'quant-ph']
        assert daily_type in type_l, f"daily_type:{daily_type} must be one of {type_l}"
        url = "https://arxiv.org/list/" + daily_type + "/new"
        ## url = f"http://xxx.itp.ac.cn/list/{daily_type}/new"
        print(f"url:{url}")
        logging.info(f"fetching {url}")

        try:
            response = self.get_response(url)
        except Exception as e:
            logging.error(f'All {self.max_retry} retries failed, the error message is: {str(e)}')
            raise e
        return response


    def parse_daily_data(self,
                         response=None,
                         max_return: int = 10,
                         line_length: int = 15,
                         return_md: bool = True,
                        ):
        """
        Search for articles on arXiv matching the query.

        :param query: A string representing the search query.
        :param max_results: Maximum number of results to return.
        :return: A list of articles (dictionaries or custom objects) matching the query.
        """
        # Implement the search functionality
        # This could involve sending a request to arXiv's search API
        # parse html
        soup = BeautifulSoup(response.text, "html.parser")

        # Links are all in <dl> tag after New submissions
        cross_lists_tag = soup.find("h3", string=re.compile("New submissions"))
        # Get next tag
        ## Cross submissions list
        cross_lists = cross_lists_tag.find_next("dl")
        ## New submissions list
        # new_lists = cross_lists.find_previous("dl")
        links_containers = cross_lists.find_all("dt", limit=2 * max_return)
        parsed_links = []
        for i, container in enumerate(links_containers):
            link = container.find("a", attrs={"title": "Download PDF"})
            print(f"link:{link}")
            if link is not None:
                parsed_links.append("https://arxiv.org" + link["href"])
                ## parsed_links.append("http://xxx.itp.ac.cn" + link["href"])
            else:
                logging.warning(f'No link found in article {i}')
                parsed_links.append(None)
        # links = cross_lists.find_all("a", attrs={"title": "Download PDF"})
        # parsed_links = ["https://arxiv.org" + link["href"] for link in links]
        print(f"parsed_links:{len(parsed_links)}")
        articles = []
        meta_data = cross_lists.find_all("div", attrs={"class": "meta"}, limit=2 * max_return)
        for i, meta_info in enumerate(meta_data):
            # get raw title and abstract(html format) from meta_info
            title_text = str(meta_info.find("div", attrs={"class": "list-title mathjax"}))
            abs_text = str(meta_info.find("p", attrs={"class": "mathjax"}))
            authors = str(meta_info.find("div", attrs={"class": "list-authors"}))
            title, abstract, authors = self.extra_pure_text(title_text, abs_text, authors, line_length=line_length,return_md=return_md)
            if return_md:
                title = '[' + title + '](' + parsed_links[i] + ')' if parsed_links[i] is not None else title
            article = Article(title=title, abstract=abstract, authors=authors, pdf_url=parsed_links[i])
            articles.append(article)

        articles = self.remove_na_duplicates(articles)

        num_links = len(articles)
        if max_return < num_links:
            logging.info(f"Found {num_links} new papers > max_return:{max_return} today, return {max_return} links")
            num_links = max_return
        else:
            logging.info(f"Found {num_links} new papers < max_return:{max_return} today, return {num_links} links")
        return articles[:num_links]


    def run_keyword_crawler(self,
                            key_word: Union[str, None] = None,
                            searchtype='all',
                            abstracts='show',
                            order='-announced_date_first',
                            size=50,
                            max_return=10,
                            line_length=15,
                            return_md: bool = True,
                            ):
        response = self.get_keyword_data(key_word=key_word,
                                         searchtype=searchtype,
                                         abstracts=abstracts,
                                         order=order,
                                         size=size,
                                         )
        articles = self.parse_keyword_data(response=response,
                                           max_return=max_return,
                                           line_length=line_length,
                                           return_md=return_md,
                                           )
        if self.download:
            articles = tqdm(articles)
            for article in articles:
                dir_name = article.pdf_url.split('/')[-1].replace('.', '_')
                file_name = dir_name + '.pdf'
                file_path = osp.join(self.save_dir, dir_name, file_name)
                articles.set_description(f"Downloading {article.pdf_url} to dir: {osp.abspath(file_path)}")
                self.download_pdf(article_url=article.pdf_url,
                                  file_path=file_path,
                                  )
                article.pdf_path = file_path
        return articles

    def run_daily_crawler(self,
                            daily_type:str = 'cs',
                            max_return=10,
                            line_length=15,
                            return_md: bool = True,
                            ):
        response = self.get_daily_data(daily_type=daily_type)
        articles = self.parse_daily_data(response=response,
                                         max_return=max_return,
                                         line_length=line_length,
                                         return_md= return_md,
                                        )
        if self.download:
            articles = tqdm(articles)
            for article in articles:
                dir_name = article.pdf_url.split('/')[-1].replace('.', '_')
                file_name = dir_name + '.pdf'
                file_path = osp.join(self.save_dir, dir_name, file_name)
                articles.set_description(f"Downloading {article.pdf_url} to dir: {osp.abspath(file_path)}")
                self.download_pdf(article_url=article.pdf_url,
                                  file_path=file_path,
                                  )
                article.pdf_path = file_path
        return articles




    def download_pdf(self,
                    article_url: str,
                    file_path: str,
                    ):
        response = self.get_response(article_url)
        if response.status_code == 200:
            dir_path = osp.dirname(osp.abspath(file_path))
            if not osp.exists(dir_path):
                os.makedirs(dir_path)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {article_url} to {file_path}")
            return True
        else:
            logging.error(f"Failed to download {article_url}, status code: {response.status_code}")
            return False


    @staticmethod
    def remove_na_duplicates(items: list):
        """Remove duplicate and NA links."""
        if isinstance(items[0], Article):
            original_len = len(items)
            data = [vars(item) for item in items]
            df = pd.DataFrame(data)
            df = df.dropna(subset=['_pdf_url']).drop_duplicates(subset=['_pdf_url'])
            logging.info("exist {} NA or duplicate links".format(original_len - len(df)))
            return [Article.from_dict(row) for row in df.to_dict('records')]

        else:
            links = items
            no_na = list(filter(lambda x: x is not None, links))
            no_na_dup = list(set(no_na))
            return sorted(no_na_dup, key=links.index)

    def extra_pure_text(self,
                        title_text: str,
                        abs_text: str,
                        authors_text: str,
                        line_length=15,
                        return_md=True):
        """Extract pure text of title and abstract."""
        pattern = r">([^<]+)<"
        title = re.findall(pattern, title_text)
        abstract = re.findall(pattern, abs_text)
        authors = re.findall(pattern, authors_text)
        abstract, title, authors = ''.join(abstract).strip(), ''.join(title).strip(), ''.join(authors).strip()
        hidden_index = abstract.find('▽')
        abstract = abstract[:hidden_index]
        if return_md:
            abstract = self.html2md(abstract, line_length=line_length)
            title = self.html2md(title, line_length=line_length)
            authors = self.html2md(authors, line_length=line_length)

        return title, abstract, authors

    def html2md(self,html: str, line_length=15, return_latex=True):
        """Convert html to string."""
        html_dic = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&nbsp;': ' ',
            '&quot;': '"',
            '&apos;': "'",
            '&deg;': '°'
        }
        html = html.replace('\n', ' ').replace('      ', '').replace('\\', ' ').replace('$', '').replace('dagger',
                                                                                                         '&dagger;').replace(
            '-', '–')
        for key in html_dic.keys():
            html = html.replace(key, html_dic[key])
        html = self.line_break(html, n=line_length)
        if not return_latex:
            return html.strip()
        return "$$" + html.replace(' ', '~').strip() + "$$"

    @staticmethod
    def line_break(text: str, n=15):
        """Add line break."""
        text = text.split(' ')
        for i in range(1, len(text) + 1):
            if i % n == 0:
                text[i - 1] = text[i - 1] + '\\\\'
        return ' '.join(text)


if __name__ == "__main__":
    crawler = ArxivCrawler()
    print("crawler:",crawler)
    articles = crawler.run_keyword_crawler(key_word='quantum computing', max_return=5)
    print(articles)
    articles = crawler.run_daily_crawler(daily_type='cs', max_return=5)
    print(articles)
    print("Done")
