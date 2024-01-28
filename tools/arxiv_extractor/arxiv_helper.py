import logging
from typing import Union
import re
from bs4 import BeautifulSoup
import pandas as pd
from MMAPIS.config.config import LOGGER_MODES, CONFIG
from MMAPIS.tools.utils import init_logging
from MMAPIS.tools.arxiv_extractor.crawler_base import CrawlerBase,Article
import reprlib




class ArxivCrawler(CrawlerBase):
    def __init__(self,
                 proxy=None,
                 headers=None,
                 max_retry=3,
                 timeout=1000):
        # Initialization, if needed (e.g., setting base URLs, API keys)
        super().__init__(proxy,headers,max_retry,timeout)

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
                            ):
        soup = BeautifulSoup(response.text, 'html.parser')
        # get the information of the papers
        content = soup.find_all('p', {'class': 'list-title is-inline-block'}, limit=2 * max_return)
        links = []
        for i in range(len(content)):
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
            title, abstract, author = self.extra_pure_text(title_t, abs_t, author_t, line_length=line_length)
            title = '[' + title + '](' + links[i] + ')'
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
        cross_lists = cross_lists_tag.find_next("dl")
        links_containers = cross_lists.find_all("span", attrs={"class": "list-identifier"}, limit=2 * max_return)
        parsed_links = []
        for i, container in enumerate(links_containers):
            link = container.find("a", attrs={"title": "Download PDF"})
            if link is not None:
                parsed_links.append("https://arxiv.org" + link["href"])
            else:
                logging.warning(f'No link found in article {i}')
                parsed_links.append(None)
        # links = cross_lists.find_all("a", attrs={"title": "Download PDF"})
        # parsed_links = ["https://arxiv.org" + link["href"] for link in links]
        articles = []
        meta_data = cross_lists.find_all("div", attrs={"class": "meta"}, limit=2 * max_return)
        for i, meta_info in enumerate(meta_data):
            # get raw title and abstract(html format) from meta_info
            title_text = str(meta_info.find("div", attrs={"class": "list-title mathjax"}))
            abs_text = str(meta_info.find("p", attrs={"class": "mathjax"}))
            authors = str(meta_info.find("div", attrs={"class": "list-authors"}))
            title, abstract, authors = self.extra_pure_text(title_text, abs_text, authors, line_length=line_length)
            title = '[' + title + '](' + parsed_links[i] + ')'
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
                                                )
        return articles

    def run_daily_crawler(self,
                            daily_type:str = 'cs',
                            max_return=10,
                            line_length=15,
                            ):
        response = self.get_daily_data(daily_type=daily_type)
        articles = self.parse_daily_data(response=response,
                                                max_return=max_return,
                                                line_length=line_length,
                                                )
        return articles




    def download_pdf(self,
                    article_url: str,
                    file_path: str,
                    ):
        response = self.get_response(article_url)
        if response.status_code == 200:
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
                        line_length=15):
        """Extract pure text of title and abstract."""
        pattern = r">([^<]+)<"
        title = re.findall(pattern, title_text)
        abstract = re.findall(pattern, abs_text)
        authors = re.findall(pattern, authors_text)

        title = self.html2md(''.join(title), line_length=line_length)
        authors = self.html2md(''.join(authors), line_length=line_length)
        abstract = ''.join(abstract)
        hidden_index = abstract.find('▽')
        abstract = abstract[:hidden_index]
        abstract = self.html2md(abstract, line_length=line_length)

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

# Example usage
if __name__ == "__main__":
    logger = init_logging(logger_mode=LOGGER_MODES)
    arxiv = ArxivCrawler(proxy=CONFIG['arxiv']['proxy'],headers=CONFIG['arxiv']['headers'])
    print(arxiv)
    keyword_articles = arxiv.run_keyword_crawler(key_word="quantum physics", max_return=10)

    for article in keyword_articles:
        print(article)
    print("*" * 100)
    daily_articles = arxiv.run_daily_crawler(daily_type='cs', max_return=10)
    for d_article in daily_articles:
        print(d_article)
    arxiv.download_pdf(d_article.pdf_url, f"{d_article.pdf_url.split('/')[-1]}.pdf")


