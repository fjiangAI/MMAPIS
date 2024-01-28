import requests
from tenacity import retry, stop_after_attempt, wait_fixed
import reprlib

class Article:
    def __init__(self, title, abstract, authors, pdf_url):
        self._title = title
        self._abstract = abstract
        self._authors = authors
        self._pdf_url = pdf_url

    def __str__(self):
        return f"Article: {self._title},url:{self._pdf_url}"

    def __repr__(self):
        return f"Aritcle(title={reprlib.repr(self._title)}, abstract={reprlib.repr(self._abstract)}, authors={reprlib.repr(self._authors)}, pdf_url={reprlib.repr(self._pdf_url)})"

    @classmethod
    def from_dict(cls, data: dict):
        if list(data.keys())[0].startswith('_'):
            data = {key.lstrip('_'): value for key, value in data.items()}
        return cls(**data)

    @property
    def title(self):
        return self._title

    @property
    def abstract(self):
        return self._abstract

    @property
    def authors(self):
        return self._authors

    @property
    def pdf_url(self):
        return self._pdf_url




class CrawlerBase:
    def __init__(self, proxy=None, headers=None, max_retry=3, timeout=1000):
        self.proxy = proxy
        self.headers = headers
        self.max_retry = max_retry
        self.timeout = timeout
        self.get_response = retry(stop=stop_after_attempt(self.max_retry), wait=wait_fixed(self.timeout))(
            self._get_response)

    def _get_response(self, url: str):
        response = requests.get(url, proxies=self.proxy, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f'HTTP error, status = {response.status_code}')
        return response

    def get_data(self):
        pass

    def parse_data(self):
        pass

    def save_data(self):
        pass

    def run_crawler(self):
        raise NotImplementedError