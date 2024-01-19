import logging
import sys
import time
from retrying import  retry
import requests
from bs4 import BeautifulSoup #!pip install beautifulsoup4
import re
import os
# import winreg
import pandas as pd


def remove_duplicates(items:list):
    """Remove duplicate and NA links."""
    if isinstance(items[0],list):
        df = pd.DataFrame({'links':items[0],'title': items[1],'abstract' : items[2],'authors':items[3]})
        df = df.dropna(subset=['links']).drop_duplicates(subset=['links'])
        links,title,abstract,authors = df['links'].tolist(),df['title'].tolist(),df['abstract'].tolist(),df['authors'].tolist()
        return links,title,abstract,authors

    else:
        links = items
        no_dup = list(filter(lambda x: x is not None, links))
        no_dup = list(set(no_dup))
        return sorted(no_dup, key=links.index)


def html2md(html: str,line_length=15,return_latex=True):
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
    html = html.replace('\n', ' ').replace('      ', '').replace('\\', ' ').replace('$','').replace('dagger','&dagger;').replace('-','–')
    for key in html_dic.keys():
        html = html.replace(key, html_dic[key])
    html = line_break(html,n=line_length)
    if not return_latex:
        return html.strip()
    return "$$" + html.replace(' ', '~').strip() + "$$"


def extra_pure_text(title_text: str,
                    abs_text: str,
                    authors_text: str,
                    line_length=15):
    """Extract pure text of title and abstract."""
    pattern = r">([^<]+)<"
    title = re.findall(pattern, title_text)
    abstract = re.findall(pattern, abs_text)
    authors = re.findall(pattern, authors_text)

    title = html2md(''.join(title),line_length=line_length)
    authors = html2md(''.join(authors),line_length=line_length)
    abstract = ''.join(abstract)
    more_index = abstract.find('▽')
    abstract = abstract[:more_index]
    abstract = html2md(abstract,line_length=line_length)

    return title, abstract, authors


def line_break(text: str, n=15):
    """Add line break."""
    text = text.split(' ')
    for i in range(1,len(text)+1):
        if i % n == 0:
            text[i-1] = text[i-1] + '\\\\'
    return ' '.join(text)




# def html2md(html:str):
#     """Convert html to string."""
#     html_dic = {
#         '&amp;':'\&',
#         '&lt;':'<',
#         '&gt;':'>',
#         '&nbsp;':' ',
#         '&quot;':'"',
#         '&apos;':"'",
#         '&deg;':'°'
#     }
#     html = html.replace('\n','').replace('      ','').replace(' ','~').replace('\\','\\\\')
#     for key in html_dic.keys():
#         html = html.replace(key,html_dic[key])
#     return "$$"+html.strip()+"$$"


# def extra_pure_text(title_text:str,
#                     abs_text:str):
#     """Extract pure text of title and abstract."""
#     pattern = r">([^<]+)<"
#     title = re.findall(pattern,title_text)
#     abstract = re.findall(pattern,abs_text)
#     print('title:',title)
#     title = html2md(''.join(title))
#     abstract = html2md(''.join(abstract))
#     return title,abstract




def get_daily_links(proxies = None,
                    max_num = 10,
                    show_meta_data = False,
                    line_length :int = 15,
                    daily_type:str = 'cs',
                    header = None,
                    max_retry = 3,
                    wait_fixed = 1000):
    """Get the links to the daily new papers."""
    type_l = ['cs','math','physics','q-bio','q-fin','stat','eess','econ','astro-ph','cond-mat','gr-qc','hep-ex','hep-lat','hep-ph','hep-th','math-ph','nucl-ex','nucl-th','quant-ph']
    assert daily_type in type_l, f"daily_type:{daily_type} must be one of {type_l}"
    new_papers_url = "https://arxiv.org/list/"+daily_type+"/new"
    logging.info(f"fetching {new_papers_url}")
    #-----------------get response -----------------
    # new_papers_url = "http://xxx.itp.ac.cn/list/astro-ph.CO/new"
    # for i in range(max_retry):
    #     try:
    #         response = requests.get(new_papers_url,proxies = proxies,headers = header)
    #         break
    #     except ProxyError:
    #         logging.warning(f'ProxyError happened. This is attempt #{i + 1}.')
    #         if i < max_retry - 1:  # i is zero indexed
    #             time.sleep(1)  # wait a bit before trying again
    #             continue
    #         else:
    #             logging.error(f'All {max_retry} retries failed.')
    #----------------------------------------------
    @retry(stop_max_attempt_number=max_retry, wait_fixed=wait_fixed)
    def _get_response(url: str, proxies=None, header=None):
        """Get response."""
        response = requests.get(url, proxies=proxies, headers=header)
        if response.status_code != 200:
            raise Exception(f'HTTP error, status = {response.status_code}')
        return response

    try:
        response = _get_response(new_papers_url,proxies = proxies,header = header)
    except Exception as e:
        logging.error(f'All {max_retry} retries failed, the error message is: {str(e)}')
        raise e


    # parse html
    soup = BeautifulSoup(response.content, "html.parser")

    # Links are all in <dl> tag after New submissions
    cross_lists_tag = soup.find("h3", string=re.compile("New submissions"))
    # Get next tag
    cross_lists = cross_lists_tag.find_next("dl")
    links_containers = cross_lists.find_all("span", attrs={"class": "list-identifier"},limit=2*max_num)
    parsed_links = []
    for i,container in enumerate(links_containers):
        link = container.find("a", attrs={"title": "Download PDF"})
        if link is not None:
            parsed_links.append("https://arxiv.org" + link["href"])
        else:
            logging.warning(f'No link found in article {i}')
            parsed_links.append(None)
    logging.info(f'links_containers:{len(links_containers)},len(links):{len(parsed_links)},None in links:{parsed_links.count(None)}')
    # links = cross_lists.find_all("a", attrs={"title": "Download PDF"})
    # parsed_links = ["https://arxiv.org" + link["href"] for link in links]
    parsed_abs, parsed_title, parsed_authors = [], [], []

    # need to parse meta data
    if show_meta_data:
        meta_data = cross_lists.find_all("div", attrs={"class": "meta"},limit=2*max_num)
        for i,meta_info in enumerate(meta_data):
            # get raw title and abstract(html format) from meta_info
            title_text = str(meta_info.find("div", attrs={"class": "list-title mathjax"}))
            abs_text = str(meta_info.find("p", attrs={"class": "mathjax"}))
            authors = str(meta_info.find("div", attrs={"class": "list-authors"}))
            title,abstract,authors = extra_pure_text(title_text,abs_text,authors,line_length=line_length)
            parsed_title.append(title)
            parsed_abs.append(abstract)
            parsed_authors.append(authors)
        links,parsed_title,parsed_abs,parsed_authors = remove_duplicates([parsed_links,parsed_title,parsed_abs,parsed_authors])
        logging.info(f"Found {len(links)} links(drop Na/duplicate num:{len(parsed_links)-len(links)}), {len(parsed_title)} titles, {len(parsed_abs)} abstracts and {len(parsed_authors)} authors today.")
        parsed_title = ['['+title+']('+links[i]+')' for i,title in enumerate(parsed_title)]
    else:
        links = remove_duplicates(parsed_links)

    if max_num < len(links):
        print(f"Found {len(links)} new papers today,extracting {max_num} papers.")
        links = links[:max_num]
    else:
        logging.info(f"max_num:{max_num}, but only found {len(links)} new papers today.")
    if show_meta_data:
        return links,parsed_title[:len(links)],parsed_abs[:len(links)],parsed_authors[:len(links)]

    # return links,None,None,just keep the same format
    return links,parsed_title,parsed_abs,parsed_authors

def get_keyword_links(key_word:str,
                      proxies = None,
                      max_num = 10,
                      line_length :int = 15,
                      searchtype = 'all',
                      abstracts = 'show',
                      order = '-announced_date_first',
                      size = 50,
                      show_meta_data = False,
                      header = None,
                      max_retry = 3,
                      wait_fixed = 1000):
    assert searchtype in ['all','title','abstract','author','comment',
                          'journal_ref','subject_class','report_num','id_list'],\
        f"searchtype:{searchtype} does not in one of ['all','title','abstract','author'," \
        f"'comment','journal_ref','subject_class','report_num','id_list']"
    assert abstracts in ['show','hide'],f"abstracts:{abstracts} does not in one of ['show','hide']"
    assert order in ['-announced_date_first','submitted_date',
                     '-submitted_date','announced_date_first',''] ,\
        f"order:{order} does not in one of ['-announced_date_first'," \
        f"'submitted_date','-submitted_date','announced_date_first','']"
    assert size in [25,50,100,200],f"size:{size} does not in one of [25,50,100,200]"

    # transfer key_word to url format
    if key_word.startswith('http'):
        url = key_word
    else:
        key_word = key_word.replace(' ','+')
        url = f'https://arxiv.org/search/?query={key_word}&searchtype={searchtype}' \
              f'&abstracts={abstracts}&order={order}&size={size}'
    logging.info(f"fetching {url}\nkew_word:{key_word}")

    #-----------------get response -----------------
    # url = f'http://xxx.itp.ac.cn/search/?query={key_word}&searchtype={searchtype}' \
    #       f'&abstracts={abstracts}&order={order}&size={size}'
    # for i in range(max_retry):
    #     try:
    #         respon = requests.get(url, proxies=proxies, headers=header)
    #         break
    #     except ProxyError:
    #         logging.warning(f'ProxyError happened. This is attempt #{i + 1}.')
    #         if i < max_retry - 1:
    #             time.sleep(1)
    #             continue
    #         else:
    #             logging.error(f'All {max_retry} retries failed.')
    #----------------------------------------------
    @retry(stop_max_attempt_number=max_retry, wait_fixed=wait_fixed)
    def _get_response(url: str, proxies=None, header=None):
        """Get response."""
        response = requests.get(url, proxies=proxies, headers=header)
        if response.status_code != 200:
            raise Exception(f'HTTP error, status = {response.status_code}')
        return response

    try:
        respon = _get_response(url,proxies = proxies,header = header)
    except Exception as e:
        logging.error(f'All {max_retry} retries failed, the error message is: {str(e)}')
        raise e


    soup = BeautifulSoup(respon.text, 'html.parser')
    # get the information of the papers
    content = soup.find_all('p', {'class': 'list-title is-inline-block'},limit=2*max_num)
    links = []
    for i in range(len(content)):
        link = content[i].find_all('a',string=re.compile("pdf"))
        if link:
            links.append(link[0]['href'])
        else:
            logging.warning(f"Could not find pdf link for {content[i].find('a')['href']}")
            links.append(None)
    parsed_abs,parsed_title,parsed_author = [],[],[]
    if show_meta_data:
        title_text = soup.find_all("p", attrs={"class": "title is-5 mathjax"},limit=2*max_num)
        abs_text = soup.find_all("p", attrs={"class": "abstract mathjax"},limit=2*max_num)
        author_text = soup.find_all("p", attrs={"class": "authors"},limit=2*max_num)
        links,title_text,abs_text,author_text = remove_duplicates([links,title_text,abs_text,author_text])
        for i,(title_t,abs_t,author_t) in enumerate(zip(title_text,abs_text,author_text)):
            title_t , abs_t , author_t = str(title_t),str(abs_t),str(author_t)
            title,abstract,author = extra_pure_text(title_t,abs_t,author_t,line_length=line_length)
            print('author:',author)
            title = '['+title+']('+links[i]+')'
            parsed_title.append(title)
            parsed_abs.append(abstract)
            parsed_author.append(author)
    else:
        links = remove_duplicates(links)
    if len(links) < max_num:
        print(f'found {len(links)} links in {key_word},which is'
              f'smaller than max_num:{max_num},return all links({len(links)})')
    else:
        print(f'found {len(links)} links in {key_word},which is '
              f'larger than max_num:{max_num},return {max_num} links')
        links = links[:max_num]

    if show_meta_data:
        return links,parsed_title[:len(links)],parsed_abs[:len(links)],parsed_author[:len(links)]
    # return links,None,None,just keep the same format
    return links,parsed_title,parsed_abs,parsed_author






def get_arxiv_links(key_word = None,
                    proxies = None,
                    max_num = 10,
                    line_length = 15,
                    searchtype = 'all',
                    abstracts = 'show',
                    order = '-announced_date_first',
                    size = 50,
                    show_meta_data = True,
                    daily_type = "cs",
                    headers = None,
                    max_retry = 3,
                    wait_fixed = 1000):
    """Get the links to the PDFs of the daily new papers. Stop parsing once we see <h3>Cross-lists"""

    # if proxies is None:
    #     ps = ProxyServer()
    #     proxies = ps.get_default_proxy_info()
    if key_word is None:
        links,titles,abs,authors = get_daily_links(proxies = proxies,
                                                   daily_type=daily_type,
                                                   max_retry=max_retry,
                                                   max_num = max_num,
                                                   show_meta_data = show_meta_data,
                                                   line_length=line_length,
                                                   header=headers,
                                                   wait_fixed=wait_fixed)

    else:
        links,titles,abs,authors = get_keyword_links(
            key_word = key_word,
            proxies = proxies,
            max_num = max_num,
            searchtype = searchtype,
            abstracts = abstracts,
            order = order,
            size = size,
            show_meta_data = show_meta_data,
            line_length=line_length,
            header=headers,
            max_retry=max_retry,
            wait_fixed=wait_fixed
        )
    logging.info(f"found {len(links)} links in total,links:{links}")
    return links,titles,abs,authors







# 处理代理服务器

# class ProxyServer:
#
#     def __init__(self):
#         self.__path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
#         self.__INTERNET_SETTINGS = winreg.OpenKeyEx(winreg.HKEY_CURRENT_USER,
#                                                     self.__path, 0, winreg.KEY_ALL_ACCESS)
#
#     def get_default_proxy_info(self):
#         """get proxy server from Windows"""
#         ip, port = "", ""
#         if self.is_open_proxy_form_Win():
#             try:
#                 ip, port = winreg.QueryValueEx(self.__INTERNET_SETTINGS, "ProxyServer")[0].split(":")
#                 print("get proxy information：{}:{}".format(ip, port))
#             except FileNotFoundError as err:
#                 print("no proxy information：" + str(err))
#             except Exception as err:
#                 print("other error：" + str(err))
#
#             return {"http": "http://{}:{}".format(ip, port),
#                     "https": "http://{}:{}".format(ip, port),
#                     "ftp": "ftp://{}:{}".format(ip, port),
#                     "host": "http://localhost:{}".format(port)}
#         else:
#             print("system does not open proxy")
#             return {"http": None,
#                     "https": None,
#                     "ftp": None,
#                     "host": None}
#
#     def is_open_proxy_form_Win(self):
#         """if open proxy"""
#         try:
#             if winreg.QueryValueEx(self.__INTERNET_SETTINGS, "ProxyEnable")[0] == 1:
#                 return True
#         except FileNotFoundError as err:
#             print("no proxy information：" + str(err))
#         except Exception as err:
#             print("other error：" + str(err))
#         return False

if __name__ == '__main__':

    proxies = {'http': 'http://127.0.0.1:7890',
           'https': 'http://127.0.0.1:7890', 'ftp': 'ftp://127.0.0.1:7890'}

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"}
    links,title,abs,authors = get_arxiv_links(key_word = None,proxies = proxies,headers=headers,max_num = 10,show_meta_data = True)
    print('links:','\n',links)
    print('title:','\n',title)
    print('abs:','\n',abs)
    print('authors:','\n',authors)
    url = "https://baidu.com"

    response = requests.get(url,proxies=proxy)
    print(response.status_code)

