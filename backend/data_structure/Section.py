import logging
import reprlib
from typing import List, Union
import tiktoken
import re

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-16k-0613", detailed_img: bool = False):
    """
    Returns the number of tokens used by a list of messages.

    Args:
        messages: A list of messages. Each message can be:
            - dict: With keys "role" and "content".
            - string: In which case the role is assumed to be "user".
            - list: Already normalized into a list of dicts.
        model: The model to use. Defaults to "gpt-3.5-turbo-16k-0613".
        detailed_img: Bool, whether to use detailed image token count.

    Returns:
        int: The number of tokens used by the messages.
    """
    # Map of model to token encoding scheme
    model_map = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "text-embedding-ada": "cl100k_base",
        "text-davinci": "p50k_base",
        "Codex": "p50k_base",
        "davinci": "p50k_base",
    }

    encode_model = None
    for key, value in model_map.items():
        if key in model:
            encode_model = value
            break

    if not encode_model:
        logging.error(f"Model {model} not found. Loading default model: cl100k_base")
        encode_model = "cl100k_base"

    encoding = tiktoken.get_encoding(encode_model)

    if isinstance(messages, dict):
        messages = [messages]
    elif isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    num_tokens = 0

    if "gpt-3.5-turbo" in model:  # Model-specific tokenization
        for message in messages:
            num_tokens += 4  # Every message includes start and end tokens
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # Omit role token if "name" is present
                    num_tokens -= 1
            num_tokens += 2  # Additional tokens for assistant's reply
    elif "gpt-4" in model:
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                else:  # Handle lists of dicts in the content
                    for item in value:
                        for sub_key, sub_value in item.items():
                            if isinstance(sub_value, str):
                                num_tokens += len(encoding.encode(sub_value))
                            else:
                                num_tokens += 129 if detailed_img else 65
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")

    return num_tokens


class SectionText:
    def __init__(self, text: str):
        self._text = text

    def __str__(self):
        return self._text

    def __repr__(self):
        return f"subtext({reprlib.repr(self._text)})"

    def __len__(self):
        return num_tokens_from_messages(self._text)

    @property
    def text(self):
        return self._text


class SectionTitle:
    def __init__(self, title: str):
        self._title = title
        self._grained_level = None
        self._set_grained_level()

    def __str__(self):
        return self._title

    def __repr__(self):
        return f"subtitle({self._title})"

    @property
    def title(self):
        return self._title

    @property
    def grained_level(self):
        return self._grained_level

    def _set_grained_level(self):
        """
        Set the grained level based on the number of '#' symbols in the title.
        """
        count_hashes = self._title.split(' ')[0].count('#')
        if count_hashes != 0:
            self._grained_level = count_hashes
        else:
            logging.error(f'No # found in subtitle: {self._title}, parser error')

    def __len__(self):
        return num_tokens_from_messages(self._title)


class Section:

    def __init__(self,
                 title: str,
                 text: str,
                 parent:str = '',
                 summary:str = '',
                 img_paths:Union[List[str],str] = None,
                 children_list:Union[List['Section'],'Section']= None
                 ):
        """
        Initializes a Section object with a title, text, optional parent, summary, image paths, and child sections.

        :param title: The title of the section.
        :param text: The following text of the section without text from child sections.
        :param parent: The title of the parent section.
        :param summary: Summary of the section.
        :param img_paths: Paths to images associated with the section.
        :param children_list: Child sections of this section.
        """
        self._title = SectionTitle(title)
        self._following_text = SectionText(text)
        self._grained_level = self._title.grained_level
        self.children_list = children_list if isinstance(children_list,List) else [children_list] if children_list else []
        self._set_text()
        self._title_text =  self._title.title + '\n' + self._text.text
        self._set_length()
        self._set_section_titles()
        self.summary = summary
        self._img_paths = img_paths if isinstance(img_paths,List) else [img_paths] if img_paths else []
        self._parent = parent

    def get_self_title_text(self):
        """
        Generates the string representation of the section including its title and text.
        """
        return str(self._title) + "\n" + str(self._following_text)

    def get_children_title_text(self):
        """
        Concatenates the string representations of all child sections.
        """
        if self.children_list:
            return "\n".join([child.get_self_title_text() for child in self.children_list])
        else:
            return ""

    def _set_text(self):
        """
        Combines the immediate following text with the concatenated text of all child sections.
        """
        if not self.children_list:
            self._text = self._following_text
        else:
            self._text = SectionText(self._following_text.text + "\n" + self.get_children_title_text())

    def _set_section_titles(self):
        """
        Creates a list of all titles including this section's title and its children's titles.
        """
        if self.children_list:
            self._section_titles = [self._title.title] + [child.title for child in self.children_list]
        else:
            self._section_titles = [self._title.title]

    def add_img_paths(self, img_paths: Union[List[str], str]):
        """
        Adds additional image paths to the existing list of image paths.

        :param img_paths: New image paths to be added.
        """
        if isinstance(img_paths, list):
            self._img_paths.extend(img_paths)
        else:
            self._img_paths.append(img_paths)


    def _set_length(self):
        """
        Calculates the total number of tokens in the title and text combined.
        """
        self._length = num_tokens_from_messages(self._title_text)

    @property
    def grained_level(self):
        return self._grained_level

    @property
    def title(self):
        return self._title.title

    @property
    def text(self):
        return self._text.text

    @property
    def img_paths(self):
        return self._img_paths

    @property
    def title_text(self):
        return self._title_text

    @property
    def following_text(self):
        return self._following_text

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self,parent):
        self._parent = parent

    @property
    def length(self):
        return self._length

    @property
    def section_titles(self):
        return self._section_titles

    def __repr__(self):
        """
        Returns a developer-friendly string representation of the section.
        """
        return (f"Section(title={reprlib.repr(self._title.title)}, "
                f"text={reprlib.repr(self._text.text)}, "
                f"children_list={self.children_list}, "
                f"parent={self.parent}, "
                f"summary={reprlib.repr(self.summary)}, "
                f"img_paths={reprlib.repr(self._img_paths)})")


