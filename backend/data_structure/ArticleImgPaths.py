from typing import List, Union
import reprlib
import re
from PIL import Image
from typing import List, Union
from MMAPIS.backend.tools.utils import strip_title

class SectionIMGData():
    def __init__(self,
                 img_data:List[Image.Image],
                 section_name):
        self._img_data = img_data if isinstance(img_data, List) else [img_data]
        self._section_name = section_name

    def __repr__(self):
        return f"SectionIMGData(section_name={self._section_name}, img_data={reprlib.repr(self._img_data)})"

    def add_img(self, img:Union[Image.Image, List[Image.Image]]):
        if isinstance(img, List):
            self._img_data.extend(img)
        else:
            self._img_data.append(img)

    @property
    def img_data(self):
        return self._img_data

    @property
    def section_name(self):
        return self._section_name

class SectionIMGPaths():
    def __init__(self,img_path:Union[str,List[str]],
                 section_name:str,
                 parent:str=''):
        self._img_path = img_path if isinstance(img_path, List) else [img_path]
        self._section_name = section_name
        self.parent = parent

    def __repr__(self):
        return f"SectionIMGPaths(section_name={self._section_name}, parent={self.parent}, img_path={reprlib.repr(self._img_path)})"

    def add_path(self, path:Union[str,List[str]]):
        if isinstance(path, List):
            self._img_path.extend(path)
        else:
            self._img_path.append(path)

    @property
    def img_path(self):
        return self._img_path

    @property
    def section_name(self):
        return self._section_name

    def __len__(self):
        return len(self._img_path)



class ArticleIMGPaths():
    def __init__(self,image_paths:List[SectionIMGPaths]):
        self._image_paths = image_paths

    def __repr__(self):
        return f"ArticleIMGPaths({reprlib.repr(self._image_paths)})"

    def update_image_paths(self, article):
        for section_img_paths in self._image_paths:
            img_section_name = section_img_paths.section_name
            if img_section_name.startswith("unknown_section_"):
                continue
            flag = False
            for i, section in enumerate(article.sections):
                section_names = section.section_titles
                for section_name in section_names:
                    if section_name.startswith("# "):
                        continue
                    section_name = strip_title(section_name)
                    if img_section_name.lower() in section_name.lower():
                        flag = True
                        section_img_paths.parent = section.parent
                        break
                if flag:
                    break

    def __getitem__(self, section_name):
        return self._image_paths[section_name]

    def __iter__(self):
        return iter(self._image_paths)

    def __len__(self):
        return len(self._image_paths)


