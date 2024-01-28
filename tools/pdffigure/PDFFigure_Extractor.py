from MMAPIS.tools.pdffigure import PDFFigure2PaperParser
from MMAPIS.tools.pdffigure.config import SNAP_WITH_CAPTION
from MMAPIS.tools.utils import strip_title
import reprlib
from itertools import chain
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
from MMAPIS.tools.pdffigure.pdf_extract import get_objpixmap
from typing import Dict, List, Union

class SectionFigures():
    def __init__(self,
                 img_data:List[Image.Image],
                 section_name,
                 parent=''):
        self._img_data = img_data if isinstance(img_data, List) else [img_data]
        self._section_name = section_name
        self.parent = parent

    def __repr__(self):
        return f"PDFFigure(section_name={self._section_name}, parent={self.parent}, img_data={reprlib.repr(self._img_data)})"

    def add_img(self, img):
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
        return f"FigurePath(section_name={self._section_name}, parent={self.parent}, img_path={reprlib.repr(self._img_path)})"

    def add_path(self, path):
        self._img_path.append(path)

    @property
    def img_path(self):
        return self._img_path

    @property
    def section_name(self):
        return self._section_name

    def __len__(self):
        return len(self._img_path)





class PDFFigureExtractor:
    def __init__(self,pdf_path):
        """
        Initialize the PDFFigure helper.
        """
        # Initialization, if needed
        self.pdf_path = pdf_path
        self.pdf_parser = PDFFigure2PaperParser(pdf_path)


    def extract_figures_and_tables(self, snap_with_caption=SNAP_WITH_CAPTION, verbose=False, get_tmpfile_dir: bool = False)->List[SectionFigures]:
        """
        Extract figures and tables from a PDF file and identify their section names.

        :param pdf_path: Path to the PDF file.
        :return: A list of tuples, each containing a figure/table and its section name.
        """
        # Implement the logic to extract figures and tables
        # and to associate them with their respective section names

        # Example (pseudocode):
        # content = pdffigure_tool.read_pdf(pdf_path)
        # figures_tables = pdffig

        section_names = self.pdf_parser.get_section_titles()
        img_dict = self.pdf_parser.get_section_imagedict(
            snap_with_caption=snap_with_caption, verbose=verbose)
        res_img_ls = []
        for name in section_names[:-1]:
            img_ls = img_dict.get(name)
            name = strip_title(name)
            res_img_ls.append(SectionFigures(img_data=[], section_name=name))
            if img_ls:
                for img in img_ls:
                    img_data = get_objpixmap(self.pdf_parser.pdf, img, get_tmpfile_dir=get_tmpfile_dir)
                    img_data = self.pixmap_to_img(img_data)
                    res_img_ls[-1].add_img(img_data)
        return res_img_ls

    @staticmethod
    def pixmap_to_img(pixmap)->Image.Image:
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return image

    def extract_save_figures(self, save_dir: str = None, header: bool = False) -> List[SectionIMGPaths]:
        img_ls = self.extract_figures_and_tables()
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(self.pdf_path), "img")
        else:
            save_dir = os.path.join(save_dir, "img")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if img_ls:
            if header:
                img_paths = [SectionIMGPaths(img_path=[], section_name="header")]
                img_ls = chain.from_iterable([section_imgs.img_data for section_imgs in img_ls])
                for i, img in enumerate(img_ls):
                    img_path = os.path.join(save_dir, f"{i}.png")
                    img.save(img_path)
                    img_paths[-1].add_path(path=os.path.join("img", f"{i}.png"))
                logging.info(f"parser success: {len(img_paths[-1])} img found in {self.pdf_path}")
                return img_paths
            else:
                img_paths = []
                for section_imgs in img_ls:
                    section_name = strip_title(section_imgs.section_name)
                    img_paths.append(SectionIMGPaths(img_path=[], section_name=section_name))
                    for i,img in enumerate(section_imgs.img_data):
                        img_path = os.path.join(save_dir, f"{section_name}_{i}.png")
                        img.save(img_path)
                        img_paths[-1].add_path(path=os.path.join("img", f"{section_name}_{i}.png"))

                logging.info(f"parser success: {sum(map(len,img_paths))} img found in {self.pdf_path}")
                return img_paths

        else:
            logging.error(f"parser error: no img found in {self.pdf_path}")
            return {}

# Example usage
if __name__ == "__main__":
    pdf_path = './Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper - 副本.pdf'
    pdf_parser = PDFFigureExtractor(pdf_path)
    x = pdf_parser.extract_figures_and_tables()
    print("img_dict:",x)
    for section_imgs in x:
        for img in section_imgs.img_data:
            plt.imshow(img)
            plt.text(0, 0, section_imgs.section_name)
            plt.show()

    paths = pdf_parser.extract_save_figures()
    for path in paths:
        print(path)



