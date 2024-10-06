from MMAPIS.backend.preprocessing.pdffigure import PDFFigure2PaperParser
from MMAPIS.backend.preprocessing.pdffigure.config import SNAP_WITH_CAPTION
from MMAPIS.backend.tools.utils import strip_title
import reprlib
from itertools import chain
import os
import logging
from PIL import Image
import matplotlib.pyplot as plt
from MMAPIS.backend.preprocessing.pdffigure.pdf_extract import get_objpixmap
from typing import Dict, List, Union
from MMAPIS.backend.data_structure.ArticleImgPaths import SectionIMGData, SectionIMGPaths

class PDFFigureExtractor:
    def __init__(self,pdf_path):
        """
        Initializes the PDFFigureExtractor with the specified PDF file path.

        :param pdf_path: The path to the PDF file from which figures and tables will be extracted.
        """
        self.pdf_path = pdf_path
        self.pdf_parser = PDFFigure2PaperParser(pdf_path)


    def extract_figures_and_tables(self, snap_with_caption=SNAP_WITH_CAPTION, verbose=False, get_tmpfile_dir: bool = False)->List[SectionIMGData]:
        """
        Extracts figures and tables from the PDF file and associates them with their respective section names.

        :param snap_with_caption: Determines whether to include captions in the snapshots.
        :param verbose: If True, provides detailed logging of the extraction process.
        :param get_tmpfile_dir: If True, temporary directories are generated during the extraction.
        :return: A list of SectionIMGData objects, each containing the extracted images and associated section name.
        """
        section_names = self.pdf_parser.get_section_titles()
        img_dict = self.pdf_parser.get_section_imagedict(snap_with_caption=snap_with_caption, verbose=verbose)
        img_data_ls = []

        for i, name in enumerate(section_names[:-1]):
            img_ls = img_dict.get(name)
            if img_ls:
                # Sanitize the section name
                name = strip_title(name)
                if not name:
                    name = f"unknown_section_{i}"

                # Initialize a new SectionIMGData object
                section_data = SectionIMGData(img_data=[], section_name=name)
                img_data_ls.append(section_data)

                # Convert pixmap images to PIL images and add them to the section
                for img in img_ls:
                    img_data = get_objpixmap(self.pdf_parser.pdf, img, get_tmpfile_dir=get_tmpfile_dir)
                    img_data = self.pixmap_to_img(img_data)
                    section_data.add_img(img_data)

        return img_data_ls


    @staticmethod
    def pixmap_to_img(pixmap) -> Image.Image:
        """
        Converts a pixmap object to a PIL Image.

        :param pixmap: The pixmap object to be converted.
        :return: A PIL Image object.
        """
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


    def extract_save_figures(self, save_dir: str = None, header: bool = False) -> List[SectionIMGPaths]:
        """
        Extracts figures and tables from the PDF file, saves them to the specified directory,
        and associates the saved image paths with their respective section names.

        :param save_dir: The directory where the images will be saved. If not specified, defaults to a subdirectory in the PDF's location.
        :param header: If True, all images are saved in the same directory under a single "header" section.
        :return: A list of SectionIMGPaths objects, each containing the paths to the saved images and associated section name.
        """
        img_data_ls = self.extract_figures_and_tables()
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(self.pdf_path), "img")
        else:
            save_dir = os.path.join(save_dir, "img")

        # Create the directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        if img_data_ls:
            if header:
                # Save all images under a single "header" section
                return self._save_images_as_header(img_data_ls, save_dir)
            else:
                # Save images in individual sections
                return self._save_images_by_section(img_data_ls, save_dir)
        else:
            logging.error(f"Parser error: No images found in {self.pdf_path}")
            return []


    def _save_images_as_header(self, img_data_ls: List[SectionIMGData], save_dir: str) -> List[SectionIMGPaths]:
        """
        Saves all images under a single "header" section.

        :param img_data_ls: A list of SectionIMGData objects containing the images to be saved.
        :param save_dir: The directory where the images will be saved.
        :return: A list of SectionIMGPaths objects containing the paths to the saved images.
        """
        img_paths = [SectionIMGPaths(img_path=[], section_name="header")]
        img_ls = chain.from_iterable(section.img_data for section in img_data_ls)

        for i, img in enumerate(img_ls):
            img_path = os.path.join(save_dir, f"{i}.png")
            img.save(img_path)
            img_paths[-1].add_path(path=os.path.abspath(img_path))

        logging.info(f"Parser success: {len(img_paths[-1])} images found in {self.pdf_path}")
        return img_paths

    def _save_images_by_section(self, img_data_ls: List[SectionIMGData], save_dir: str) -> List[SectionIMGPaths]:
        """
        Saves images in individual sections, each associated with its respective section name.

        :param img_data_ls: A list of SectionIMGData objects containing the images to be saved.
        :param save_dir: The directory where the images will be saved.
        :return: A list of SectionIMGPaths objects containing the paths to the saved images.
        """
        img_paths = []

        for section_imgs in img_data_ls:
            section_name = section_imgs.section_name
            section_paths = SectionIMGPaths(img_path=[], section_name=section_name)
            img_paths.append(section_paths)

            for i, img in enumerate(section_imgs.img_data):
                img_path = os.path.join(save_dir, f"{section_name}_{i}.png")
                img.save(img_path)
                section_paths.add_path(path=os.path.abspath(img_path))

        logging.info(f"Parser success: {sum(map(len, img_paths))} images found in {self.pdf_path}")
        return img_paths
