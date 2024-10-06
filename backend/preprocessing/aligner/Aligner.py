from MMAPIS.backend.data_structure import Article, ArticleIMGPaths,SectionIMGPaths
import spacy
from typing import List, Union


class Aligner:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')

    def img_ls_to_md(self,
                     img_ls: Union[List[str], str],
                     img_width: int = 400,
                     margin: int = 10) -> str:
        if isinstance(img_ls, str):
            img_ls = [img_ls]
        if not img_ls:
            return ""

        # # A version that can be used in the markdown but cannot correctly display in html
        # prefix = f'<div style="display: flex; justify-content: center; overflow-x: auto; padding: 10px; background: #f0f0f0; border-radius: 8px;">\n  <div style="display: flex; min-width: min-content;">\n'
        # img_template = f'    <div style="flex: 0 0 auto; margin-right: {margin}px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">\n' \
        #                f'      <img src="{{}}" alt="Image 0" style="height: {img_width}px; width: auto; object-fit: contain; border-top-left-radius: 8px; border-top-right-radius: 8px;">\n' \
        #                f'      <div style="padding: 10px; text-align: center; font-weight: bold; color: #333;">img{{}}</div>\n' \
        #                f'    </div>\n'
        # suffix = "  </div>\n</div>"

        # A version that can be both used in markdown and correctly display in html, but when few images are displayed, the images are not centered
        prefix = f'<div style="width: 100%; overflow-x: auto; white-space: nowrap; padding: 20px 0;">\n'
        img_template = f'    <div style="display: inline-block; margin-right: {margin}px; text-align: center;">\n' \
                       f'      <img src="{{}}" alt="Image {{}}" style="max-height: {img_width}px; width: auto; vertical-align: top;">\n' \
                       f'      <div style="margin-top: 10px; font-weight: bold;">img{{}}</div>\n' \
                       f'    </div>\n'
        suffix = "</div>"
        img_md = ''.join([img_template.format(img, idx,idx) for idx, img in enumerate(img_ls)])
        return f"{prefix}{img_md}{suffix}"

    def align_raw_md_text(self, raw_md_text: str, img_paths: List[SectionIMGPaths], min_grained_level: int = 3, max_grained_level: int = 4) -> ArticleIMGPaths:
        article = Article(article=raw_md_text, min_grained_level=min_grained_level, max_grained_level=max_grained_level)
        article_img_paths = ArticleIMGPaths(image_paths=img_paths)
        article_img_paths.update_image_paths(article)
        return article_img_paths

    def generate_markdown_res(self,article, img_width,margin):
        return article.extra_info + "\n" + self.img_ls_to_md(article.fail_match_img_paths)+'\n\n' + '\n'.join(
            [section.title + '\n' + self.img_ls_to_md(section.img_paths, img_width=img_width,
                                                                        margin=margin)
             + '\n' + section.text for section in article.sections])

    def align(self,
              text: str,
              img_paths: List[SectionIMGPaths],
              raw_md_text: str = None,
              min_grained_level: int = 3,
              max_grained_level: int = 4,
              img_width: int = 400,
              margin: int = 10,
              threshold: float = 0.9,
              ) -> str:
        article = Article(article=text, file_name="pdf", min_grained_level=min_grained_level, max_grained_level=max_grained_level)
        align_raw_md_text = False
        if raw_md_text:
            img_paths = self.align_raw_md_text(raw_md_text, img_paths, min_grained_level, max_grained_level)
            align_raw_md_text = True
        article.assign_img2section(img_paths, self.nlp, threshold,align_raw_md_text=align_raw_md_text)
        return self.generate_markdown_res(article=article,img_width=img_width,margin=margin)

