import os
import sys
from MMAPIS.config.config import NOUGAT_CONFIG, GENERAL_CONFIG
from MMAPIS.tools.utils import get_batch_size,torch_gc
import urllib.error
from pathlib import Path
import logging
import re
from functools import partial
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from MMAPIS.tools.nougat.nougat_main.nougat import NougatModel
from MMAPIS.tools.nougat.nougat_main.nougat.utils.dataset import LazyDataset
from MMAPIS.tools.nougat.nougat_main.nougat.postprocessing import markdown_compatible
import fitz
import torch
from typing import List,Union,Dict,Tuple
import reprlib
from datetime import datetime
from torch import profiler
from subprocess import Popen, PIPE


from MMAPIS.tools import ArxivCrawler



class NougatArticle:
    def __init__(self,
                 file_name: str,
                 content: str,
                 ):
        self._file_name = file_name
        self._content = content

    def __repr__(self):
        return f"NougatArticle(file_name={self._file_name}, content={self._content})"

    def __str__(self):
        return f"NougatArticle:file_name={self._file_name}, content={reprlib.repr(self._content)})"

    @property
    def file_name(self):
        return self._file_name

    @property
    def content(self):
        return self._content

class NougatPredictor:
    def __init__(self,
                 checkpoint: str = NOUGAT_CONFIG["check_point"],
                 recompute: bool = NOUGAT_CONFIG["recompute"],
                 proxy: Dict = GENERAL_CONFIG["proxy"],
                 headers: Dict = GENERAL_CONFIG["headers"],
                 out: Path = GENERAL_CONFIG["save_dir"],
                 markdown: bool = NOUGAT_CONFIG["markdown"],
                 batch_size: int = get_batch_size()):
        self.checkpoint = checkpoint
        self.recompute = recompute
        self.proxy = proxy
        self.headers = headers
        self.out = Path(out)
        self.markdown = markdown
        self.batch_size = batch_size

    def __repr__(self):
        return f"NougatPredictor(checkpoint={self.checkpoint}, recompute={self.recompute}, proxy={self.proxy}, headers={self.headers}, out={self.out}, markdown={self.markdown}, batch_size={self.batch_size})"

    def __str__(self):
        return f"NougatPredictor: checkpoint={self.checkpoint}, recompute={self.recompute}, proxy={reprlib.repr(self.proxy)}, headers={reprlib.repr(self.headers)}, out={os.path.abspath(self.out)}, markdown={self.markdown}, batch_size={self.batch_size}"

    def pdf2md_text(self,
                    pdfs: Union[Path,bytes,str,List[Union[Path, str, bytes]]],
                    pdf_names: Union[str, List[str]] = None) -> List[NougatArticle]:
        """
        Convert pdf to markdown text
        :param pdf: pdf path or pdf links or bytes of pdf data
        :param pdf_names: pdf names
        :return: List of markdown text, List of pdf names
        """
        if not isinstance(pdfs, List):
            pdfs = [pdfs]
        if pdf_names is None:
            pdf_names = [None] * len(pdfs)
        if not isinstance(pdf_names, List):
            pdf_names = [pdf_names]
        assert len(pdfs) == len(pdf_names), f"Length of pdf and pdf_names should be the same, got {len(pdfs)} and {len(pdf_names)}"
        model = NougatModel.from_pretrained(self.checkpoint,ignore_mismatched_sizes=True).to(torch.bfloat16)
        model = self.move_to_device(model,cuda=self.batch_size > 0)
        model.eval()
        datasets = []
        pdf_process_bar = tqdm(pdfs)
        for i,(pdf,pdf_name) in enumerate(zip(pdf_process_bar,pdf_names)):
            pdf_name = pdf_name if pdf_name else self._get_pdf_name(pdf)
            pdf_names[i] = pdf_name
            pdf_process_bar.set_description(f"[Nougat] Loading {pdf_name} into memory")
            if isinstance(pdf,Path):
                if not pdf.exists():
                    print(f"File {pdf} does not exist.")
                    continue
                if self.out:
                    out_path = self.out /pdf_name / pdf.with_suffix(".md").name
                    # if exists and not recompute, skip
                    if out_path.exists() and not self.recompute:
                        logging.info(
                            f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                        )
                        continue
            try:

                dataset = LazyDataset(
                    pdf, partial(model.encoder.prepare_input, random_padding=False),proxy=self.proxy,headers=self.headers,pdf_name= pdf_name
                )
                logging.info(f"Loaded {str(pdf) if not isinstance(pdf,bytes) else pdf_name},length: {len(dataset)}")
            # if pdf is corrupted, skip
            except fitz.FileDataError:
                logging.info(f"Could not load file in path {str(pdf)}.")
                continue
            except urllib.error.URLError:
                logging.info(f"Could not load file in URL {str(pdf)}.")
                continue
            datasets.append(dataset)
        if len(datasets) == 0:
            logging.error("No PDFs to process.")
            return []
        # shape
        # datasets: [num_files*num_pages]
        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        predictions = []
        file_index = 0
        page_num = 0
        # [sample(img) , is_last_page] is data from a pdf file (1 page)
        # sample: [batch_size,num_channels(3),h,w],is_last_page = [batch_size] in ['' or pdf_name]
        files = []
        dataloader_bar = tqdm(dataloader,leave=True,position=0)
        for i, items in enumerate(dataloader_bar):
            dataloader_bar.set_description(f"[nougat] Processing img to markdown")
            if items is None:
                logging.warning(f"Ignore empty batch in page {page_num+1}.")
                continue
            else:
                (sample, is_last_page) = items
            model_output = model.inference(image_tensors=sample)
            # check if model output is faulty
            # output [batch_Size,decoder_seq_len]
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    logging.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1

                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                    continue
                if model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logging.warning(f"Skipping page {page_num} due to repetitions.")
                        predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i*self.batch_size+j+1}]\n\n"
                        )
                else:
                    if self.markdown:
                        output = markdown_compatible(output)
                    predictions.append(output)
                if is_last_page[j]:
                    out = "".join(predictions).strip()
                    out = re.sub(r"\n{3,}", "\n\n", out).strip()
                    out = self.format_transformer(out)
                    if self.out:
                        file_name = Path(Path(pdf_names[file_index]).with_suffix(".md").name)
                        out_path = self.out / file_name.stem / file_name
                        print("out_path:", out_path,"file_name:",file_name)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text(out, encoding="utf-8")
                        files.append(NougatArticle(file_name=str(file_name).split(".")[0],content=out))
                    else:
                        print(out, "\n\n")
                    predictions = []
                    page_num = 0
                    file_index += 1
        torch_gc()
        return files
    @staticmethod
    def format_transformer(text: str) -> str:
        """
        convert latex math to markdown math
        Args:
            text: text with latex math

        Returns: text with markdown math

        """
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
        text = re.sub(r'\\mathbbm', r"\\mathbb", text)
        return text
    @staticmethod
    def _get_pdf_name(pdf):
        if isinstance(pdf,Path):
            return pdf.name
        elif isinstance(pdf,bytes):
            name = f"unk_pdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            return name
        elif isinstance(pdf,str):
            if pdf.startswith("http"):
                name = pdf.split("/")[-1].replace('.', '_')
                return name
            else:
                return Path(pdf).name
    @staticmethod
    def move_to_device(model, cuda: bool = True):

        if cuda and torch.cuda.is_available():
            model = model.to("cuda")
        return model

def __repr__(self):
    return f"NougatPredictor(batch_size={self.batch_size},out={self.out},recompute={self.recompute},markdown={self.markdown},proxy={self.proxy},headers={self.headers})"


