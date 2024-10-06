"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import logging

import fitz
from pathlib import Path
from tqdm import tqdm
import io
from PIL import Image
from typing import Optional, List
import requests
from MMAPIS.backend.tools.utils import get_pdf_doc
def rasterize_paper(
    pdf: [Path,str,bytes],
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
    proxy=None,
    headers=None
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """


    pils = []
    if outpath is None:
        return_pil = True
    try:
        pdf,_,_ = get_pdf_doc(pdf,proxy=proxy,headers=headers)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pils.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        logging.error(f"Error rasterizing {pdf if not isinstance(pdf, bytes) else 'bytes'}")
        pass
    if return_pil:
        return pils


