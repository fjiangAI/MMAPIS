import sys
from pathlib import Path
import logging
import logging.config
from typing import List,Union
from datetime import datetime


def save_mmd_file(abs_text: str,
                re_abs_text: str,
                file_name: Union[str,Path],
                save_dir: Union[str, Path],
                summary_sve_dir: str = 'summary_mmd',
                re_summary_sve_dir: str = 're_summary_mmd'
                ):
        assert abs_text is not None , "summary_text is None"
        assert re_abs_text is not None , "re_summary_text is None"
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        file_name ,save_dir = Path(file_name),Path(save_dir)
        filename = file_name.stem + f"_{timestamp}"+file_name.suffix
        abs_path = save_dir / Path(summary_sve_dir) / filename
        re_abs_path = save_dir / Path(re_summary_sve_dir) / filename
        logging.info(f"file {str(file_name)},suammry file save to {abs_path},re_suammry file save to {re_abs_path}")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(abs_text, encoding="utf-8")
        re_abs_path.parent.mkdir(parents=True, exist_ok=True)
        re_abs_path.write_text(re_abs_text, encoding="utf-8")

        return abs_path,re_abs_path

def save_temp_file(file_content:Union[str,bytes],file_name:Union[str,Path],save_dir:Union[str,Path]):

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    file_name ,save_dir = Path(file_name),Path(save_dir)
    filename = file_name.stem + f"_{timestamp}"+file_name.suffix
    file_path = save_dir / filename
    logging.info(f"file {str(file_name)},save to {file_path}")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_content = file_content.encode('utf-8') if isinstance(file_content,str) else file_content
    file_path.write_bytes(file_content)
    return file_path

