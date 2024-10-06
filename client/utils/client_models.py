from dataclasses import dataclass, field
from typing import List, Dict
from MMAPIS.client.config import ALIGNMENT_CONFIG,GENERAL_CONFIG



@dataclass
class RangeConfig:
    min: float
    max: float
    default: float


@dataclass
class MMAPISClientConfig:
    # Internal Config
    max_entries: int = 50
    ttl: int = 60 * 60 * 24  # 1 day
    max_page_num: int = 5

    # User Config
    color_ls: List[str] = field(default_factory=lambda: ["blue", "green", "orange", "red", "violet", "gray", "rainbow"])
    compatible_models: List[str] = field(
        default_factory=lambda: ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"])

    line_length: RangeConfig = field(default_factory=lambda: RangeConfig(min=10, max=40, default=20))
    num_pdf: RangeConfig = field(default_factory=lambda: RangeConfig(min=1, max=10, default=5))
    img_width: RangeConfig = field(
        default_factory=lambda: RangeConfig(min=100, max=1000, default=ALIGNMENT_CONFIG['img_width']))
    min_grained_level: RangeConfig = field(default_factory=lambda: RangeConfig(min=1, max=5, default=2))
    max_grained_level: RangeConfig = field(default_factory=lambda: RangeConfig(min=2, max=5, default=4))
    max_tokens_map: Dict[str, int] = field(default_factory=lambda: {
        "gpt-3.5-turbo": 16385,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000
    })
    ignore_titles: List[str] = field(default_factory=lambda: [
        "abstract", "introduction", "background", "related work", "reference",
        "appendix", "acknowledge"
    ])
    ignore_title_map: Dict[str, str] = field(default_factory=lambda: {
        "abstract": "abs",
        "introduction": "intro",
        "acknowledge": "acknowledg"
    })
    default_ignore_titles: List[str] = field(default_factory=lambda: ["reference", "appendix", "acknowledge"])

    threshold: RangeConfig = field(default_factory=lambda: RangeConfig(min=0.0, max=1.0, default=0.8))
    num_processes: RangeConfig = field(default_factory=lambda: RangeConfig(min=1, max=10, default=5))
    rpm_limit: RangeConfig = field(default_factory=lambda: RangeConfig(min=0, max=10, default=3))
    temperature : RangeConfig = field(default_factory=lambda: RangeConfig(min=0.0, max=1.0, default=0.5))
    top_p: RangeConfig = field(default_factory=lambda: RangeConfig(min=0.0, max=1.0, default=1.0))
    frequency_penalty: RangeConfig = field(default_factory=lambda: RangeConfig(min=-2.0, max=2.0, default=0.1))
    presence_penalty: RangeConfig = field(default_factory=lambda: RangeConfig(min=-2.0, max=2.0, default=0.2))
    prompt_ratio: RangeConfig = field(default_factory=lambda: RangeConfig(min=0.0, max=1.0, default=0.8))


@dataclass
class ArxivRequestConfig:
    daily_type: List[str] = field(default_factory=lambda: [
        'cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'eess', 'econ', 'astro-ph', 'cond-mat', 'gr-qc',
        'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nucl-ex', 'nucl-th', 'quant-ph'
    ])
    show_abstracts: List[bool] = field(default_factory=lambda: [True, False])
    searchtype: List[str] = field(default_factory=lambda: [
        'all', 'title', 'abstract', 'author', 'comment',
        'journal_ref', 'subject_class', 'report_num', 'id_list'
    ])
    search_order: List[str] = field(default_factory=lambda: [
        '-announced_date_first', 'submitted_date',
        '-submitted_date', 'announced_date_first', ''
    ])
    search_size: List[int] = field(default_factory=lambda: [25, 50, 100, 200])

MMAPIS_CLIENT_CONFIG = MMAPISClientConfig()
ARXIV_REQUEST_CONFIG = ArxivRequestConfig()