# General title
# FIXME: Misidentification exists
SECTION_TITLE_MATCHSTR = ["[IVX1-9]{1,4}[\.\s][\sA-Za-z]{1,}|[1-9]{1,2}[\s\.\n][\sA-Za-z]{1,}",  # Level 1
                          "[A-M]{1}\.[\sA-Za-z]{1,}|[1-9]\.[1-9]\.[\sA-Za-z]{1,}"]  # Level 2


ABS_MATCHSTR = "ABSTRACT|Abstract|abstract"
REF_MATCHSTR = "Reference|REFERENCE|Bibliography"
SNAP_WITH_CAPTION = True    # Generate images & tables with caption (Only valid when USE_PDFFIGURE2 is True)
"""Debuging"""
DEBUG_MODE = False
CUSTOM_DIR = './json/'
