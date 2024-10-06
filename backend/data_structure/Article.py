import re
import logging
from typing import List, Union, Tuple, Optional
from MMAPIS.backend.data_structure.Section import Section
import spacy
from functools import partial
from MMAPIS.backend.tools.utils import strip_title


class Article:
    def __init__(self, article: str,
                 min_grained_level: int = 2,
                 max_grained_level: int = 4,
                 ignore_title: List[str] = ['appendix', 'reference', 'acknowledg'],
                 file_name: str = None):
        """
        Initializes an Article object and preprocesses the content for parsing.

        :param article: The raw text of the article.
        :param min_grained_level: The minimum level of section granularity.
        :param max_grained_level: The maximum level of section granularity.
        :param ignore_title: Titles to be ignored during parsing.
        :param file_name: The name of the file from which the article was loaded.
        """
        self._file_name = file_name
        self._article = self._normalize_abstract(article)
        self._remove_tables_and_figures()
        self._ignore_title = ignore_title
        self._titles, self._authors, self._affiliations = self._extract_title_contributors()
        self.sections = self._get_sections_from_md(min_grained_level, max_grained_level)
        self._fail_match_img_paths = []

    @staticmethod
    def _normalize_abstract(article: str) -> str:
        """
        Standardizes the abstract header in the article.

        :param article: The article text.
        :return: The article text with standardized abstract header.
        """
        return re.sub(r'#+\s+Abstract', '## Abstract', article, 1)

    def _remove_tables_and_figures(self):
        """
        Removes table and figure descriptions from the article.
        """
        self._article = re.sub(r'(\\begin{table}.*?\\end{table})\W+(.*?)\n', '', self._article, flags=re.DOTALL)
        self._article = re.sub(r'\n(Figu?r?e?\.?\W?\d+[:|\.].*?)(?=\n|$)', '', self._article, flags=re.DOTALL)

    def _extract_title_contributors(self)-> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extracts the title, authors, and affiliations from the article.

        :return: A tuple containing the title, authors, and affiliations.
        """
        title_pattern = re.compile(r'(#\s+.*?)\n+(.*?)#+', re.DOTALL)
        match = re.match(title_pattern, self._article)

        if match:
            titles, contributors = match.group(1).strip(), match.group(2).strip()
            contributors = self._clean_contributors(contributors)
            authors, affiliations = self._split_contributors(contributors)
            if authors is not None:
                return titles, self._format_transfer(authors), self._format_transfer(affiliations)
            elif contributors:
                contributors_list = contributors.split('<br><br>')
                authors, affiliations = self._filter_with_ner(contributors_list, ner_type=['PERSON', 'ORG'])
                return titles, self._format_transfer(' '.join(authors)), self._format_transfer(' '.join(affiliations))
            else:
                logging.error(f'Article "{self._file_name}": Contributors not found, parser error')
                return titles, contributors, ''

        else:
            logging.warning(f'Article "{self._file_name}": Title not found, parser error')
            return None, None, None

    @staticmethod
    def _clean_contributors(contributors):
        """
        Cleans up the contributor information by removing unnecessary spaces and tags.

        :param contributors: Raw contributor information.
        :return: Cleaned contributor information.
        """
        contributors = re.sub(r'\\+\(\{\}\^\{\\*([^\\]*?)\\*\}\\+\)(\\+\(\{\}\^\{([^\\]*?)\}\\+\))?',r'<sup>\1,\3</sup>', contributors)
        contributors = re.sub(r'>([^,]*?),<',r'>\1<',contributors)
        contributors = re.sub(r'\n', r'<br>', contributors)
        return contributors

    def _split_contributors(self, contributors: str):
        """
        Splits the contributors into authors and affiliations.

        :param contributors: The cleaned contributor information.
        :return: A tuple of lists containing authors and affiliations.
        """
        split_pattern = re.compile(r'<br><br>(?=<sup>)|affiliation', re.IGNORECASE)
        try:
            author_part, affiliation_part = re.split(split_pattern, contributors, 1)
        except ValueError:
            return None, None

        authors = self._extract_with_pattern(author_part, r'([^<>]*?<sup>.*?</sup>)')
        affiliations = self._extract_with_pattern(affiliation_part, r'(<sup>.*?</sup>.*?)(?=<sup>|\n+|$)')
        return ' '.join(self._filter_with_ner(authors, 'PERSON')), ' '.join(self._filter_with_ner(affiliations, 'ORG'))

    @staticmethod
    def _extract_with_pattern(text: str, pattern: str) -> List[str]:
        """
        Extracts the matches from the text using the specified pattern.
        """
        return [match.strip() for match in re.findall(pattern, text)]

    @staticmethod
    def _format_transfer(text: str) -> str:
        """
        Formats the given text for better readability in markdown.

        :param text: The text to format.
        :return: Formatted text.
        """
        format_map = {
            r'>dagger<': '>&dagger;<',
            r'>ddagger<': '>&ddagger;<',
            r'>S<': '>&sect;<',
            r'>P<': '>&para;<',
            r'>clubsuit<': '>&clubs;<',
            r'>diamondsuit<': '>&diams;<',
            r'>heartsuit<': '>&hearts;<',
            r'>spadesuit<': '>&spades;<',
            r'>flat<': '>&flat;<',
            r'>natural<': '>&natural;<',
            r'>sharp<': '>&sharp;<',
        }
        for key, value in format_map.items():
            text = re.sub(rf'{key}', value, text, flags=re.IGNORECASE)
        return text

    def _filter_with_ner(self,
                         text: List[str],
                         ner_type: Union[str, List[str]] = 'PERSON') -> Union[List[str], tuple]:
        """
        Filters text using Named Entity Recognition (NER) to identify specific entities like authors or affiliations.

        :param text: List of strings to be filtered.
        :param ner_type: The entity type(s) to recognize, e.g., 'PERSON' for people.
        :return: A list or tuple containing filtered text based on the specified entity type.
        """
        nlp = spacy.load("en_core_web_lg")
        pattern = r'([^<>]*?)<sup>.*?</sup>' if ner_type == 'PERSON' else r'<sup>.*?</sup>([^<>]*?)'

        def filter_func(doc,ner_type=ner_type):
            return all(ent.label_ == ner_type for ent in doc.ents)

        # input text is filtered by regular expression, so it format is fixed
        if isinstance(ner_type, str):
            pure_text = [re.search(pattern, i).group(1) for i in text]
            docs = [nlp(i) for i in pure_text]
            return [text[i] for i in range(len(text)) if filter_func(docs[i])]

        # input text is not formatted
        else:
            filter_func = partial(filter_func, ner_type=ner_type[0])
            docs = [nlp(i) for i in text]
            list1 = []
            list2 = []
            for index, i in enumerate(docs):
                if filter_func(i):
                    list1.append(text[index])
                else:
                    list2.append(text[index])
            return list1, list2

    def _is_ignore_title(self, title: str) -> bool:
        """
        Checks whether the given title is in the ignore list.

        :param title: Title to be checked.
        :return: Boolean indicating whether the title should be ignored.
        """
        return any(ignore in title.strip().lower() for ignore in self._ignore_title)

    def _is_valuable_title(self, section_title: str) -> bool:
        """
        Determines if a title is valuable, i.e., not empty and not in the ignore list.

        :param section_title: Section title to be evaluated.
        :return: Boolean indicating whether the title is valuable.
        """
        return bool(section_title.strip()) and not self._is_ignore_title(section_title)

    def _get_sections_from_md(self, min_grained_level: int, max_grained_level: int) -> List['Section']:
        """
        Parses the article into sections based on specified granularity levels.
        This method identifies sections within an article using a regular expression pattern that
        matches titles from the minimum to maximum heading level (e.g., # for H1, ## for H2, etc.).
        It then constructs a list of Section objects, where each object represents a section title
        and its corresponding content.

        :param min_grained_level: Minimum level of section granularity.
        :param max_grained_level: Maximum level of section granularity.
        :return: A list representing the sections in the article.
        """

        section_pattern = re.compile(
            r'\n+(#{{1,{}}}\s+.*?)\n+(.*?)(?=\n+#{{1,{}}}\s+|$)'.format(max_grained_level, max_grained_level),
            re.DOTALL
        )
        section_ls = []
        section_matches = re.findall(section_pattern, self._article)
        for section in section_matches:
            section_title, section_text = section
            if self._is_valuable_title(section_title):
                # Handle cases where the following text starts with another header
                if section_text.strip().startswith('#'):
                    # Add a newline to the start of the text to maintain formatting
                    section_text = "\n" + section_text.strip()
                    # Search for another section within this text
                    # Since it is uncommon to encounter a structure like "## [title]\n### [title]\n#### [title]\n".
                    # We perform an additional single unpack operation,to address the scenario where the following pattern occurs: "## [title]\n### [title]\n[following text]",
                    matches = re.search(section_pattern, section_text)
                    # If a sub-section is found, append both the current and sub-section
                    if matches:
                        section_ls.append(Section(title=section_title,text=""))
                        section_ls.append(Section(title=matches.group(1), text=matches.group(2)))
                    else:
                        # If no sub-section is found, append the current section with its text
                        section_ls.append(Section(title=section_title,text=section_text))
                else:
                    section_ls.append(Section(title=section_title, text=section_text))
        return self._resort_sections(section_ls,min_grained_level)


    def _resort_sections(self,
                         sections: List[Section],
                         min_grained_level:int,
                         ) -> List['Section']:
        """
        Reorders and organizes sections into a hierarchical structure.
        This function ensures that sections at or below the specified minimum granularity level are properly
        nested under their respective parent sections. Sections above the minimum level are treated as top-level
        sections without any children.

        :param sections: List of Section objects to be reordered.
        :param min_grained_level: The minimum granularity level for sections.
        :return: A reordered list of Section objects with hierarchical structure.
        """

        # As the GPT-generated summary might omit details from the finer-grained sections(e.g., level 3 or 4) while keeping the top-level sections(e.g., level 2),
        # we assign these sections to their respective parent sections. For instance, sections with a granularity level of 2.
        PARENT_LEVEL = 2

        res = []
        length = len(sections)
        i = j = 0
        parent = ''

        while i < length:
            cur_section = sections[i]
            # If the current section is at the defined parent level, set it as the new parent
            if cur_section.grained_level == PARENT_LEVEL:
                parent = cur_section.title

            # If the current section is above the minimum granularity level, add it as a top-level section
            if cur_section.grained_level < min_grained_level:
                res.append(Section(title=cur_section.title,text=cur_section.text,parent=parent))
                i += 1
                j = i
                continue

            # If the current section is at the minimum granularity level, gather its sub-sections
            elif cur_section.grained_level == min_grained_level:
                j = i + 1
                while j < length:
                    if sections[j].grained_level <= min_grained_level:
                        break
                    sections[j].parent = parent
                    j += 1

                # Collect all sub-sections of the current section
                sub_sections = sections[i + 1:j]
                res.append(Section(
                    title=cur_section.title,
                    text=cur_section.text,
                    parent=parent,
                    children_list=sub_sections
                ))
                i = j

            # Process sections that exceed the minimum granularity level and lack a preceding parent section.
            # This is necessary when the initial pages are not successfully parsed.
            else:
                j = i + 1
                while j < length:
                    if sections[j].grained_level <= min_grained_level:
                        break
                    sections[j].parent = parent
                    j += 1

                # Collect all sub-sections of the current section
                sub_sections = sections[i + 1:j]

                res.append(Section(
                    title=cur_section.title,
                    text=cur_section.text,
                    parent=parent,
                    children_list=sub_sections
                ))
                i = j
        return res

    @staticmethod
    def _judge_na(text: str) -> bool:
        """
        Determines whether a given text is None or empty.

        :param text: The text to be evaluated.
        :return: Boolean indicating whether the text is None or empty.
        """
        return not text or not text.strip()

    @property
    def extra_info(self) -> str:
        """
        Returns additional article information such as title, authors, and affiliations.

        :return: A string containing the extra information.
        """
        titles = '' if self._judge_na(self._titles) else self._titles.replace('<br>', '\n') + "\n"
        authors = '' if self._judge_na(self._authors) else "- Authors: " + self._authors.replace('<br>', '\n') + "\n"
        affiliations = '' if self._judge_na(self._affiliations) else "- Affiliations: " + self._affiliations.replace(
            '<br>', '\n') + "\n"
        return '\n'.join([titles, authors, affiliations])

    def __str__(self) -> str:
        """
        Returns a string representation of the article, including the name and sections.

        :return: A string describing the article.
        """
        res_msg = f"Article name: {self._file_name}, sections:\n"
        res_msg += "\n".join([str(section) for section in self.sections])
        return res_msg

    def __repr__(self) -> str:
        """
        Returns a formal string representation of the Article object.

        :return: A string representing the Article object.
       """
        return f"Article(file_name={self._file_name}, titles={self._titles})"

    def iter_sections(self):
        """
        Returns an iterator containing the title and text of each section.

        :return: An iterator over the section titles and texts.
        """
        return iter([[section.title, section.text] for section in self.sections])

    def assign_img2section(self,
                           img_paths: List['SectionIMGPaths'],
                           nlp,
                           threshold: float = 0.8,
                           align_raw_md_text: bool = False):
        """

        Args:
            img_paths:
            nlp:
            threshold:

        Returns:

        """
        if align_raw_md_text:
            self.enhanced_alignment(img_paths, nlp, threshold)
        else:
            self.normal_alignment(img_paths, nlp, threshold)


    def enhanced_alignment(self, img_paths, nlp, threshold: float = 0.8):
        """
        Assigns image paths to the appropriate sections within the document. The matching is performed
        based on the similarity between section titles and image section/parent names, using a natural
        language processing (NLP) model, specifically spaCy.

        :param img_paths: A list of SectionIMGPaths objects, each containing image paths and associated section names.
        :param nlp: An NLP model used to calculate the similarity between section titles and image section names.
        :param threshold: The similarity threshold above which an image will be assigned to a section.
        :return: None. The image paths are assigned to sections in the document. Unmatched images are stored in _fail_match_img_paths.
        """
        fail_match_img_paths = []

        # Iterate through each set of image paths to assign them to the correct section.
        for section_img_paths in img_paths:
            img_section_name = section_img_paths.section_name
            img_parent_name = section_img_paths.parent
            similarities = []
            matched = False  # Flag to indicate if a match was found

            for i, section in enumerate(self.sections[::-1]):
                section_name = strip_title(section.title)
                # Calculate similarity between the image's section/parent name and the document section titles
                if img_parent_name:
                    similarity = max(nlp(section_name).similarity(nlp(img_section_name)),
                                     nlp(section_name).similarity(nlp(img_parent_name)))
                    similarities.append(similarity)
                else:
                    # If no parent name is provided, consider the image paths as unmatched
                    fail_match_img_paths.extend(section_img_paths.img_path)
                    matched = True
                    break

                # If an exact match is found by name, assign the image paths directly to the section
                if (img_section_name.lower() in section_name.lower() or
                        img_parent_name.lower() in section_name.lower()):
                    matched = True
                    section.add_img_paths(section_img_paths.img_path)
                    break

            # If no direct match was found, use the highest similarity score for assignment
            if not matched and similarities:
                max_similarity = max(similarities)
                if max_similarity > threshold:
                    max_similarity_index = similarities.index(max_similarity)
                    self.sections[max_similarity_index].add_img_paths(section_img_paths.img_path)
                else:
                    fail_match_img_paths.extend(section_img_paths.img_path)

        # Store unmatched image paths
        self._fail_match_img_paths = fail_match_img_paths

    def normal_alignment(self, img_paths, nlp, threshold: float = 0.8):
        """
        Aligns image paths to their corresponding sections within the document based on the similarity
        between section titles and image section names using a natural language processing (NLP) model.

        Args:
            img_paths (list[SectionIMGPaths]): A list of SectionIMGPaths objects, each containing
                                               image paths and their associated section names.
            nlp (callable): An NLP model or function that calculates the similarity between
                            section titles and image section names.
            threshold (float): The similarity threshold above which an image will be assigned to a section.

        Returns:
            None: The image paths are assigned to the appropriate sections in the document.
                  Any unmatched images are stored in the _fail_match_img_paths attribute.
        """
        fail_match_img_paths = []

        # Iterate through each set of image paths to assign them to the appropriate section.
        for section_img_paths in img_paths:
            img_section_name = section_img_paths.section_name

            # If the section name is unknown, add the image paths to the list of unmatched paths
            if img_section_name.startswith("unknown_section_"):
                fail_match_img_paths.extend(section_img_paths.img_path)
                continue

            similarities = []
            matched = False  # Flag to indicate if a match was found

            for i, section in enumerate(self.sections):
                if section.title.startswith("# "):
                    continue
                section_name = strip_title(section.title)
                # Check if the image section name is a substring of the document section name
                if img_section_name.lower() in section_name.lower():
                    matched = True
                    section.add_img_paths(section_img_paths.img_path)
                    break
                else:
                    # Calculate similarity between the section title and the image section name
                    similarity = nlp(section_name).similarity(nlp(img_section_name))
                    similarities.append(similarity)

            # If a match was not directly found, use the highest similarity score for alignment
            if not matched and similarities:
                max_similarity = max(similarities)
                if max_similarity > threshold and section_img_paths.img_path:
                    max_similarity_index = similarities.index(max_similarity)
                    self.sections[max_similarity_index].add_img_paths(section_img_paths.img_path)
                else:
                    # If no suitable match is found, add the image paths to the unmatched list
                    fail_match_img_paths.extend(section_img_paths.img_path)

        # Store unmatched image paths
        self._fail_match_img_paths = fail_match_img_paths

    @property
    def fail_match_img_paths(self):
        return self._fail_match_img_paths
