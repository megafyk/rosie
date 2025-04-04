import hashlib
import json
import re
import unicodedata
from collections import deque
from timeit import default_timer as timer

import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame
from underthesea import text_normalize


def generate_id_md5(input_string: str) -> str:
    # Encode the string into bytes
    encoded_string = input_string.encode('utf-8')
    # Compute the MD5 hash
    md5_hash = hashlib.md5(encoded_string)
    # Return the hexadecimal digest
    return md5_hash.hexdigest()


def clean_text(text: str) -> str:
    """
    Normalize and clean text.
    """
    text = text.lower()
    return text_normalize(" ".join(
        re.sub(r'[\\|\s+|\d+(\.\d+)+]', ' ', line.strip())
        for line in unicodedata.normalize("NFKC", text).splitlines()
        if line.strip()
    ))


def extract_tables(table) -> list[DataFrame]:
    """Converts an HTML table to a list of pandas DataFrames, treating nested tables as separate tables."""
    tables = []
    rows = []
    for row in table.find_all('tr'):
        cells = []
        for cell in row.find_all(['td', 'th']):
            nested_tables = cell.find_all(['table'])
            for nt in nested_tables:
                tables.extend(extract_tables(nt))
                nt.decompose()
            cells.append(clean_text(cell.get_text(separator=" ", strip=True)))
        if cells:
            rows.append(cells)

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        df.columns = df.iloc[0]  # Set first row as header
        df = df[1:].reset_index(drop=True)
        df = df.loc[:, df.columns.notna() & (df.columns != '')]  # remove None header columns
        df = df.loc[:, ~df.columns.duplicated(keep="first")]  # remove duplicate columns
        tables.append(df)

    return tables


def extract_section_data(sibling, paragraphs, tables):
    if sibling.name in ("p", "span", "ul", "ol", "strong", "em", "u"):
        tx = clean_text(sibling.get_text(separator=" ", strip=True))
        if tx:
            paragraphs.append(tx)
    elif sibling.name == "table":
        tables.extend(extract_tables(sibling))


def extract_sections_from_page(id, title, body):
    """
    Extract text from HTML content.
    """
    soup = BeautifulSoup(body, "html.parser")

    header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    headers = soup.find_all(header_tags)
    if not headers:
        paragraphs = []
        tables = []
        extract_section_data(soup, paragraphs, tables)
        return [{
            'id': f"{id}_{generate_id_md5(title)}",
            'section': title,
            'paragraphs': paragraphs,
            'tables': tables,
        }]

    secs = []
    for idx, header in enumerate(headers):
        cur_section = header.get_text(strip=True)
        paragraphs = []
        tables = []
        for sibling in header.find_next_siblings():
            if sibling.name in header_tags:
                break
            extract_section_data(sibling, paragraphs, tables)

        if paragraphs or tables:
            secs.append({
                'id': f"{id}_{generate_id_md5(cur_section)}",
                'section': clean_text(cur_section),
                'paragraphs': paragraphs,
                'tables': tables,
            })
    return secs


def extract_data_from_pages(pages):
    data = []
    queue = deque(pages)

    while queue:
        page = queue.popleft()
        queue.extend(page['children'])

        for sec in extract_sections_from_page(page['id'], page['title'], page['body']):
            page_title = clean_text(f"{page['title']}")
            section_title = clean_text(f"{sec['section']}")
            content = ' '.join(sec['paragraphs'])

            if content:
                data.append({'id': sec['id'], 'page_id': page['id'], 'title': section_title,
                             'content': f"{page_title} - {section_title} - {content}"})

            df: DataFrame
            metadata = {
                "table_name": page_title,
                "table_description": section_title
            }
            for df in sec['tables']:
                tbl_content = {
                    **metadata,
                    "data": df.to_dict(orient='records')
                }
                data.append({'id': sec['id'], 'page_id': page['id'], 'title': section_title,
                             'content': json.dumps(tbl_content, ensure_ascii=False)})

    return data


if __name__ == "__main__":
    """ Testing code
        with open("../datasets/abc.html", encoding="utf-8") as f:
        html_page = f.read()

        sections = extract_sections_from_page(0, "1", html_page)
        print(sections)
        print(sections[6]['tables'][19].to_json(orient='records', force_ascii=False))
    """

    # read data from space_hierarchy.json line by line and extract text from HTML content
    start_time = timer()
    print("----------------start extract----------------")
    with open("../datasets/space_hierarchy.json", encoding="utf-8") as file:
        space_data = extract_data_from_pages(json.load(file))

    file_name = "../datasets/space_data.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(space_data, f, indent=4, ensure_ascii=False)
    end_time = timer()
    print("-----------------end extract----------------")
    print(f"Elapped {end_time - start_time}s")
