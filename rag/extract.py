import hashlib
import json
import re
from collections import deque

import unicodedata
from bs4 import BeautifulSoup


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
    text = text.replace("\\", " ")
    return " ".join(
        re.sub(r'\s+', ' ', line.strip())
        for line in unicodedata.normalize("NFKC", text).splitlines()
        if line.strip()
    )

def extract_sections_from_page(id, title, body):
    """
    Extract text from HTML content.
    """
    soup = BeautifulSoup(body, "html.parser")
    # Cleanup unwanted elements
    for tag in soup.find_all(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer']):
        tag.decompose()

    return [{'id': f"{id}_{generate_id_md5(title)}", 'title': title,
             'content': clean_text(soup.get_text(strip=True))}]


def extract_data_from_pages(pages):
    data = []
    for page in pages:
        q = deque([page])
        while q:
            n = len(q)
            for _ in range(n):
                page = q.popleft()
                sections = extract_sections_from_page(page['id'], page['title'], page['body'])
                q.extend(page['children'])
                for section in sections:
                    data.append({
                        'id': section['id'],
                        'page_id': page['id'],
                        'title': section['title'],
                        'content': section['content']
                    })

    return data


if __name__ == "__main__":
    # read data from space_hierarchy.json line by line and extract text from HTML content
    print("----------------start extract----------------")
    with open("../datasets/space_hierarchy.json") as file:
        space_data = extract_data_from_pages(json.load(file))

    file_name = "../datasets/space_data.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(space_data, f, indent=4, ensure_ascii=False)

    print("-----------------end extract----------------")
