import hashlib
import json
from collections import deque

from bs4 import BeautifulSoup, Tag


def generate_id_md5(input_string: str) -> str:
    # Encode the string into bytes
    encoded_string = input_string.encode('utf-8')
    # Compute the MD5 hash
    md5_hash = hashlib.md5(encoded_string)
    # Return the hexadecimal digest
    return md5_hash.hexdigest()


def extract_sections_from_page(id, title, body):
    """
    Extract text from HTML content.
    """
    no_data_tags = ['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer']
    header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

    soup = BeautifulSoup(body, "html.parser")
    # Cleanup unwanted elements
    for element in soup(no_data_tags):
        element.decompose()

    # split soup data into sections, each sections is a header tag

    headers = soup.find_all(header_tags)

    if not headers:
        # No headers found - return default section
        text = soup.get_text(strip=True)
        return [{'id': f"{id}_{generate_id_md5(title)}", 'title': title, 'content': ' '.join(text.split())}]

    return [{
        'id': f"{id}_{generate_id_md5(h.get_text(strip=True))}",
        'title': h.get_text(strip=True),
        'content': ' '.join([t.get_text(strip=True) if isinstance(t, Tag) else str(t).strip()
                             for t in h.next_siblings
                             if t is not nh and not (isinstance(t, Tag) and t.name in header_tags)])
    } for h, nh in zip(headers, headers[1:] + [None])]


def extract_data_from_pages(pages):
    data = []
    for page in pages:
        q = deque([page])
        while q:
            print(f"process page id {page['id']}")
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
