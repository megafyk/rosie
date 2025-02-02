import json

from bs4 import BeautifulSoup, Tag


def extract_sections_from_page(id, title, html):
    """
    Extract text from HTML content.
    """
    no_data_tags = ['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer']
    header_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

    soup = BeautifulSoup(html, "html.parser")
    # Cleanup unwanted elements
    for element in soup(no_data_tags):
        element.decompose()

    # split soup data into sections, each sections is a header tag

    headers = soup.find_all(header_tags)

    if not headers:
        # No headers found - return default section
        text = soup.get_text(strip=True)
        return [{'id': f"{id}_{title}", 'content': ' '.join(text.split())}]

    return [{
        'id': f"{id}_{h.get_text(strip=True)}",
        'content': ' '.join([t.get_text(strip=True) if isinstance(t, Tag) else str(t).strip()
                             for t in h.next_siblings
                             if t is not nh and not (isinstance(t, Tag) and t.name in header_tags)])
    } for h, nh in zip(headers, headers[1:] + [None])]


if __name__ == "__main__":
    # read data from space_hierarchy.json line by line and extract text from HTML content
    with open("../datasets/space_hierarchy.json") as file:
        print("----------------start extract----------------")

        # test
        page = json.load(file)[0]["children"][2]["children"][1]["children"][1]["children"][0]
        page_id = page["id"]
        page_title = page["title"]
        page_body = page["body"]
        print(extract_sections_from_page(page_id, page_title, page_body))

        print("-----------------end extract----------------")
