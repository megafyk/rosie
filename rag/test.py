import re

import unicodedata
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Normalize and clean text.
    """
    return "\n".join(
        re.sub(r'\s+', ' ', line.strip())
        for line in unicodedata.normalize("NFKC", text).splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    # Sample HTML content (replace with your Confluence HTML content)
    html_content = """
    <html>
      <body>
        <p>This  is  a  sample   text!!</p>
        <p>It contains  extra   spaces,     multiple newlines, and   HTML tags.</p>
        <p></p>
        <p>Another   paragraph??   With duplicate punctuation!!!</p>
        <div>MEGAFYK<div>
      </body>
    </html>
    """

    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")
    print(clean_text(soup.get_text(separator="\n", strip=True)))
