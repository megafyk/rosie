from ssl import ALERT_DESCRIPTION_INTERNAL_ERROR
from atlassian import Confluence
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Confluence API
confluence = Confluence(
    url=os.getenv('CONFLUENCE_URL'),
    username=os.getenv('CONFLUENCE_USERNAME'),
    password=os.getenv('CONFLUENCE_PASSWORD')
)
SPACE_KEY = os.getenv('CONFLUENCE_SPACE_KEY')

def get_child_pages(page_id, start=0, limit=100):
    """
    Fetch all child pages of a given page ID.
    """
    children = confluence.get_page_child_by_type(
        page_id=page_id,
        type="page",
        start=start,
        limit=limit,
        expand="body.storage"
    )
    return children

def fetch_page_content_recursively(page_id, level=0):
    """
    Fetch the content of a page and recursively fetch its child pages.
    """
    # Get the current page
    page = confluence.get_page_by_id(
        page_id=page_id,
        expand="body.storage"
    )

    title = page.get("title")
    body = page["body"]["storage"]["value"]  # HTML content of the page

    # Print the page structure for visualization
    print(" " * (level * 4) + f"- {title} (ID: {page_id})")

    # Store page data
    page_data = {
        "id": page_id,
        "title": title,
        "body": body,
        "children": []
    }

    # Fetch child pages
    children = get_child_pages(page_id)
    for child in children:
        child_data = fetch_page_content_recursively(child["id"], level + 1)
        page_data["children"].append(child_data)

    return page_data

def get_space_hierarchy(space_key, start=0, limit=100):
    """
    Build the hierarchy of pages in a Confluence space.
    """
    # Fetch all root pages (pages with no ancestors)
    root_pages = [
        page for page in confluence.get_all_pages_from_space(
            space=space_key,
            start=start,
            limit=limit,
            expand="ancestors"
        ) if not page["ancestors"]
    ]


    # Build the hierarchy by recursively fetching content
    space_hierarchy = []
    for root_page in root_pages:
        space_hierarchy.append(fetch_page_content_recursively(root_page["id"]))

    return space_hierarchy

if __name__ == "__main__":
    # Get the full hierarchy of the space
    print(f"Fetching all pages and their child content from space '{SPACE_KEY}'...\n")
    space_hierarchy = get_space_hierarchy(SPACE_KEY)

    with open("datasets/space_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(space_hierarchy, f, indent=4, ensure_ascii=False)

    print("Hierarchy saved to 'space_hierarchy.json'")
