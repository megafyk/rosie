from atlassian import Confluence
from dotenv import load_dotenv
import os

load_dotenv()

# Confluence API
# login to confluence
confluence = Confluence(url=os.getenv('CONFLUENCE_URL'), username=os.getenv('CONFLUENCE_USERNAME'), password=os.getenv('CONFLUENCE_PASSWORD'))
