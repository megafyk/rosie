import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_MIN_CONN= os.getenv('DB_MIN_CONN')
DB_MAX_CONN = os.getenv('DB_MAX_CONN')
NUM_CPUS = os.getenv('NUM_CPUS')
NUM_GPUS = os.getenv('NUM_GPUS')
NUM_NODES = os.getenv('NUM_NODES')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
