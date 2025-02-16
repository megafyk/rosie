from pgvector.psycopg2 import register_vector
from psycopg2 import pool

from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_MIN_CONN, DB_MAX_CONN


class Database:
    def __init__(self, db_name, db_user, db_password, db_host, db_port):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.pool = pool.ThreadedConnectionPool(
            minconn=DB_MIN_CONN,
            maxconn=DB_MAX_CONN,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def get_connection(self):
        conn = self.pool.getconn()
        register_vector(conn)  # Register vector type with connection
        return conn

    def return_connection(self, conn):
        self.pool.putconn(conn)

    def execute_query(self, query, params=None):
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            if params is not None:
                cur.execute(query, params)
            else:
                cur.execute(query)
            results = cur.fetchall()
            return results
        finally:
            self.return_connection(conn)


# Create a Database object
db = Database(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
