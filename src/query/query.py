# src/query/query.py
class Query:
    def __init__(self, database):
        self.database = database

    def execute_query(self, query_text, num_neighbors=5):
        return self.database.query_database(query_text, num_neighbors)
