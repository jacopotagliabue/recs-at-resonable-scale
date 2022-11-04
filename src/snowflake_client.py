import sys
import snowflake.connector
from snowflake.connector import DictCursor


class SnowflakeClient:

    def __init__(self, user, pwd, account, role, warehouse, keep_alive=False):
        self.snowflake_client = snowflake.connector.connect(
            user=user,
            password=pwd,
            account=account,
            warehouse=warehouse,
            role=role,
            client_session_keep_alive=keep_alive
        )

        return

    def get_version(self):
        q = """select current_version();"""
        return self.fetch_one(query=q)

    def fetch_one(self, query, params=None):
        cs = self.snowflake_client.cursor(DictCursor)
        try:
            cs.execute(query, params)
            one_row = cs.fetchone()
        finally:
            cs.close()
        return one_row

    def execute_query(self, query, params=None, is_debug=False):
        if is_debug:
            print("EXECUTING: \n{}\n".format(query))
            
        is_executed = True
        cs = self.snowflake_client.cursor(DictCursor)
        try:
            cs.execute(query, params)
        except Exception as ex:
            print(ex)
            is_executed = False
        finally:
            cs.close()

        return is_executed

    def fetch_all(self, query, params=None, debug=False):
        cs = self.snowflake_client.cursor(DictCursor)
        rows = None
        try:
            cs.execute(query, params)
            rows = cs.fetchall()
        except:
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            cs.close()

        if rows and debug:
            print('SF debug: # rows {}, first {}'.format(len(rows), rows[0]))

        return rows