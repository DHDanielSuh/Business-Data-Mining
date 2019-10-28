#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mysql.connector
from Common import HoonUtils as utils
import pandas as pd
import sys


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 3306


class MariaDbHandler:

    def __init__(self, username, passwd, hostname=DEFAULT_HOST, port=DEFAULT_PORT,
                 database=None, table=None, logger=None, show_=False):

        self.con = None
        self.db_name = database
        self.table_name = table
        self.db_name = database
        self.logger = logger
        if self.logger is None:
            self.logger = utils.setup_logger(None, None, logger_=False)

        if show_:
            self.logger.info(" # connect db : {}:{}".format(hostname, port))
            self.logger.info(" # Create the instance of MariaDB handler, \"{}.{}\"".format(database, table))
        try:
            self.con = mysql.connector.connect(host=hostname,
                                               port=port,
                                               user=username,
                                               passwd=passwd,
                                               database=database)
        except Exception as e:
            self.logger.error(e)

    def create_database(self, db_name):
        cursor = self.con.cursor()
        try:
            cursor.execute("CREATE DATABASE " + db_name)
        except Exception as e:
            self.logger.error(e)

    def show_databases(self):
        cursor = self.con.cursor()
        cursor.execute("SHOW DATABASES")
        self.print_cursor(cursor)

    def use_database(self, db_name, show_=False):
        if show_:
            self.logger.info(" # Use \"{}\" database.".format(db_name))
        cursor = self.con.cursor()
        try:
            cursor.execute("USE " + db_name)
        except Exception as e:
            self.logger.error(e)

    def create_table(self, table_name, csv_fname, skip_row_num=0, skip_col_num=0, auto_inc_pk_=True, show_=False):
        if show_:
            self.logger.info(" # Create \"{}\" table.".format(table_name))

        utils.file_exists(csv_fname, exit_=True)
        df = pd.read_csv(csv_fname, skiprows=skip_row_num)
        df = df.drop(df.columns[[x for x in range(skip_col_num)]], axis=1)
        sql = "CREATE TABLE " + table_name + " ("
        if auto_inc_pk_:
            sql += "id INT AUTO_INCREMENT PRIMARY KEY,"

        sql += " {} {}".format(df.loc[0][0].strip(), df.loc[0][1].strip())
        if isinstance(df.loc[0][2], str) and len(df.loc[0][2]) > 0:
            sql += " " + df.loc[0][2].strip()

        for row in range(1, df.shape[0]):
            if not isinstance(df.loc[row][0], str):
                break

            sql += ", {} {}".format(df.loc[row][0].strip(), df.loc[row][1].strip())
            if isinstance(df.loc[row][2], str) and len(df.loc[row][2]) > 0:
                sql += " " + df.loc[row][2].strip()
        sql += " )"

        cursor = self.con.cursor()
        try:
            cursor.execute(sql)
        except Exception as e:
            self.logger.error(e)

    def describe_table(self, table_name):
        cursor = self.con.cursor()
        cursor.execute("DESCRIBE " + table_name)
        self.logger.info("")
        self.logger.info(" # TABLE SCHEMA :{}".format(table_name))
        self.print_cursor(cursor)

    def show_table(self):
        cursor = self.con.cursor()
        cursor.execute("SHOW TABLES")
        self.print_cursor(cursor)

    def drop_table(self, table_name, show_=False):

        if show_:
            self.logger.info(" # Drop \"{}\" table.".format(table_name))

        try:
            cursor = self.con.cursor()
            cursor.execute("DROP TABLE " + " IF EXISTS " + table_name)
        except Exception as e:
            self.logger.error(e)
            sys.exit(1)

    @staticmethod
    def get_string_with_quotes(values):
        values_str = ""
        for value in values:
            values_str += "\'" + value + "\', "
        return values_str[:-2]

    def insert_into_table(self, table_name, value_dicts, logging=False):
        if len(value_dicts)==0:
            return False

        sql  = "INSERT INTO " + table_name + " (" + ', '.join(value_dicts.keys()) + ") "
        sql += "VALUES (" + self.get_string_with_quotes(value_dicts.values()) + ")"

        try:
            cursor = self.con.cursor()
            cursor.execute(sql)
            self.con.commit()
            self.print_cursor(cursor)
            if logging:
                self.logger.info("")
                self.logger.info(" # Insert record into table : {}".format(sql))
        except Exception as e:
            self.logger.error(e)

        return True

    def insert_csv_file_into_table(self, table_name, csv_fname, row_num=-1):

        utils.file_exists(csv_fname, exit_=True)
        df = pd.read_csv(csv_fname, dtype=object)

        sql  = "INSERT INTO " + table_name + " (" + ', '.join(df.columns) + ") "
        sql += "VALUES (" + ", ".join(["%s"] * len(df.columns)) + ")"

        if row_num < 0:
            row_num = df.shape[0]

        vals = []
        for i in range(row_num):
            if not isinstance(df.loc[i][0], str):
                break
            val = []
            for j in range(len(df.columns)):
                col = df.loc[i][j]
                if not isinstance(col, str):
                    col = ""
                val.append(col)
            vals.append(tuple(val))

        try:
            cursor = self.con.cursor()
            cursor.executemany(sql, vals)
            self.con.commit()
            self.print_cursor(cursor)
            self.logger.info("")
            self.logger.info(" # Insert csv file into table: {:d} was inserted.".format(cursor.rowcount))
        except Exception as e:
            self.logger.error(e)

        return True

    def update_column_into_row(self,
                               table_name,
                               col_name,
                               col_cond,
                               row_name,
                               row_cond,
                               show_=False):

        if show_:
            self.logger.info(" # Update \"{}\" table.".format(table_name))

        try:
            cursor = self.con.cursor()
            sql = "UPDATE " + table_name + \
                  " SET " + col_name + " = \'" + col_cond + "\'" + \
                  " WHERE " + row_name + " = \'" + row_cond + "\'"
            cursor.execute(sql)
            self.con.commit()
            self.logger.info(" # Update column into row : {:d} record(s) affected.".format(cursor.rowcount))
        except Exception as e:
            self.logger.error(e)

        return True

    def update_columns_into_row(self,
                                table_name,
                                value_dicts,
                                row_name,
                                row_cond,
                                show_=False):

        if show_:
            self.logger.info(" # Update \"{}\" table.".format(table_name))

        try:
            cursor = self.con.cursor()
            sql = "UPDATE " + table_name + \
                  " SET "

            value_list = []
            for col_name in value_dicts.keys():
                value_list.append(col_name + " = \'" + value_dicts[col_name] + "\'")

            sql += ", ".join(value_list)
            sql += " WHERE " + row_name + " = \'" + row_cond + "\'"
            cursor.execute(sql)
            self.con.commit()
            self.logger.info(" # Update columns into row : {:d} record(s) affected.".format(cursor.rowcount))
        except Exception as e:
            self.logger.error(e)

        return True

    def select_all(self, table_name):

        try:
            cursor = self.con.cursor()
            cursor.execute("SELECT * FROM " + table_name)
            result = cursor.fetchall()
            self.logger.info("")
            self.logger.info(" # Select ALL")
            self.print_cursor(result)
        except Exception as e:
            self.logger.error(e)

    def execute(self, query, show_=False):

        try:
            cursor = self.con.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            if show_:
                self.print_cursor(result)
            return result
        except Exception as e:
            self.logger.error(e)
            return []

    def select_columns(self, table_name, column_names, show_=False):

        try:
            cursor = self.con.cursor()
            cursor.execute("SELECT " + ", ".join(column_names) + " FROM " + table_name)
            result = cursor.fetchall()
            if show_:
                self.print_cursor(result)
            return result
        except Exception as e:
            self.logger.error(e)
            return []

    def select_with_filter(self, table_name, filter_string, col_names=None, show_=False):

        try:
            sql  = "SELECT " + ", ".join(col_names) if col_names else "SELECT *"
            sql += " FROM " + table_name + " WHERE " + filter_string

            cursor = self.con.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            if show_:
                self.print_cursor(result)
            return result
        except Exception as e:
            self.logger.error(e)
            return []

    def print_cursor(self, cursor):
        for x in cursor:
            self.logger.info(x)

    """


    def set_collection(self, collection):
        try:
            self.collection = self.database[collection]
        except Exception as e:
            self.logger.error(" @ Error while setting collection : " + collection)
            raise e

    def insert_one(self, param_dict):
        try:
            self.collection.insert_one(param_dict)
        except Exception as e:
            self.logger.error(" @ Error while inserting one document : " + param_dict)
            raise e

    def insert_many(self, param_list):
        try:
            self.collection.insert_many(param_list)
        except Exception as e:
            self.logger.error(" @ Error while inserting many documents : " + param_list)
            raise e

    def find(self):
        return self.collection.find()

    def find_one(self, param_dict):
        # collection.find_one({"name": "이형주"})
        return self.collection.find_one(param_dict)
"""
