import os
import sqlite3

def set_sql_connect(database_name):
    return sqlite3.connect(database_name)

def set_sql_cursor(database_connect):
    return database_connect.cursor()

def close_connect(database_connect):
    if database_connect:
        database_connect.commit()
        database_connect.close()

def set_connect_and_cursor(path='Data/database.sqlite'):
    vt = set_sql_connect(path)
    db = set_sql_cursor(vt)
    return vt, db

def create_table(table_name, columns):
    vt, db = set_connect_and_cursor()
    db.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")
    close_connect(vt)

def get_data(sql_command):
    vt, db = set_connect_and_cursor()
    db.execute(sql_command)
    gelen_veri = db.fetchall()
    close_connect(vt)
    return gelen_veri

def add_data(table, values):
    vt, db = set_connect_and_cursor()
    placeholders = ', '.join('?' * len(values))
    db.execute(f"INSERT INTO {table} VALUES ({placeholders})", values)
    close_connect(vt)
