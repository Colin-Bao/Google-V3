#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :mysql_dao.py
# @Time      :2022/6/20 09:06
# @Author    :Colin
# @Note      :None

from my_tools import tools


# 通用的执行语句
def excute_sql(sql, tups=None):
    import mysql.connector
    cnx = mysql.connector.connect(user='root', password='',
                                  host='127.0.0.1',
                                  database='wechat_offacc')
    cur = cnx.cursor(buffered=True)

    # 查询以外的sql
    try:
        if tups is not None:
            cur.executemany(sql, tups)
            cnx.commit()
        else:
            # 查询sql
            cur.execute(sql)
    except mysql.connector.Error as e:
        print(e)
    finally:
        cur.close()
        cnx.close()


# 创建一个表格,传入表格名与字段名和类型,主键
# 例如
# create_table('testtable', {'test1': 'VARCHAR(25)', 'test2': 'VARCHAR(25)', 'PRIMARY KEY': 'test2'})
def create_table(table_name: str, column_dict: dict):
    # 解析column_dict
    def transform_dict():
        # 主键处理
        # 追加NOT NULL
        if 'PRIMARY KEY' in column_dict.keys():
            pk_column = column_dict['PRIMARY KEY']
            column_dict[pk_column] = column_dict[pk_column] + ' NOT NULL'

        colum_list = ['' if i == 'PRIMARY KEY' else i + ' ' + j for i, j in column_dict.items()]

        colum_list += ['PRIMARY KEY' + '(`' + column_dict['PRIMARY KEY'] + '`)']
        colum_str = '(' + ','.join(colum_list) + ')'

        # 去掉连续逗号
        colum_str = colum_str.replace(',,', ',')

        return colum_str

    def excute():
        sql_column = transform_dict()

        sql = ("CREATE TABLE IF NOT EXISTS " + table_name + sql_column
               )
        excute_sql(sql)

    excute()


# 需要传入df
def insert_table(table_name: str, df_values):
    # 转换插入的df
    def transform_df():
        import pandas as pd
        df = pd.DataFrame(df_values)

        # column_str
        column_str = ['`' + i + '`' for i in df.columns]
        column_str = '(' + ','.join(column_str) + ')'

        # values_str
        values_str = ['%s' for i in range(len(df.columns))]
        values_str = '(' + ','.join(values_str) + ')'

        # tup_values
        values_tup = tools.df_to_tup(df)

        return column_str, values_str, values_tup

    def execute():
        # 获取要插入的数据
        column, values, tups = transform_df()

        # 创建sql语句
        sql = ("INSERT IGNORE INTO " + '`' + table_name + '`' + column +
               " VALUES " + values
               )

        # 执行sql语句
        excute_sql(sql, tups)

    execute()


# 需要传入df
def update_table(table_name: str, df_values):
    # 转换插入的df
    def transform_df():
        import pandas as pd
        df = pd.DataFrame(df_values)

        # column_str
        column_str = ['`' + i + '`' for i in df.columns]
        column_str = '(' + ','.join(column_str) + ')'

        # values_str
        values_str = ['%s' for i in range(len(df.columns))]
        values_str = '(' + ','.join(values_str) + ')'

        # tup_values
        values_tup = tools.df_to_tup(df)

        return column_str, values_str, values_tup

    def execute_many():
        # 获取要插入的数据
        tups = ()
        # 创建sql语句
        sql = ("UPDATE " + '`' + table_name + '` ' +
               "SET log_return = %s,log_return_2 = %s "
               "WHERE date_ts = %s ")

        # 执行
        cnx = tools.conn_to_db()
        cur = cnx.cursor(buffered=True)
        cur.executemany(sql, tups)
        cnx.commit()
        cnx.close()

    execute_many()


if __name__ == '__main__':
    create_table('testtable', {'test1': 'VARCHAR(25)', 'test2': 'VARCHAR(25)', 'PRIMARY KEY': 'test2'})

    data = {'test2': ['aaaa', 'sasa', 'sadsa']
            }

    insert_table('testtable', data)
