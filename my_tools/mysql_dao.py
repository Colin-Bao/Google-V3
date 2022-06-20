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
            return cur
        else:
            # 查询sql
            cur.execute(sql)

            return tools.query_to_df(cur)
    except mysql.connector.Error as e:
        print(e)
    finally:
        cur.close()
        cnx.close()


# 查询表格所有字段并返回
def select_columns(table_name: str):
    def excute():
        sql = (
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE table_name = {0}".format('\'' + table_name + '\'')
        )
        return excute_sql(sql)['COLUMN_NAME']

    return excute().tolist()


# 增加表格的字段
# 传入字段列表,默认为float
def alter_table(table_name: str, column_list: list):
    # 解析column_dict
    def transform_dict():
        # 已经有的更新
        column_old = select_columns(table_name)
        column_new = [i for i in column_list if i not in column_old]

        if column_new is not None:
            # 生成sql
            colunm_list = ['ADD COLUMN ' + '`' + i + '`' + ' ' + 'FLOAT' for i in column_new]
            colunm_list = ','.join(colunm_list)
            return colunm_list

    def excute():
        add_column = transform_dict()
        # 如果该列不存在默认增加VARCHAR
        if add_column:
            sql = ("ALTER TABLE {0}".format('`' + table_name + '`' + ' ') +
                   "{0}".format(add_column)
                   )
            excute_sql(sql)

    excute()


# 创建一个表格,传入表格名与字段名和类型,主键
# 例如
# create_table('testtable', {'test1': 'VARCHAR(25)', 'test2': 'VARCHAR(25)', 'PRIMARY KEY': 'test2'})
def create_table(table_name: str, column_dict: dict):
    # 解析column_dict
    def transform_dict():
        # 主键处理
        pk_flag = 'PK'

        colum_list = ['' if i == pk_flag else i + ' ' + j for i, j in column_dict.items()]

        # 主键追加NOT NULL
        if pk_flag in column_dict.keys():
            pk_column = column_dict[pk_flag]
            column_dict[pk_column] = column_dict[pk_column] + ' NOT NULL'
            colum_list += ['PRIMARY KEY ' + '(`' + column_dict[pk_flag] + '`)']

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
        # 如果没有该collumn则创建一个
        alter_table(table_name, df.columns.tolist())

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

        # 如果没有该collumn则创建一个
        alter_table(table_name, df.columns.tolist())

        # update_str
        update_str = ['`' + i + '`' + '=%s' for i in df.columns]
        # 切分where
        where_str = update_str[-1:]
        where_str = ','.join(where_str)

        update_str = update_str[:-1]
        update_str = ','.join(update_str)

        # tup_values
        values_tup = tools.df_to_tup(df)

        # 默认最后一列为where列

        return update_str, where_str, values_tup

    def execute():
        # 获取要插入的数据
        update_str, where_str, tups = transform_df()

        # 创建sql语句
        sql = ("UPDATE " + '`' + table_name + '` ' +
               "SET " + update_str + " " +
               "WHERE " + where_str)
        # print(sql)
        # 执行
        excute_sql(sql, tups)

    execute()


if __name__ == '__main__':
    # create_table('testtable2', {'test1': 'VARCHAR(25)', 'test2': 'VARCHAR(25)', 'PK': 'test1'})
    #
    # data = {'test1': ['21', '21', '21'], 'test1111': ['1', '2', '3']
    #         }
    # insert_table('testtable2', data)

    data = {'test1111': ['111', '222', '333'], 'test1': ['1', '2', '3']
            }
    update_table('testtable2', data)
    # print(select_columns('testtable2'))
    # alter_table('testtable2', ['testfloat', ])
