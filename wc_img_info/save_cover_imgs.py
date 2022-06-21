"""
读取article中的推文信息的在线url,保存到本地
"""
import time
from functools import wraps

import pandas as pd
import requests
import mysql.connector
import os
from datetime import datetime, date


# 用于计算程序运行时间
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


# 与数据库建立连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# date类型转ts
def date_to_ts(date_type):
    dt = datetime.combine(date_type, datetime.min.time())
    # datetime.fromtimestamp(p_date)
    return int(dt.timestamp())


# 按照url下载图片 需要提前建立好公众号的子目录 所有封面放在cover_imgs下面
def down_from_url(biz_name, artical_name, img_url):
    try:
        r = requests.get(img_url, stream=True)
        if r.status_code == 200:
            local_path = 'cover_imgs/' + str(biz_name) + '/' + str(
                artical_name) + '.jpg'
            open(local_path, 'wb').write(r.content)  # 将内容写入图片
            return local_path
        else:
            return None

    except BaseException as e:
        print(e)
        return None


# 按照公众号顺序下载图片
def get_gzh_list() -> pd.DataFrame:
    from my_tools import mysql_dao
    return mysql_dao.select_table('gzhs', ['biz'])


@timethis
# 从数据库中按照filter_biz filter_date条件筛选并下载到本地,更新article_imgs表
def down_img_url(filter_biz, filter_date):
    # date类型转ts
    p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

    # INSERT IGNORE INTO
    def insert_article_img(i_biz, i_id, i_mov):
        local_cover = 'cover_imgs/' + str(i_biz) + '/' + str(i_id) + '.jpg'
        insert_sql = (
            "INSERT IGNORE INTO article_imgs"
            " (id,local_cover,mov) "
            "VALUES (%s,%s,%s) ")
        cur_insert = cnx.cursor(buffered=True)
        cur_insert.execute(insert_sql, (i_id, local_cover, i_mov))
        cnx.commit()

    # 查询返回结果的时候同步进行图片下载
    # 查询有id对应的ulrpath,但是在img中没有的id
    def select_from_article():
        cursor_query = cnx.cursor(buffered=True)
        # 合并查询img表中没有本地路径的图片
        # 不用对应好id,直接把img中没有的id从article从下载
        # 左表是articles 右表是img
        left_join_query = ("SELECT articles.biz,articles.id,articles.cover,articles.mov FROM articles "
                           "LEFT JOIN article_imgs "
                           "ON articles.id = article_imgs.id "
                           "WHERE article_imgs.id IS NULL AND "
                           "articles.biz = %s AND "
                           "articles.p_date BETWEEN %s AND %s ")

        # 执行cursor_query 按照公众号名称biz查询
        cursor_query.execute(left_join_query, (filter_biz, p_start_ts, p_end_ts))

        # 图片按照文章id命名
        print('插入article_imgs记录条数:', cursor_query.rowcount)

        # 返回按照biz和date筛选得到的结果
        for (biz, f_id, cover, mov) in cursor_query:
            # 执行图片下载操作
            down_from_url(biz, f_id, cover)
            insert_article_img(biz, f_id, mov)

    # 建立数据库连接
    cnx = conn_to_db()
    # 检索封面图像路径并下载
    select_from_article()
    # 关闭游标和连接
    cnx.close()


# 外部调用的控制的方法

def do_save_cover_imgs(gzh_list=None):
    if gzh_list is None:
        gzhs_list = get_gzh_list()
        # 以公众号分片下载图片
        for biz in gzhs_list:
            dir_path = 'cover_imgs/' + biz[0]

            # 本地创建一个目录存储公众号所有图片
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # 在该目录下进行下载
            down_img_url(biz[0], [date(2021, 6, 1), date(2022, 6, 1)])

    else:
        for gzh_biz in gzh_list:
            dir_path = 'cover_imgs/' + gzh_biz

            # 本地创建一个目录存储公众号所有图片
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # 在该目录下进行下载
            down_img_url(gzh_biz, [date(2021, 6, 1), date(2022, 6, 1)])


# if __name__ == '__main__':
# do_save_cover_imgs(['MjM5MzMwNjM0MA=='])


def save_insert_img(biz_name, start_ts, end_ts):
    from my_tools import mysql_dao
    # 合并查询img表中没有本地路径的图片
    # 不用对应好id,直接把img中没有的id从article从下载
    # 左表是articles 右表是img
    left_join_query = ("SELECT articles.biz,articles.id,articles.cover,articles.mov FROM articles "
                       "LEFT JOIN article_imgs "
                       "ON articles.id = article_imgs.id "
                       "WHERE article_imgs.id IS NULL AND "
                       "articles.biz = %s AND "
                       "articles.p_date BETWEEN %s AND %s ")

    # 执行cursor_query 按照公众号名称biz查询
    df = mysql_dao.excute_sql(left_join_query, 'one', (biz_name, start_ts, end_ts))
    if not df.empty:
        def my_function(x: pd.Series):
            df_x = pd.DataFrame.transpose(
                pd.concat([x[['id', 'mov']],
                           pd.Series({"local_cover": down_from_url(x['biz'], x['id'],
                                                                   x['cover'])})]).to_frame())
            mysql_dao.insert_table('article_imgs', df_x, check_flag=False)
            # return df_x

        df[['biz', 'id', 'cover', 'mov']].apply(lambda x: my_function(x), axis=1)

    # print(df.columns)
    # 图片按照文章id命名


def start_download():
    for i in get_gzh_list().values:
        save_insert_img(i[0], '1621000000', '1654012800')
