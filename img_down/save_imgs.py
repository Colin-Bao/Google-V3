"""
读取article中的推文信息的在线url,保存到本地
"""
import numpy as np
import pandas as pd
import requests
import os
from tqdm.auto import tqdm
from img_down import __config as gv

from log_rec.log import Logger

logger = Logger(logname=gv.LOG_PATH + __name__ + '.log', loggername=__name__).getlog()


# 按照url下载图片 需要提前建立好公众号的子目录 所有封面放在cover_imgs下面
def down_from_url(biz_name, artical_name, img_url):
    try:
        r = requests.get(img_url, stream=True)
        if r.status_code == 200:

            # 相对路径
            local_path_save = gv.LOCAL_COVER_PATH + str(biz_name) + '/' + str(
                artical_name) + '.jpg'

            # 判断该文件是否在本地存在
            # if os.path.exists(local_path_save):
            #     logger.warn('图像{0}已在本地存在'.format(img_url))
            #     return None

            # 绝对路径
            local_path_read = gv.LOCAL_ROOT_PATH + local_path_save

            open(local_path_read, 'wb').write(r.content)  # 将内容写入图片

            # 在数据库存入相对路径
            return local_path_save
        else:
            return None

    except Exception as e:
        logger.error(e)
        return None


# 按照公众号顺序下载图片
def get_gzh_list() -> pd.DataFrame:
    from tools import mysql_dao
    return mysql_dao.select_table('gzhs', ['biz'])


# 外部调用的控制的方法
# if __name__ == '__main__':
# do_save_cover_imgs(['MjM5MzMwNjM0MA=='])


def save_insert_img(biz_name, start_ts, end_ts):
    from tools import mysql_dao
    # 合并查询img表中没有本地路径的图片
    # 不用对应好id,直接把img中没有的id从article从下载
    # 左表是articles 右表是img
    left_join_query = (
        "SELECT articles.biz,articles.id,articles.cover,articles.mov,article_imgs.local_cover FROM articles "
        "LEFT JOIN article_imgs "
        "ON articles.id = article_imgs.id "
        "WHERE article_imgs.local_cover IS NULL AND "
        "articles.biz = %s AND "
        "articles.p_date BETWEEN %s AND %s ")

    # 执行cursor_query 按照公众号名称biz查询
    df = mysql_dao.excute_sql(left_join_query, 'one', (biz_name, start_ts, end_ts))

    # 获取总进度
    # from log_rec import bar
    # bar = bar.Bar('DownLoad IMG {0}'.format(biz_name), df.shape[0]).get_bat()

    # 继续在公众号的文章列表中循环
    if not df.empty:
        def my_function(x: pd.Series):
            # 先保存
            df_x = pd.DataFrame.transpose(
                pd.concat([x[['id', 'mov']],
                           pd.Series({"local_cover": down_from_url(x['biz'], x['id'],
                                                                   x['cover'])})]).to_frame())
            # 再插入数据库
            mysql_dao.insert_table('article_imgs', df_x, check_flag=False)
            # 更新进度
            # bar.update(1)
            # return df_x

        # 继续在公众号的文章列表中循环
        # df = pd.DataFrame(df)
        tqdm.pandas(desc='DownLoad IMG {0}'.format(biz_name))
        df[['biz', 'id', 'cover', 'mov']].progress_apply(lambda x: my_function(x), axis=1)

        #  循环结束

    #     循环结束
    # bar.update(bar.total - bar.n)


def del_from_table():
    from tools import mysql_dao
    mysql_dao.excute_sql('DELETE FROM article_imgs WHERE local_cover is NULL')


def start_download():
    # 先删除img表中的空项目
    del_from_table()

    # 再插入新的img
    get_gzh_list()[['biz']].apply(lambda x: save_insert_img(x['biz'], gv.START_TS, gv.END_TS), axis=1)
