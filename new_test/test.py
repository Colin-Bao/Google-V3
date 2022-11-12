import pandas as pd
import multiprocessing as mp
import numpy as np


class DownLoader:
    DOWNLOAD_PATH_ROOT = f'/Users/mac/Downloads/load_img/'

    def __init__(self):
        from sqlalchemy import create_engine
        self.sqlite_path = 'sqlite:////Users/mac/PycharmProjects/wcplusPro7.31/db_folder/data-dev.db'
        self.engine = create_engine(self.sqlite_path, echo=False, connect_args={'check_same_thread': False})
        self.df_select = None

    def down_by_gzh(self, biz='MjM5MzMwNjM0MA=='):
        """
        按照公众号名称,分片下载\n
        :param biz:
        :return:
        """

        # 提取
        def extract():
            self.df_select = pd.read_sql(
                "SELECT id,cover,cover_local FROM articles_copy1 "
                "WHERE biz=:biz AND mov=:mov AND p_date BETWEEN :sd AND :ed  AND cover_local IS NULL",
                con=self.engine, params={'biz': biz,
                                         'sd': int(pd.to_datetime('20200101').timestamp()),
                                         'ed': int(pd.to_datetime('20210101').timestamp()),
                                         'mov': 10},
                parse_dates=["p_date"], )
            return self.df_select

        def down(df_extract):
            import requests
            import os

            # 下载图片
            load_path = self.DOWNLOAD_PATH_ROOT + f'{biz}/'
            os.makedirs(load_path, exist_ok=True)

            # self.df_select['cover_local'] = self.df_select[['id', 'cover']].progress_apply(lambda x: down_url(x),
            # 下载图片
            def down_url(x):
                with open(load_path + f"{x['id']}.jpeg", 'wb') as f:
                    if f.write(requests.get(x['cover'], stream=True).content):
                        return load_path + f"{x['cover']}.jpeg"

            import dask.dataframe as dd
            from dask.diagnostics import ProgressBar
            ddata = dd.from_pandas(df_extract, npartitions=5)
            with ProgressBar():
                df_extract['cover_local'] = ddata.map_partitions(lambda df: df.apply(lambda row: down_url(row), axis=1),
                                                                 meta=df_extract.dtypes).compute(scheduler='processes')

            return df_extract

        down(extract())
        # 更新下载完的图片
        # self.update_by_temp(self.df_select, 'articles_copy1', 'cover_local', 'id')

    def update_by_temp(self, df_temp: pd.DataFrame, update_table, update_column, update_pk):
        """
        生成中间表来更新\n
        :param df_temp:
        :param update_table:
        :param update_column:
        :param update_pk:
        :return:
        """
        update_table_temp = update_table + '_temp'
        df_temp.to_sql(update_table_temp, self.engine, index=False, if_exists='replace')
        sql = f"""
                UPDATE {update_table} AS tar
                SET {update_column} = (SELECT temp.{update_column} FROM {update_table_temp} AS temp WHERE temp.{update_pk} = tar.{update_pk})
                WHERE EXISTS(SELECT {update_pk},{update_column} FROM {update_table_temp} AS temp WHERE temp.{update_pk} = tar.{update_pk})
                """
        self.engine.execute(sql)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()


if __name__ == '__main__':
    with DownLoader() as DownLoader:
        DownLoader.down_by_gzh()

# DownLoader().down_by_gzh()


#     DownLoader.down_by_gzh()
