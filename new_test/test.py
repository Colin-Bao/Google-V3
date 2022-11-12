class DownLoader:
    DOWNLOAD_PATH_ROOT = f'/Users/mac/Downloads/load_img/'

    def __init__(self):
        from sqlalchemy import create_engine
        self.sqlite_path = 'sqlite:////Users/mac/PycharmProjects/wcplusPro7.31/db_folder/data-dev.db'
        self.engine = create_engine(self.sqlite_path, echo=False, connect_args={'check_same_thread': False})
        self.df_select = None

    def down_by_gzh(self, biz='MjM5MzMwNjM0MA=='):
        """
        按照公众号名称,分片下载
        :param biz:
        :return:
        """
        import pandas as pd
        import requests
        from tqdm.auto import tqdm
        import os

        # 提取
        self.df_select = pd.read_sql(
            "SELECT id,cover,cover_local FROM articles_copy1 "
            "WHERE biz=:biz AND mov=:mov AND p_date BETWEEN :sd AND :ed  AND cover_local IS NULL ",
            con=self.engine, params={'biz': biz,
                                     'sd': int(pd.to_datetime('20200101').timestamp()),
                                     'ed': int(pd.to_datetime('20210101').timestamp()),
                                     'mov': 10},
            parse_dates=["p_date"], )

        # 下载图片
        load_path = self.DOWNLOAD_PATH_ROOT + f'{biz}/'
        os.makedirs(load_path, exist_ok=True)

        def down_url(x):
            with open(load_path + f"{x['id']}.jpeg", 'wb') as f:
                if f.write(requests.get(x['cover'], stream=True).content):
                    return load_path + f"{x['id']}.jpeg"
            return None

        tqdm.pandas(desc=f'DownLoad IMG By {biz}')
        self.df_select['cover_local'] = self.df_select[['id', 'cover']].progress_apply(lambda x: down_url(x), axis=1)
        self.df_select.to_sql('temp', self.engine, index=False, if_exists='replace')

        # 更新下载完的图片
        sql = """
        UPDATE articles_copy1 AS tar
        SET cover_local = (SELECT temp.cover_local FROM temp WHERE temp.id = tar.id)
        WHERE EXISTS(SELECT id,cover_local FROM temp WHERE temp.id = tar.id)
        """
        self.engine.execute(sql)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()


# DownLoader().down_by_gzh()
with DownLoader() as DownLoader:
    DownLoader.down_by_gzh()

#     DownLoader.down_by_gzh()
