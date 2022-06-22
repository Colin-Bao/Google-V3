# 全局时间变量
def run_log():
    pass


# 3.把article的本地封面图片下载到新表article_img
def run_wc_img_info():
    from wc_img_info import save_cover_imgs
    save_cover_imgs.start_download()


# 4.计算article_img中的图像情绪
# 暂时只有封面图像情绪
def run_img_sentiment():
    from img_sentiment import predict_imgsent_fromdb
    predict_imgsent_fromdb.start_predict()


# 5.把article的日期映射到交易日期
def run_map_date():
    from map_date import do_map_date
    do_map_date.start()


# 5.5下载和计算金融市场数据
def run_fin_data():
    # 建表
    from financial_data import do_fin_data
    do_fin_data.start()


# 6.每家公众号按照日期聚合article,并计算聚合后的指标
def run_group_date():
    from group_date import group_gzh_imgs
    group_gzh_imgs.start_group()


# 7.合并聚合后的group与金融市场数据


def main():
    # 0.记录日志
    import logging
    logging.basicConfig(filename='global.log', encoding='utf-8', level=logging.INFO,
                        )

    # 1.爬取数据
    # 2.同步数据库
    logging.info('2.同步数据库 请在Navicat中传输更新sqlite数据库')

    # 3.(已测试)把article的本地封面图片下载到新表article_img()
    logging.info('3.(已测试)把article的本地封面图片下载到新表article_img()')
    run_wc_img_info()

    # 4.(已测试)计算article_img中的图像情绪
    # 存在0和1非常多的问题
    logging.info('4.(已测试)计算article_img中的图像情绪')
    run_img_sentiment()

    # 5.(已测试)把article的日期映射到交易日期
    logging.info('5.(已测试)把article的日期映射到交易日期')
    run_map_date()

    # 5.5下载和计算金融市场数据
    logging.info('5.5下载和计算金融市场数据')
    run_fin_data()

    # 6.(已测试)每家公众号按照日期聚合article,并计算聚合后的指标,再与金融市场数据连接
    logging.info('6.(已测试)每家公众号按照日期聚合article,并计算聚合后的指标,再与金融市场数据连接')
    run_group_date()

    # 8.在Stata中重新排列时间序列,计算滞后变量回归


if __name__ == '__main__':
    main()
