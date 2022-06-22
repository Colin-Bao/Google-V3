def del_logs():
    PATH = '/Users/mac/PycharmProjects/Google-V3/log_rec/log_file'
    import shutil
    import os
    try:
        shutil.rmtree(PATH)
        os.mkdir(PATH)
    except Exception as e:
        print(e)


# 3.把article的本地封面图片下载到新表article_img
def run_wc_img_info():
    # logger.info('把article的本地封面图片下载到新表article_img')
    # logging.logger.info('3.(已测试)把article的本地封面图片下载到新表article_img()')
    from img_down import save_imgs
    save_imgs.start_download()


# 4.计算article_img中的图像情绪
# 暂时只有封面图像情绪
def run_img_sentiment():
    # logging.logger.info('4.计算article_img中的图像情绪')
    from img_predict import predict_sent
    predict_sent.start_predict()


# 5.把article的日期映射到交易日期
def run_map_date():
    # logging.logger.info('5.把article的日期映射到交易日期')
    from date_map import do_map
    do_map.start()


# 5.5下载和计算金融市场数据
def run_fin_data():
    # logging.logger.info('5.5下载和计算金融市场数据')
    # 建表
    from data_down import do_findata
    do_findata.start()


# 6.每家公众号按照日期聚合article,并计算聚合后的指标
def run_group_date():
    # logging.logger.info('6.每家公众号按照日期聚合article,并计算聚合后的指标')
    from date_group import group_imgs
    group_imgs.start_group()


# 7.合并聚合后的group与金融市场数据


def run_check_date():
    from data_clean import check_null
    # check_null.start_check()


if __name__ == '__main__':
    del_logs()
    # logging.basicConfig(filename='sqldao.log', encoding='utf-8', level=logging.INFO, filemode='w')
    # 创建控制台handler
    # 创建logger实例

    # 0.记录日志
    # run_log()
    # 1.爬取数据

    # 2.同步数据库

    # 3.(已测试)把article的本地封面图片下载到新表article_img()

    run_wc_img_info()

    # 4.(已测试)计算article_img中的图像情绪
    # 存在0和1非常多的问题
    run_img_sentiment()

    # 5.(已测试)把article的日期映射到交易日期
    run_map_date()

    # 5.5下载和计算金融市场数据
    run_fin_data()

    # 6.(已测试)每家公众号按照日期聚合article,并计算聚合后的指标,再与金融市场数据连接
    run_group_date()

    run_check_date()

    # 8.在Stata中重新排列时间序列,计算滞后变量回归
