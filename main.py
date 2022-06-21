# 全局时间变量


# 3.把article的本地封面图片下载到新表article_img
def run_wc_img_info():
    print('3.把article的本地封面图片下载到新表article_img')
    from wc_img_info import save_cover_imgs
    save_cover_imgs.do_save_cover_imgs()


# 4.计算article_img中的图像情绪
# 暂时只有封面图像情绪
def run_img_sentiment():
    print('4.计算article_img中的图像情绪')
    from img_sentiment import predict_imgsent_fromdb
    predict_imgsent_fromdb.predict_by_batch(512)


# 5.把article的日期映射到交易日期
def run_map_date():
    print('5.把article的日期映射到交易日期')
    from map_date import do_map_date
    do_map_date.run_self()


# 5.5下载和计算金融市场数据
def run_fin_data():
    print('5.5下载和计算金融市场数据')
    # 建表
    from financial_data import load_fin_data
    download_financial_data.start_download()
    from financial_data import cal_fin_data
    cal_fin_data.start_cal()


# 6.每家公众号按照日期聚合article,并计算聚合后的指标
def run_group_date():
    print('6.每家公众号按照日期聚合article,并计算聚合后的指标')
    from group_date import group_gzh_imgs
    group_gzh_imgs.start_group_by_date()


# 7.合并聚合后的group与金融市场数据

if __name__ == '__main__':
    # 1.爬取数据
    # 2.同步数据库
    print("请在Navicat中传输更新sqlite数据库")

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

    # 8.在Stata中重新排列时间序列,计算滞后变量回归
