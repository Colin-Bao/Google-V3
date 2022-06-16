# 全局时间变量


# 3.把article的本地封面图片下载到新表article_img
def run_wc_img_info():
    from wc_img_info import save_cover_imgs
    save_cover_imgs.do_save_cover_imgs()


# 4.计算article_img中的图像情绪
# 暂时只有封面图像情绪
def run_img_sentiment():
    from img_sentiment import predict_imgsent_fromdb
    predict_imgsent_fromdb.predict_by_batch(512)


# 5.把article的日期映射到交易日期
def run_map_date():
    from map_date import map_article_tdate
    map_article_tdate.start_map_tdate()


if __name__ == '__main__':
    # 1.爬取数据
    # 2.同步数据库
    print("请在Navicat中传输更新sqlite数据库")

    # 3.(已测试)把article的本地封面图片下载到新表article_img()
    # run_wc_img_info()

    # 4.(已测试)计算article_img中的图像情绪
    # run_img_sentiment()

    # 5.(已测试)把article的日期映射到交易日期
    run_map_date()

    # 6.每家公众号按照日期聚合article,并计算聚合后的指标

    # 7.合并聚合后的group与金融市场数据

    # 8.在Stata中重新排列时间序列,计算滞后变量回归
