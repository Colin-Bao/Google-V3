import time
from datetime import datetime, date
from PIL import Image
import tushare_api
import os
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras import callbacks
from keras_visualizer import visualizer
import matplotlib.pyplot as plt

import PIL
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adagrad
from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras_visualizer import visualizer
from keras import layers
from keras.models import load_model
from keras.preprocessing import image


# 初始化
def init_settings():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 对需要进行限制的GPU进行设置
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=4096)])


# 生成数据集
def gen_data():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ((x/255)-0.5)*2  归一化到±1之间
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(directory='./twitter2/train',
                                                        target_size=(299, 299),  # Inception V3规定大小
                                                        batch_size=32)
    val_generator = val_datagen.flow_from_directory(directory='./twitter2/validation',
                                                    target_size=(299, 299),
                                                    batch_size=32)

    test_generator = test_datagen.flow_from_directory(directory='./twitter2/test',
                                                      target_size=(299, 299),
                                                      batch_size=32)

    return train_generator, val_generator, test_generator


# 开始迁移学习
def transfer_learning(epoch_num):
    train_generator, val_generator, test_generator = gen_data()
    # 构建基础模型
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # 冻结基础模型
    for layer in base_model.layers:
        layer.trainable = False

    # 增加新的输出层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # 编译模型
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 开始训练
    remote = callbacks.RemoteMonitor(root='http://localhost:9000')
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=int(617 / 32),  # 800
                                  epochs=epoch_num,  # 2
                                  validation_data=val_generator,
                                  validation_steps=int(176 / 32),
                                  workers=8,
                                  callbacks=[remote]

                                  )
    # 存储模型
    model.save('twitter_tl_' + str(epoch_num) + '.h5')

    # 在测试集上预测
    predit_res = model.predict_generator(generator=test_generator)

    pd.DataFrame(predit_res).to_csv('result.csv')

    # 可视化结果
    def show_result(test_generator, predit_res):
        def plot_images(images, cls_true, cls_pred=None, smooth=True):
            class_names = [0, 1]
            assert len(images) == len(cls_true)

            # Create figure with sub-plots.
            fig, axes = plt.subplots(3, 3)

            # Adjust vertical spacing.
            if cls_pred is None:
                hspace = 0.3
            else:
                hspace = 0.6
            fig.subplots_adjust(hspace=hspace, wspace=0.3)

            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            for i, ax in enumerate(axes.flat):
                # There may be less than 9 images, ensure it doesn't crash.
                if i < len(images):
                    # Plot image.
                    ax.imshow(plt.imread(images[i]),
                              interpolation=interpolation)

                    # Name of the true class.
                    cls_true_name = class_names[cls_true[i]]

                    # Show true and predicted classes.
                    if cls_pred is None:
                        xlabel = "True: {0}".format(cls_true_name)
                    else:
                        # Name of the predicted class.
                        cls_pred_name = cls_pred[i]
                        pre_0, prq_1 = cls_pred_name
                        pre_0 = round(float(pre_0), 2)
                        prq_1 = round(float(prq_1), 2)

                        # pre_0
                        label_str = "Pos:{0} \n Neg:{1}".format(pre_0, prq_1)

                        xlabel = label_str

                    # Show the classes as the label on the x-axis.
                    ax.set_xlabel(xlabel)

                # Remove ticks from the plot.
                ax.set_xticks([])
                ax.set_yticks([])

            # Ensure the plot is shown correctly with multiple plots
            # in a single Notebook cell.
            plt.show()

        # predit_res = pd.DataFrame(predit_res).to_numpy()
        images = test_generator.filepaths[80:89]
        cls_true = test_generator.classes[80:89]

        # plt.imshow(plt.imread(images[1]))
        # plt.show()
        # df_res = pandas.DataFrame(predit_res)

        plot_images(images, cls_true, cls_pred=predit_res[80:89])

    show_result(test_generator, predit_res)


# 传入模型并且预测结果,但是得到的顺序是错的
def load_and_predict(h_path):
    model = load_model(h_path)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    test_generator = test_datagen.flow_from_directory(directory='./twitter2/test', batch_size=32)

    # 批量预测的结果
    print('--------批量预测的结果------------')
    predit_res = model.predict_generator(test_generator)
    print(test_generator.filepaths[0])
    print(predit_res[0])

    print('--------单张预测的结果------------')
    img = image.load_img(test_generator.filepaths[0])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))
    print('--------------------')
    # predit_res = model.predict(test_generator)
    # print(test_generator.filepaths[0])
    # print(predit_res[0])
    # print(model.predict(generator=test_generator))
    time.sleep(11111)

    # predit_res = model.predit_res(generator=test_generator)

    # ev_res = model.evaluate_generator(generator=test_generator)
    # 把predit_res返回的结果和test_generator路径拼在一起看看结果
    df_res = pd.DataFrame(predit_res)
    df_img = pd.DataFrame(test_generator.filepaths)
    df_con = pd.concat([df_res, df_img], axis=1, ignore_index=True)

    df_con.to_csv('result_img.csv')


# 重新遍历预测结果,得到的结果是对的
# 传入的图片路径是一个可迭代的对象
def load_and_predict_img(h_path, img_path):
    # 忽略硬件加速的警告信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # df_img = pd.read_csv('result_img.csv')['2']
    df_img = img_path
    # print(df_img)
    # df_res = pd.DataFrame()
    # 把图片读取出来放到列表中
    img = []
    # ran_img = range(len(df_img))
    for i in range(len(df_img)):
        # print(df_img[i])
        images = image.load_img(df_img[i], target_size=(299, 299))
        x = image.img_to_array(images)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img.append(x)
        # print(x)
        # print('loading no.%s image' % i)
    # 把图片数组联合在一起
    x = np.concatenate([x for x in img])

    model = load_model(h_path)
    y_pred = pd.DataFrame(model.predict(x))
    df_c = pd.concat([y_pred, df_img], axis=1)
    df_c = df_c.rename(columns={0: 'neg', 1: 'pos', '2': 'path'})
    df_c.to_csv('test_batch.csv')

    # pd.DataFrame(y).to_csv('test_batch.csv')
    #
    # time.sleep(2222)
    # for i in range(89):
    #     img_path = df_img[i]
    #     img = image.load_img(img_path)
    #
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #
    #     neg, pos = model.predict(x)[0]
    #
    #     # df_res = pd.concat([])
    #     # print(neg, pos)
    #
    #     df_res = df_res.append({'img_path': img_path, 'neg': neg, 'pos': pos}, ignore_index=True)
    #
    # df_res.to_csv('img_predict.csv')


# 用于可视化的类
def plot_images(images, cls_true, cls_pred=None, cls_pred2=None, smooth=True, root_dir=''):
    class_names = [0, 1]
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(10, 5)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    # fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            # print(root_dir + images[i])
            # temp_path = '/Users/mac/PycharmProjects/Investor-Sentiment/cover_imgs/MjM5NzQ5MTkyMA==/a65f47f09010b5db1d5b513982c3410e.png'
            # img = Image.open(temp_path)
            # plt.imshow(plt.imread(temp_path))
            # plt.show()
            ax.imshow(Image.open(root_dir + images[i]),
                      interpolation=interpolation)

            # Name of the true class.
            # cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                # xlabel = "True: {0}".format('')
                pass
            else:
                # Name of the predicted class.
                # print(cls_pred[i])
                neg = cls_pred[i]
                pos = cls_pred2[i]

                # pre_0
                label_str = "Pos:{0:.2f} \n Neg:{1:.2f}".format(pos, neg)

                xlabel = label_str

            # Show the classes as the label on the x-axis.
            font_my = {'family': 'Times New Roman',
                       'weight': 'normal', 'size': 8,
                       }
            # ax.set_xlabel(xlabel, font_my)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    # plt.savefig('sinc.png', dpi=300)
    plt.show()


# 在预测结果中按照行X列展示指定行的结果
def show_result(columns, img_path='result_img.csv'):
    df_pre = pd.read_csv(img_path)

    df_pre = df_pre.iloc[columns]
    # df_pre = df_pre.drop(labels=0, axis=1)
    df_pre.index = range(len(df_pre))

    path, neg, pos = df_pre['img_path'], df_pre['neg'], df_pre['pos']
    images = path
    cls_true = path

    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos)


# 展示神经网络的结构
def show_cnn_structure():
    def show_cnn2():
        model1 = models.Sequential()
        model1.add(Conv2D(8, (3, 3), padding="same", input_shape=(299, 299, 3), activation="relu"))
        model1.add(Dense(16, input_shape=(784,)))
        model1.add(Dense(8))
        model1.add(Dense(4))
        visualizer(model1, format='png', view=True)

    def show_cnn():
        # Building model architecture
        model = models.Sequential()
        model.add(Conv2D(8, (3, 3), padding="same", input_shape=(299, 299, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(2))
        model.summary()

        visualizer(model, format='png', view=True)

    show_cnn()
    show_cnn2()


# 迁移学习
# init_settings()
# transfer_learning(1)

#
# load_and_predict_img('twitter_tl_500.h5')

#
# show_result_img('result_img.csv', 0)

# show_result([46, 53, 73, 76, 81, 85, 66, 83, 4, 58, 67, 54, 75, 42, 78, 39])
# show_result([35, 34, 7, 23, 24, 33, 1, 49, 18, 10, 9, 3, 11, 5, 32, 25])
# show_cnn2()
# show_cnn()

# 预测并展示单张图片的预测结果
# 预测单张图像
def predict_img(h_path, img_path):
    plt.rcParams['font.sans-serif'] = ['STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

    model = load_model(h_path)

    img = image.load_img(img_path)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    neg, pos = model.predict(x)[0]
    plt.axis('off')
    plt.imshow(plt.imread(img_path))
    plt.title('开心: {0:.4f} 难过: {1:.4f}'.format(pos, neg), )

    plt.show()
    # print(neg, pos)


# date类型转ts
def date_to_ts(date_type):
    dt = datetime.combine(date_type, datetime.min.time())
    # datetime.fromtimestamp(p_date)
    return int(dt.timestamp())


# 获取数据库连接
def conn_to_db():
    return mysql.connector.connect(user='root', password='',
                                   host='127.0.0.1',
                                   database='wechat_offacc')


# 用于在数据库中的路径进行情绪预测
def predict_imgsent_from_db():
    # 传入图片路径,返回可以预测的x
    def filepath_to_img(df_img, root_path):
        img = []
        # ran_img = range(len(df_img))
        for i in range(len(df_img)):
            try:
                images = image.load_img(root_path + df_img[i], target_size=(299, 299))
                x = image.img_to_array(images)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                img.append(x)
                # print(x)
                print('loading no.%s image' % i)

            except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
                print(e, root_path + df_img[i])
                continue

        # 把图片数组联合在一起
        x = np.concatenate([x for x in img])
        return x

    # 根据x计算预测值 和id,path,预测值,拼在一起 返回df
    def predict_img(x, model_path):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        model = load_model(model_path)
        y_pred = pd.DataFrame(model.predict(x))
        return y_pred

        # 与数据库建立连接

    # 根据返回的图像路径进行情绪计算
    def cal_img_sentiment(df_query):
        # 图片路径读取成可以预测的格式
        # 第2列是img路径
        x = filepath_to_img(df_query[1], '/Users/mac/PycharmProjects/Investor-Sentiment/')

        # 预测
        y = predict_img(x, 'twitter_tl_500.h5')

        # 预测结果与原表拼在一起
        #  id path neg pos
        df_c = pd.concat([df_query, y], axis=1)

        # 返回新增了neg pos的新df
        return df_c

    # 创建字段
    def create_attr():
        create_download_flag = (
            "alter table articles "
            "add (cover_neg float, cover_pos float) "
        )
        cursor_ca = cnx.cursor(buffered=True)
        cursor_ca.execute(create_download_flag)
        cnx.commit()
        cursor_ca.close()

    # 查询数据库中存在的文件路径 且不为空的地方
    def select_pic_path(filter_date):
        cursor_query = cnx.cursor(buffered=True)
        query = ("SELECT id,local_cover FROM articles "
                 "WHERE local_cover IS NOT NULL AND "
                 "cover_neg IS NULL AND "
                 "p_date BETWEEN %s AND %s "
                 "LIMIT 512")
        # 执行cursor_query 按照路径不为空的表提取情绪
        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])
        cursor_query.execute(query, (p_start_ts, p_end_ts))

        # 对返回的cursor_query中的记录进行处理
        # 按照id处理并转换为dfx
        print('查询到记录条数:', cursor_query.rowcount)
        if cursor_query.rowcount == 0:
            return 0
            # 切片
        # df_sql = pd.DataFrame(cursor_query)
        # df_sql = df_query[:512, :]
        return cursor_query.rowcount, pd.DataFrame(cursor_query)

    # 更新情绪数据(单条)
    def update_sent_todb(id, neg, pos):
        update_old_sent = (
            "UPDATE articles SET cover_neg = %s , cover_pos = %s "
            "WHERE id = %s ")
        # print(update_old_flag)
        curA = cnx.cursor(buffered=True)
        try:
            curA.execute(update_old_sent, (neg, pos, id))
            cnx.commit()
        except mysql.connector.Error as err:
            print("Failed creating database: {}".format(err))
        finally:
            curA.close()

    # 更新情绪
    def update_sent(df_con):
        # 只要0,2,3 id neg pos
        df_con = df_con.iloc[:, [2, 3, 0]]

        # 转成元组方便mysql插入
        merge_result_tuples = [tuple(xi) for xi in df_con.values]
        # print(merge_result_tuples)

        # 更新语句 按照id更新
        update_old_sent = (
            "UPDATE articles SET cover_neg = %s , cover_pos = %s "
            "WHERE id = %s ")

        cur_sent = cnx.cursor(buffered=True)
        try:
            cur_sent.executemany(update_old_sent, merge_result_tuples)
            cnx.commit()
        except mysql.connector.Error as err:
            print("Failed creating database: {}".format(err))
        finally:
            cur_sent.close()

    # 创立需要计算的字段(已运行)
    # create_attr()

    # 按照查询图像路径

    # 记录的数量 分片计算,内存不够

    i = 512
    while i >= 512:
        # 建立连接
        cnx = conn_to_db()

        # 没有查询结果就退出
        if select_pic_path([date(2021, 6, 1), date(2022, 6, 1)]) == 0:
            cnx.close()
            return

        rec_cont, df_query = select_pic_path([date(2021, 6, 1), date(2022, 6, 1)])

        # 根据路径计算情绪 一次计算512条就够了
        df_sentiment = cal_img_sentiment(df_query)

        # 还需要详细的情绪数据,细化的
        # 包括了封面党的位置,颜色,类型标签等数据库,用localurl作为主键和外键

        # 更新情绪数据
        update_sent(df_sentiment)

        # 关闭连接
        cnx.close()

        i = rec_cont


# 从数据库中提取情绪分析的结果
def show_imgsent_from_db():
    # 条件查询
    def select_top_sent(filter_date):
        # 创建游标
        cursor_neg, cursor_pos = cnx.cursor(buffered=True), cnx.cursor(buffered=True)

        # 分开查询
        query_neg = ("SELECT local_cover,cover_neg,cover_pos FROM articles "
                     "WHERE cover_neg IS NOT NULL AND "
                     "p_date BETWEEN %s AND %s "
                     "ORDER BY cover_neg DESC "
                     "LIMIT 128")

        query_pos = ("SELECT local_cover,cover_neg,cover_pos FROM articles "
                     "WHERE cover_pos IS NOT NULL AND "
                     "p_date BETWEEN %s AND %s "
                     "ORDER BY cover_pos DESC "
                     "LIMIT 128")
        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

        # 执行查询语句
        cursor_neg.execute(query_neg, (p_start_ts, p_end_ts))
        cursor_pos.execute(query_pos, (p_start_ts, p_end_ts))

        # 按照id处理并转换为dfx
        print('查询到记录条数:', cursor_neg.rowcount)
        # 切片
        # df_sql = pd.DataFrame(cursor_query)
        # df_sql = df_query[:512, :]
        return pd.DataFrame(cursor_neg), pd.DataFrame(cursor_pos)

    # 建立连接
    cnx = conn_to_db()

    # 按照消极情绪排行查询
    # 筛选公众号+日期+条数
    df_neg, df_pos = select_top_sent([date(2021, 6, 1), date(2022, 6, 1)])

    # 断开连接
    cnx.close()

    # 调用可视化
    df_neg, df_pos = df_neg.iloc[:50, :], df_pos.iloc[:50, :]
    # print(df_neg[0])
    # time.sleep(1111)
    path, neg, pos = df_neg[0], df_neg[1], df_neg[2],
    images = path
    cls_true = path
    # print()

    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos,
                root_dir='/Users/mac/PycharmProjects/Investor-Sentiment/')

    path, neg, pos = df_pos[0], df_pos[1], df_pos[2],
    images = path
    cls_true = path
    plot_images(images, cls_true, cls_pred=neg, cls_pred2=pos,
                root_dir='/Users/mac/PycharmProjects/Investor-Sentiment/')


# 一些用于计算的函数
# 用于join article和article_imgs表,并把每个公众号按照日期聚合得到gzh_imgs_bydate表
# 如果不按照公众号聚合也可以,得到总表
def merge_article_img(filter_biz, filter_date):
    # 先把数据取出来再计算(聚合公众号)
    # 条件查询 合并其他表,更改的时候填写需要改的字段即可
    # 这里是从articles表和article_imgs表连接查找数据
    def select_article_img():
        # 创建游标
        cursor_sent = cnx.cursor(buffered=True)

        # 查询id和用于计算的值
        query_sent = (
            "SELECT articles.id,articles.p_date,articles.t_date,articles.mov,articles.cover_neg,articles.cover_pos "
            "FROM articles INNER JOIN article_imgs on articles.id = article_imgs.id "
            "WHERE articles.biz = %s AND "
            "articles.t_date IS NOT NULL AND "
            "articles.p_date BETWEEN %s AND %s "
            "ORDER BY articles.p_date ASC "
        )

        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

        # 执行查询语句
        cursor_sent.execute(query_sent, (filter_biz, p_start_ts, p_end_ts))

        # 按照id处理并转换为df
        print('查询到记录条数:', cursor_sent.rowcount)

        # cursor_sent.close()

        return cursor_sent.rowcount, pd.DataFrame(cursor_sent)

    # 分析的是article和text表合并的数据
    # 主要是聚合的运算 概率阈值的选择等
    def group_articleimg_date(df):

        # 重命名
        # articles.id,articles.p_date,articles.t_date,articles.mov,articles.cover_neg,articles.cover_pos "
        def rename_column_fromdb():
            df.rename(columns={0: 'id', 1: 'p_date', 2: 't_date', 3: 'mov', 4: 'cover_neg', 5: 'cover_pos'},
                      inplace=True)

        # 一些指标的计算
        def cal_colunms():
            rename_column_fromdb()

            # 增加一列用于把ts换成天数(后续可以精确的时间) 前面的中扩繁相当于传入的参数x
            # 这个函数的作用是聚合,因此不需要time
            df['p_date'] = df[['p_date', ]].apply(lambda x: datetime.fromtimestamp(x['p_date']).date(), axis=1)

            df['t_date'] = df[['t_date', ]].apply(lambda x: datetime.fromtimestamp(x['t_date']).date(), axis=1)
            # df['t_date'] = df[['t_date', ]].apply(
            #     lambda x: datetime.fromtimestamp(x['t_date']).date() if not np.isnan(x['t_date']) else x['t_date'],
            #     axis=1)
            # print(df['t_date'])

            # 聚合操作
            # df_agg_date = df.groupby('date').agg({0: 'count', })
            # transform操作
            # df['id_count'] = df.groupby('date')[0].transform('count')
            df['c_neg_count'] = df[['cover_neg', ]].apply(lambda x: 1 if x['cover_neg'] >= 0.7 else 0, axis=1)
            df['c_pos_count'] = df[['cover_pos', ]].apply(lambda x: 1 if x['cover_pos'] >= 0.7 else 0, axis=1)

        # # 自定义apply函数用于计算概率
        # def count_prob(df_group):
        #     df_group['c_neg_count'] = df[['cover_neg', ]].apply(lambda x: 1 if x['cover_neg'] > 0.7 else 0, axis=1)
        #     df_group['c_pos_count'] = df[['cover_pos', ]].apply(lambda x: 1 if x['cover_pos'] > 0.7 else 0, axis=1)
        #     # print(df_group)
        #     return df_group
        #
        # # 在组中计算概率阈值计数
        # df_g = df.groupby(['p_date'], as_index=False).apply(lambda x: count_prob(x))
        # df_g = df.groupby(['t_date'], as_index=False).apply(lambda x: count_prob(x))

        # 计算完以后按照日期分组并聚合计算 只保留那些需要的列
        # !!!!!实际日期非交易日期

        # 聚合的方式
        def group_articles(column_name='t_date'):
            # 按照t_date分组
            df_group = df.groupby([column_name], as_index=False).agg(
                {'id': 'count', 'cover_neg': 'mean', 'cover_pos': 'mean', 'c_neg_count': 'sum',
                 'c_pos_count': 'sum'})
            # 分组后重命名
            df_group.rename(
                columns={'id': 'article_count', 'cover_neg': 'c_negprob_mean', 'cover_pos': 'c_posprob_mean'},
                inplace=True)

            # 分组后继续计算

            df_group['c_neg_ratio'] = df_group[['article_count', 'c_neg_count']].apply(
                lambda x: x['c_neg_count'] / x['article_count'],
                axis=1)
            df_group['c_pos_ratio'] = df_group[['article_count', 'c_pos_count']].apply(
                lambda x: x['c_pos_count'] / x['article_count'],
                axis=1)

            # 添加额外的列
            df_group['biz'] = filter_biz
            df_group['id_group_date'] = df_group[[column_name, ]].apply(
                lambda x: filter_biz + str(x[column_name]), axis=1)

            # 重新排列
            df_group = df_group[
                ['id_group_date', 'biz', column_name, 'article_count', 'c_negprob_mean', 'c_posprob_mean',
                 'c_neg_count',
                 'c_pos_count', 'c_neg_ratio', 'c_pos_ratio']]

            # df_group.to_csv('test.csv')
            return df_group

        # 更新结果到数据库gzhs_imgs_bydate
        # 主键按照公众号+日期
        def insert_groupbydate_res(df_group, table_name):
            # 'date', 'article_count', 'c_negprob_mean', 'c_posprob_mean',
            # 'c_neg_count', 'c_pos_count', 'c_neg_ratio', 'c_pos_ratio',
            # 'id_group_date', 'biz'

            # 只要0,2,3 id neg pos
            # df_con = df_con.iloc[:, [2, 3, 0]]

            # 转成元组方便mysql插入

            merge_result_tuples = [tuple(xi) for xi in df_group.values]
            # print(merge_result_tuples)

            # 更新语句 按照id更新
            insert_gzhs = (
                    "INSERT IGNORE INTO  " + str(table_name) +
                    " (id_group_date,biz,date,article_count, c_negprob_mean, c_posprob_mean,c_neg_count, c_pos_count, c_neg_ratio, c_pos_ratio) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ")

            cur_sent = cnx.cursor(buffered=True)
            try:
                cur_sent.executemany(insert_gzhs, merge_result_tuples)
                cnx.commit()
            except mysql.connector.Error as err:
                print(err)
            finally:
                cur_sent.close()

        # 计算一些指标
        cal_colunms()

        # 根据t_date聚合
        df_g = group_articles(column_name='t_date')

        # 需要将结果存储到数据库gzhs_imgs_bydate
        insert_groupbydate_res(df_g, 'gzhs_imgs_bytdate')

    # ----------------------- 开始运行主函数-------------------------#
    # 建立连接
    cnx = conn_to_db()

    # 合并其他表
    # 筛选条件按照指定公众号查询
    #
    count_row, df_article_img = select_article_img()

    if count_row == 0:
        return
    else:
        # 数据分析
        group_articleimg_date(df_article_img)

    # 关闭数据库连接
    cnx.close()


# 查询info_date中的日期与交易日期之间的映射关系
# 把article中的p_date映射到t_date,都是ts的数据类型
# 用法 map_articles_tradedate('MjM5NzQ5MTkyMA==', [date(2021, 6, 1), date(2022, 6, 1)])
def map_articles_tradedate(filter_bizname, filter_date):
    # 创建一个表把自然日期映射到交易日期
    def create_info_date():

        # 获得结果用于计算
        def select_info_date():
            # 创建游标
            cursor_s = cnx.cursor(buffered=True)

            # 查询id和用于计算的值
            query_str = (
                "SELECT date_ts "
                "FROM info_date "
                "ORDER BY date_ts ASC "
            )
            cursor_s.execute(query_str)
            return pd.DataFrame(cursor_s)

        def update_info_date(df_info):
            # 先生成字段
            def create_colun():
                # 创建游标
                cursor_s = cnx.cursor(buffered=True)

                # 增加一周中的日期虚拟变量
                alter_str = (
                    "ALTER TABLE info_date "
                    "ADD `monday` int,ADD `tuesday` int,ADD `wednesday` int,"
                    "ADD `thursday` int,ADD `friday` int,ADD `saturday` int"
                )
                cursor_s.execute(alter_str)

            def update():
                # 先获得每周星期几
                df_info['weekday'] = df_info[[0, ]].apply(lambda x: str(datetime.fromtimestamp(x[0]).weekday() + 1),
                                                          axis=1)
                df = df_info[['weekday', 0]]
                merge_result_tuples = [tuple(xi) for xi in df.values]
                # print(merge_result_tuples)

                cursor_s = cnx.cursor(buffered=True)
                # 更新语句 按照id更新
                update_old = (
                    "UPDATE info_date SET weekday = %s "
                    "WHERE date_ts = %s ")
                cursor_s.executemany(update_old, merge_result_tuples)

            update()

        update_info_date(select_info_date())

    # 先从article总表中查找,和infodate匹配后插入交易日期
    def select_article():
        # 创建游标
        cursor_sent = cnx.cursor(buffered=True)

        # 查询id和用于计算的值
        query_str = (
            "SELECT articles.id,articles.p_date "
            "FROM articles "
            "WHERE articles.biz = %s AND "
            "articles.t_date IS NULL AND "
            "articles.p_date BETWEEN %s AND %s "
            "ORDER BY p_date ASC "
        )

        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])

        # 执行查询语句
        cursor_sent.execute(query_str, (filter_bizname, p_start_ts, p_end_ts))

        # 按照id处理并转换为df
        print('查询到记录条数:', cursor_sent.rowcount)

        # cursor_sent.close()

        return cursor_sent.rowcount, pd.DataFrame(cursor_sent)

    # 先从article总表中查找,和infodate匹配后插入交易日期
    def select_infodate():
        # 创建游标
        cursor_sent = cnx.cursor(buffered=True)

        # 查询id和用于计算的值
        query_str = (
            "SELECT nature_date,nature_datetime_ts,day_tradedate,night_tradedate "
            "FROM info_date "
            "WHERE date_ts BETWEEN %s AND %s "
        )

        # date类型转ts
        p_start_ts, p_end_ts = date_to_ts(filter_date[0]), date_to_ts(filter_date[1])
        # 执行查询语句
        cursor_sent.execute(query_str, (p_start_ts, p_end_ts))

        # 按照id处理并转换为df
        print('查询到记录条数:', cursor_sent.rowcount)

        # cursor_sent.close()

        return cursor_sent.rowcount, pd.DataFrame(cursor_sent)

    # 开始匹配
    def map_date(df_article, df_info):
        # 重命名
        df_article.rename(columns={0: 'id', 1: 'datetime_ts'}, inplace=True)
        df_info.rename(columns={0: 'date', 1: 'datetime_ts'}, inplace=True)

        # 从df_article的ts中提取date
        df_article['date'] = df_article[['datetime_ts', ]].apply(
            lambda x: datetime.fromtimestamp(x['datetime_ts']).date(),
            axis=1)

        # join,左表为df_article,用date匹配
        df_con = pd.merge(df_article, df_info, how='left', on=['date'])

        # 匹配以后进行计算
        df_con['article_to_tdate'] = df_con[['datetime_ts_x', 'datetime_ts_y', 2, 3]].apply(
            lambda x: x[2] if x['datetime_ts_x'] <= x['datetime_ts_y'] else x[3], axis=1)

        # df_con.to_csv('test.csv')
        # 转换为ts方便入库
        # 空的日期是因为info_date的交易日期最后一行滞后了
        df_con.dropna(inplace=True)
        # 有空的日期转换不了timestamp,加了一个判断
        df_con['t_date'] = df_con[['article_to_tdate', ]].apply(
            lambda x: pd.to_datetime(x['article_to_tdate']).timestamp(), axis=1)

        # 如果需要检查的时候查看返回值
        # df_con.to_csv('test.csv')

        # 存储结果 筛选一下,换了位置方便数据库插入
        df_con = df_con[['t_date', 'id']]
        return df_con

    # 把匹配后的文章交易日期存储到数据库
    def update_to_article(df):
        # 转成元组方便mysql插入
        merge_result_tuples = [tuple(xi) for xi in df.values]
        # print(merge_result_tuples)

        # 更新语句 按照id更新
        update_old = (
            "UPDATE articles SET t_date = %s "
            "WHERE id = %s ")

        cur_sent = cnx.cursor(buffered=True)
        try:
            cur_sent.executemany(update_old, merge_result_tuples)
            cnx.commit()
        except mysql.connector.Error as err:
            print(err)
        finally:
            cur_sent.close()

    cnx = conn_to_db()
    # 先创建
    # create_info_date()

    # 分别在2张表中查询并返回查询结果
    count_article, df_article = select_article()
    count_infodate, df_infodate = select_infodate()
    if count_article == 0 or count_infodate == 0:
        cnx.close()
        return

    # 匹配好以后返回到articl表
    update_to_article(map_date(df_article, df_infodate))

    cnx.close()


# 获取所有的公众号列表
def select_from_gzhs():
    # 建立数据库连接
    cnx = conn_to_db()
    cursor_query = cnx.cursor(buffered=True)
    query = "SELECT biz FROM gzhs "
    cursor_query.execute(query)
    # 关闭游标和连接
    cnx.close()
    return cursor_query


# 获取金融数据
#
def get_financial_data():
    def get_from_tu(ts_code):
        # 获取K线数据的日期
        tu = tushare_api.TuShareGet('20120101', '20220601')
        # 获取的指数
        df_kline = pd.DataFrame(tu.get_index(ts_code))
        # 转换为dt方便计算
        df_kline['date_ts'] = df_kline[['trade_date', ]].apply(
            lambda x: datetime.strptime(x['trade_date'], '%Y%m%d').timestamp(),
            axis=1)
        # 排序以填充
        df = df_kline.sort_values(by='date_ts')
        # 筛选需要的行
        return df

    def create_table(table_name):
        create_sql = (
                "CREATE TABLE IF NOT EXISTS  " + table_name +
                " (`date_ts` int NOT NULL,`ts_code` varchar(40),`trade_date` varchar(40),"
                "`close` float,`open` float,`high` float,`low` float,`pre_close` float,`change` float,"
                "`pct_chg` float,`vol` float,`amount` float,"
                "PRIMARY KEY (`date_ts`),"
                "KEY `ix_date` (`date_ts`) USING BTREE)"
        )
        cur_create = cnx.cursor()
        cur_create.execute(create_sql)

    # 存储数据到mysql
    def insert_into_fintable(df, table_name):
        # 转成元组方便mysql插入
        result_tuples = [tuple(xi) for xi in df.values]
        # 更新语句 按照id更新
        insert_infodate = (
                "INSERT IGNORE INTO  " + table_name +
                " (ts_code,trade_date,`close`,`open`,high,low,pre_close,`change`,pct_chg,vol,amount, date_ts) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ")
        cur_sent = cnx.cursor(buffered=True)
        try:
            cur_sent.executemany(insert_infodate, result_tuples)
            cnx.commit()
        except mysql.connector.Error as err:
            print(err)
        finally:
            cur_sent.close()

    cnx = conn_to_db()

    index_list = ['399300.SZ', '000001.SH']
    for code in index_list:
        df = get_from_tu(code)
        create_table('`' + code + '`')
        insert_into_fintable(df, '`' + code + '`')

    cnx.close()


# 后面需要计算的字段在这里添加
def cal_other_columns():
    # 金融市场中的表
    def add_return():
        # 取出收盘价
        def select_table():
            table_list = ['`399300.SZ`', '`000001.SH`']
            # 对数收益率
            sql_str = ("SELECT date_ts,pre_close FROM `399300.SZ`"
                       "ORDER BY date_ts ASC "
                       )

            cursor = cnx.cursor()
            cursor.execute(sql_str)

            return pd.DataFrame(cursor)

        # 计算回报
        # 返回可以插入的元组
        def cal_return(df):
            # 重命名方便计算
            df.rename(columns={0: 'date_ts', 1: 'pre_close'}, inplace=True)
            df['pre_close_lag1'] = df['pre_close'].shift(1)
            # ln今日收盘价/昨日收盘价
            df['log_return'] = df[['pre_close', 'pre_close_lag1', ]].apply(
                lambda x: np.log(x['pre_close'] / x['pre_close_lag1']),
                axis=1)
            df['log_return_2'] = df[['log_return', ]].apply(
                lambda x: np.square(x['log_return']),
                axis=1)
            # 删除空行
            df.dropna(inplace=True)

            # 保留插入数据库的列
            df = df[['log_return', 'log_return_2', 'date_ts']]

            # 全部转为str方便插入
            df = df.astype(str)

            return [tuple(xi) for xi in df.values]

        # 更新计算好的数据
        def update_table(data_tup):
            sql_str = (
                "UPDATE `399300.SZ` SET log_return = %s,log_return_2 = %s "
                "WHERE date_ts = %s ")
            cur = cnx.cursor()
            cur.executemany(sql_str, data_tup)
            cnx.commit()

        # 先获得要计算的数据
        res_tuple = cal_return(select_table())
        # 更新计算好的数据
        update_table(res_tuple)

    # 增加时间戳
    def add_date_ts(table_name):
        # 取出表
        def select_table():
            # 对数收益率
            sql_str = ("SELECT id_group_date,date FROM  " + table_name +
                       " ORDER BY date ASC "
                       )

            cursor = cnx.cursor()
            cursor.execute(sql_str)

            return pd.DataFrame(cursor)

        def cal_ts(df):
            df.rename(columns={0: 'id_group_date', 1: 'date'}, inplace=True)
            # 删除空行
            df.dropna(inplace=True)
            # 计算ts
            df['date_ts'] = df[['date', ]].apply(
                lambda x: datetime.strptime(x['date'], '%Y-%m-%d').timestamp(),
                axis=1)

            # 保留插入数据库的列
            df = df[['date_ts', 'id_group_date']]

            # 全部转为str方便插入
            df = df.astype(str)

            return [tuple(xi) for xi in df.values]

        # 更新计算好的数据
        def update_table(data_tup):
            sql_str = (
                    "UPDATE " + table_name +
                    " SET date_ts = %s "
                    "WHERE id_group_date = %s ")
            cur = cnx.cursor()
            cur.executemany(sql_str, data_tup)
            cnx.commit()

        update_table(cal_ts(select_table()))

    cnx = conn_to_db()

    # 已经计算结束
    # add_return()
    add_date_ts('`gzhs_imgs_bytdate`')

    cnx.close()


if __name__ == '__main__':
    # init_settings()
    # transfer_learning(1)
    # load_and_predict('twitter_tl_500.h5')
    # test_batch_predict('twitter_tl_500.h5')

    # load_and_predict_img('twitter_tl_500.h5', pd.read_csv('result_img.csv')['2'])
    # show_result([46, 53, 73, 76, 81, 85, 66, 83, 4, 58, 67, 54, 75, 42, 78, 39])
    # show_result([35, 34, 7, 23, 24, 33, 1, 49, 18, 10, 9, 3, 11, 5, 32, 25])
    # predict_img('twitter_tl_500.h5', 'Test.jpeg')
    # show_cnn_structure()

    # 从数据库中获取图像的路径并且用已经训练好的模型进行情绪分析

    # predict_imgsent_from_db()

    # 从数据库中提取情绪分析的结果
    # show_imgsent_from_db()

    # 计算
    # 不在数据库中计算,在外部计算,方便修改函数
    # 原型成熟了可以用函数计算
    # get_financial_data()
    # merge_article_img('MjM5NDEzMTAwMA==', [date(2021, 6, 1), date(2022, 6, 1)])
    cal_other_columns()
    # map_articles_tradedate('MjM5NDEzMTAwMA==', [date(2021, 6, 1), date(2022, 6, 1)])
    # gzhs_list = select_from_gzhs()
    # for biz in gzhs_list:
    #     print('正在分析:', biz[0])
    #     merge_article_img(biz[0], [date(2021, 6, 1), date(2022, 6, 1)])
