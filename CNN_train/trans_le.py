import time
from datetime import datetime, date
from PIL import Image
from data_down import tushare_api
import os
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import callbacks
import matplotlib.pyplot as plt

import PIL
from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras_visualizer import visualizer
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
