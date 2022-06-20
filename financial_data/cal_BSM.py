from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
# 导入matplotlib的pyplot模块
import matplotlib.pyplot as plt


class BSM:
    # 计算参数的内部类
    class CalParaColumn:

        # 自定义计算函数 sqrt exp log cdf
        class Cal:
            result = None

            def cal_log(self, df_num):
                self.result = df_num.apply(lambda x: log(x) if x is not np.nan else x)
                return self.result

            def cal_exp(self, df_num):
                self.result = df_num.apply(lambda x: exp(x) if x is not np.nan else x)
                return self.result

            def cal_sqrt(self, df_num):
                self.result = df_num.apply(lambda x: sqrt(x) if x is not np.nan else x)
                return self.result

            def cal_cdf(self, df_num):
                self.result = df_num.apply(lambda x: norm.cdf(x) if x is not np.nan else x)
                return self.result

            def cal(self, para):
                function_name = 'cal_' + para
                method = getattr(self, function_name, self.Default)
                return method

            def Default(self, o, o1):
                return self.result, o, o1

        # 参数区
        df_cal = None
        start_date, end_date, expiry_day = None, None, None
        T, S, K, r, sigma, d1, d2, bs_call, bs_put = None, None, None, None, None, None, None, None, None

        def __init__(self, dataframe, date_dict, K):
            self.df_cal = dataframe
            self.start_date, self.end_date, self.expiry_day = date_dict['start_date'], date_dict['end_date'], date_dict[
                'expiry_day']
            self.K_value = K
            self.create_column()

        # 更新计算列
        def update_column(self):
            # 参数区
            self.T = self.df_cal['expiry_t']
            self.S = self.df_cal['pre_close']
            self.K = self.df_cal['K']
            self.r = self.df_cal['r_1y']
            self.sigma = self.df_cal['sigma_1y']

            self.d1 = self.df_cal['d1']
            self.d2 = self.df_cal['d2']

            self.bs_call = self.df_cal['call']
            self.bs_put = self.df_cal['put']

        # 创造用于计算的列
        def create_column(self):
            cal_column = ['K', 'r_1y', 'expiry_t', 'sigma_1y', 'd1', 'd2', 'call', 'put']
            for cal in cal_column:
                if cal not in self.df_cal.columns:
                    self.df_cal[str(cal)] = np.nan

            # 参数区
            self.update_column()

        # 控制外部访问的参数计算方法
        def switch_para(self, para):
            function_name = 'cal_' + para
            method = getattr(self, function_name, self.Default)
            return method

        # 筛选出持有期的所有行执行结果
        def cal_execute(self, para, cal_result):
            # if para not in self.df_cal.columns:
            #     self.df_cal[str(para)] = np.nan

            dt_start, dt_end = datetime.strptime(self.start_date, '%Y%m%d'), datetime.strptime(self.end_date,
                                                                                               '%Y%m%d')
            self.df_cal.loc[(pd.to_datetime(self.df_cal['trade_date'], format='%Y-%m-%d') <= dt_end) & (
                    pd.to_datetime(self.df_cal['trade_date'], format='%Y-%m-%d') >= dt_start), (
                                para,)] = cal_result
            self.update_column()

        # --------------------------------------【参数计算区】----------------------------------------- #

        # 筛选日期并计算K
        def cal_K(self):
            cal_result = self.K_value
            self.cal_execute('K', cal_result)
            return self.df_cal

        # 筛选日期并计算sigma
        def cal_r(self):
            risk_free_year = ASHARE_Select.TuShareGet(self.start_date,
                                                      self.end_date).get_shibor()
            # 用最后一行的1年利率
            risk_free_year_now = risk_free_year.iloc[-1]['1y'] / 100.0
            cal_result = risk_free_year_now
            self.cal_execute('r_1y', cal_result)
            return self.df_cal

        # 筛选日期并计算sigma
        def cal_sigma(self):
            # 筛选出观测期
            def cal_observe_sigma():
                # 先计算回报率
                def cal_return(df):
                    # print(self.start_date)
                    df = df.sort_values(by="trade_date")
                    # assign是在新增计算行的时候同时增加
                    cal_df = df.assign(close_day_before=df.pre_close.shift(1))
                    cal_df['return'] = ((cal_df.pre_close - cal_df.close_day_before) / cal_df.close_day_before)
                    return cal_df

                # 更新原有的df
                self.df_cal = cal_return(self.df_cal)

                # 用提前一年和start_date计算方差
                dt_start = datetime.strptime(self.start_date, '%Y%m%d')
                one_year_ago = dt_start.replace(year=dt_start.year - 1)

                # 筛选日期
                df_observe = self.df_cal[
                    (pd.to_datetime(self.df_cal['trade_date'], format='%Y-%m-%d') <= dt_start) & (
                            pd.to_datetime(self.df_cal['trade_date'], format='%Y-%m-%d') >= one_year_ago)]
                # 使用观测期计算方差
                return np.sqrt(252) * df_observe['return'].std()

            # 计算观测期的sigma
            observe_sigma = cal_observe_sigma()

            self.cal_execute('sigma_1y', observe_sigma)

            return self.df_cal

        # 筛选日期并计算t
        def cal_t(self):
            # ------------------ 0.生成计算公式 ------------------ #
            dt_expiry = datetime.strptime(self.expiry_day, '%Y%m%d')
            cal_result = dt_expiry - pd.to_datetime(self.df_cal['trade_date'], format='%Y-%m-%d')

            # ------------------ 1.执行计算公式 ------------------ #
            self.cal_execute('expiry_t', cal_result)

            # ------------------ 2.数据格式处理 ------------------ #
            def date_clean():
                # 日期格式转换
                self.df_cal['expiry_t'] = pd.to_datetime(self.df_cal['expiry_t'],
                                                         infer_datetime_format=True) - datetime.strptime(
                    '1970-01-01', "%Y-%m-%d")
                # 转换datatime64
                self.df_cal['expiry_t'] = self.df_cal['expiry_t'].astype('timedelta64[D]').astype('float') / 365.0

            date_clean()
            self.update_column()

            return self.df_cal

        # 筛选日期并计算d1
        def cal_d1(self):
            # ------------------ 0.生成计算公式 ------------------ #
            my = self.Cal()
            # self.d1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
            cal_result = ((my.cal('log')(self.S / self.K)) + (self.r + self.sigma ** 2 / 2) * self.T) / (
                    self.sigma * my.cal('sqrt')(self.T))

            # ------------------ 1.执行计算公式 ------------------ #
            self.cal_execute('d1', cal_result)

            return self.df_cal

        # 筛选日期并计算d2
        def cal_d2(self):
            # ------------------ 0.生成计算公式 ------------------ #
            my = self.Cal()
            #  self.d2 = self.cal_d1(S, K, T, r, sigma) - sigma * sqrt(T)
            cal_result = self.d1 - self.sigma * (my.cal('sqrt')(self.T))

            # ------------------ 1.执行计算公式 ------------------ #
            self.cal_execute('d2', cal_result)

            return self.df_cal

        # 筛选日期并计算call
        def cal_c(self):
            # ------------------ 0.生成计算公式 ------------------ #
            my = self.Cal()
            #    bs_call = S * norm.cdf(self.d1) - K * exp(-r * T) * norm.cdf(self.d2)
            cal_result = self.S * (my.cal('cdf')(self.d1)) - self.K * (my.cal('exp')(-self.r * self.T)) * (
                my.cal('cdf')(self.d2))

            # ------------------ 1.执行计算公式 ------------------ #
            self.cal_execute('call', cal_result)

            return self.df_cal

        # 筛选日期并计算put
        def cal_p(self):
            # ------------------ 0.生成计算公式 ------------------ #
            my = self.Cal()
            #     self.bs_put = K * exp(-r * T) - S + self.cal_bs_call(S, K, T, r, sigma)
            cal_result = self.K * (my.cal('exp')(-self.r * self.T)) - self.S + self.bs_call

            # ------------------ 1.执行计算公式 ------------------ #
            self.cal_execute('put', cal_result)

            return self.df_cal

        # 默认方法
        def Default(self):
            return self.df_cal

    # 类成员
    share_name = None
    df_kline = None
    date_dict = {'start_date': '', 'end_date': '', 'expiry_day': ''}

    # 初始化类 输入K, share_name, date_dict初始化，得到所有参数生成df
    def __init__(self, K, share_name, date_dict):
        # ----------- 0.自定义参数输入 ---------#
        self.share_name = share_name
        self.K = K
        self.date_dict = date_dict
        self.df_kline = self.get_kline()  # 往前一年获取

        # ----------- 1.使用内部类来进行参数计算 ---------#
        self.cal_para_column()

        # ----------- 2.使用内部类来进行可视化 ---------#
        # self.vis_data()

    # 加载数据
    def get_kline(self):
        file_name = self.share_name + '.csv'

        # 如果文件存在
        try:
            df_kline = pd.read_csv(file_name)

        except FileNotFoundError as e:
            print(e)
            # 需要提前一年的数据计算方差
            start_date = datetime.strptime(self.date_dict['start_date'], '%Y%m%d')
            one_year_ago = start_date.replace(year=start_date.year - 1).strftime('%Y%m%d')

            df_kline = pd.DataFrame(
                ASHARE_Select.TuShareGet(one_year_ago, self.date_dict['end_date']).get_kline(
                    self.share_name))

            # 处理df-kline
            df_kline['trade_date'] = pd.to_datetime(df_kline['trade_date'], format='%Y%m%d')

            # 储存
            df_kline.to_csv(file_name)

        return df_kline

    # 保存数据
    def save_data(self, df_kline):
        file_name = self.share_name + '.csv'
        # 储存要记得index为False
        df_kline.to_csv(file_name, index=False)

    # 计算数据
    def cal_para_column(self):

        # 用于计算的内部类
        cls_cal = self.CalParaColumn(self.df_kline, self.date_dict, self.K)

        # 参数计算
        self.df_kline = cls_cal.switch_para('K')()
        self.df_kline = cls_cal.switch_para('r')()
        self.df_kline = cls_cal.switch_para('t')()
        self.df_kline = cls_cal.switch_para('sigma')()
        self.df_kline = cls_cal.switch_para('d1')()
        self.df_kline = cls_cal.switch_para('d2')()
        self.df_kline = cls_cal.switch_para('c')()
        self.df_kline = cls_cal.switch_para('p')()

        # 更新数据
        self.save_data(self.df_kline)
