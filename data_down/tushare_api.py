import tushare as ts


# 用于给其他文件调用的类

class TuShareGet:
    __TOKEN = 'e6f824fb07fc769e95d32580a9fc8b0ab24669c24eac798b0e5056dc'
    ts_code = None
    start_date = None
    end_date = None
    data_df = None

    def __init__(self, start_date, end_date):
        self.start_date, self.end_date = start_date, end_date
        ts.set_token(self.__TOKEN)
        self.pro = ts.pro_api()

    def get_shares_list(self):
        return self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

    def get_kline(self, ts_code, flag_return=True):
        self.ts_code = ts_code
        self.data_df = self.pro.daily(ts_code=self.ts_code, start_date=self.start_date, end_date=self.end_date)

        if flag_return:
            pass
            # self.data_df = cal_return(self.data_df)

        return self.data_df

    def get_income(self, ts_code):
        self.data_df = self.pro.income(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)

    def get_gdp(self):
        return self.pro.cn_gdp(start_q=self.start_date, end_q=self.end_date)

    def get_cpi(self):
        return self.pro.cn_cpi(start_m=self.start_date, end_m=self.end_date)

    def get_ppi(self):
        return self.pro.cn_ppi(start_m=self.start_date, end_m=self.end_date)

    def get_currency(self):
        return self.pro.cn_m(start_m=self.start_date, end_m=self.end_date)

    def get_shibor(self):
        return self.pro.shibor(start_date=self.start_date, end_date=self.end_date)

    def get_index(self, code):
        return self.pro.index_daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)

    def get_opt(self, code):
        return self.pro.opt_daily(ts_code=code, start_date=self.start_date, end_date=self.end_date)

    def get_opt_basic(self):
        return self.pro.opt_basic(exchange='DCE', fields='ts_code,name,exercise_type,list_date,delist_date')
