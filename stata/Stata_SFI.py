import pandas as pd

from pystata import config


class SFI:
    # 初始化
    def __init__(self, ):
        # 初始化 SFI
        config.init('mp')
        if config.is_stata_initialized():
            # 传入数据
            from pystata import stata
            self.stata = stata
            self.df_SFI = pd.DataFrame()

    # 更新传入的数据
    def set_data_df(self, df):
        self.df_SFI = df
        self.stata.pdataframe_to_data(self.df_SFI, force=True)

    # 指定变量标签
    def set_labels(self, variable, label):
        from sfi import ValueLabel, Data
        Data.setVarLabel(variable, label)

    # 数据类型展示
    def describe(self):
        self.stata.run('describe')

    def regress(self, variable):
        self.stata.run('regress ' + variable)
        return self.stata.get_return(), self.stata.get_ereturn()

    def predict(self, variable_name):
        self.stata.run('predict ' + variable_name)
        return self.stata.pdataframe_from_data('midval')

    def run(self, stata_str):
        self.stata.run(stata_str)
