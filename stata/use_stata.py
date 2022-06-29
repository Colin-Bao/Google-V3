#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :use_stata.py
# @Time      :2022/6/29 21:26
# @Author    :Colin
# @Note      :None


import pandas as pd
from Stata_SFI import SFI
from tools import mysql_dao


# 获得公众号列表
def load_all_media() -> list:
    return mysql_dao.select_table('gzhs', ['nickname'])['nickname'].tolist()


def load_reg_data(nickname) -> pd.DataFrame:
    return mysql_dao.select_table('回归分析', ['*'], {'nick_name': '\'{0}\''.format(nickname)})


def load_data() -> list:
    return [load_reg_data(i) for i in load_all_media()]


# 基本回归分析
def do_file_reg():
    str_do = (
        """
//创建时间序列和设定
gen time=_n
tsset time

//执行回归
reg log_return c_neg_ratio_L1,r

//保存结果
outreg2 using reg.doc,append tstat bdec(3) tdec(2) ctitle(y)

        """
    )

    return str_do


# 向量自回归分析
def do_file_var():
    str_do = (
        """
//生成哑变量
tab(weekday),gen(dweekday)
drop dweekday1

//生成log2
gen log_return_s=log_return^2

// VAR
var log_return, lags(1/5) exog(L(1/5).c_neg_ratio L(1/5).log_return_s dweekday*)


//保存结果
outreg2 using reg_var.doc,append tstat bdec(3) tdec(2) ctitle(y)

        """
    )

    return str_do


# 批量处理
def start_run():
    df_lists = load_data()
    sfi_s = SFI()
    for i in df_lists:
        sfi_s.set_data_df(i)
        sfi_s.run(do_file_reg())
        sfi_s.run(do_file_var())

    print([[i + 1, j] for i, j in enumerate(load_all_media())])


def test():
    df_list = load_data()
    sfi = SFI()
    sfi.set_data_df(df_list[0])
    sfi.run(do_file_reg())


if __name__ == "__main__":
    start_run()
