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


def load_reg_data(nickname=None) -> pd.DataFrame:
    if nickname is not None:
        return mysql_dao.select_table('回归分析', ['*'], {'nick_name': '\'{0}\''.format(nickname)})
    else:
        return mysql_dao.select_table('回归分析', ['*'])


def load_data() -> list:
    return [load_reg_data(i) for i in load_all_media()]


# 只滞后一期回归分析
def do_file_regL1():
    str_do = (
        """


//创建时间序列和设定
gen time=_n
tsset time


//生成哑变量
tab(weekday),gen(dweekday)
drop dweekday1

//更改回报的单位
ge return=log_return*100
ge return_sz=log_return_sz*100
ge return_sh=log_return_sh*100
ge return_shfund=log_return_shfund*100

//生成log2
gen return_s=return^2
gen return_s_sz=return_sz^2
gen return_s_sh=return_sh^2
gen return_s_shfund=return_shfund^2

//描述性统计
outreg2 using result/des{0}.doc,replace sum(detail) keep(c_neg_ratio return return_sh return_sz return_shfund ) eqkeep(N sd mean p50 min max ) title(Decriptive statistics)

//执行回归1 CSI
reg return L1.c_neg_ratio L(1/5).return_s L(1/5).return dweekday*,r

//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(CSI300) keep(L(1/5).c_neg_ratio )

//执行回归2 SH
reg return_sh L1.c_neg_ratio L(1/5).return_s_sh L(1/5).return_sh dweekday*,r


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(000001.SH)  keep(L(1/5).c_neg_ratio)

//执行回归3 SZ
reg return_sz L1.c_neg_ratio L(1/5).return_s_sz L(1/5).return_sz dweekday*,r


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(399001.SZ)  keep(L(1/5).c_neg_ratio)

//执行回归4 SHF
reg return_shfund L1.c_neg_ratio L(1/5).return_s_shfund L(1/5).return_shfund dweekday*,r


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(000011.SH)  keep(L(1/5).c_neg_ratio)



        """
    )

    return str_do


# 基本回归分析
def do_file_reg():
    str_do = (
        """

                      
//创建时间序列和设定
gen time=_n
tsset time


//生成哑变量
tab(weekday),gen(dweekday)
drop dweekday1

//更改回报的单位
ge return=log_return*100
ge return_sz=log_return_sz*100
ge return_sh=log_return_sh*100
ge return_shfund=log_return_shfund*100

//生成log2
gen return_s=return^2
gen return_s_sz=return_sz^2
gen return_s_sh=return_sh^2
gen return_s_shfund=return_shfund^2

//描述性统计
outreg2 using result/des{0}.doc,replace sum(detail) keep(c_neg_ratio return return_sh return_sz return_shfund ) eqkeep(N sd mean p50 min max ) title(Decriptive statistics)

//执行回归1 CSI
reg return L(1/5).c_neg_ratio L(1/5).return_s L(1/5).return dweekday*,r

//假设检验
test L1.c_neg_ratio+L2.c_neg_ratio+L3.c_neg_ratio+L4.c_neg_ratio+L5.c_neg_ratio = 0


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(CSI300) keep(L(1/5).c_neg_ratio )

//执行回归2 SH
reg return_sh L(1/5).c_neg_ratio L(1/5).return_s_sh L(1/5).return_sh dweekday*,r

//假设检验
test L1.c_neg_ratio+L2.c_neg_ratio+L3.c_neg_ratio+L4.c_neg_ratio=0


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(000001.SH)  keep(L(1/5).c_neg_ratio)

//执行回归3 SZ
reg return_sz L(1/5).c_neg_ratio L(1/5).return_s_sz L(1/5).return_sz dweekday*,r

//假设检验
test L1.c_neg_ratio+L2.c_neg_ratio+L3.c_neg_ratio+L4.c_neg_ratio=0


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(399001.SZ)  keep(L(1/5).c_neg_ratio)

//执行回归4 SHF
reg return_shfund L(1/5).c_neg_ratio L(1/5).return_s_shfund L(1/5).return_shfund dweekday*,r

//假设检验
test L1.c_neg_ratio+L2.c_neg_ratio+L3.c_neg_ratio+L4.c_neg_ratio=0


//保存结果
outreg2 using result/reg.doc,append tstat bdec(3) tdec(2) ctitle(000011.SH)  keep(L(1/5).c_neg_ratio)




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
outreg2 using result/reg_var.doc,append tstat bdec(3) tdec(2) ctitle(y)

        """
    )

    return str_do


# 批量处理
def start_run_allgzh():
    df_lists = load_data()
    sfi_s = SFI()
    biz_name = load_all_media()
    for i, df in enumerate(df_lists):
        sfi_s.set_data_df(df)
        sfi_s.run(do_file_reg().format(biz_name[i]))
        sfi_s.run(do_file_var())

    print([[i + 1, j] for i, j in enumerate(load_all_media())])


# 只处理中国证券报
def start_run_csp():
    df = load_reg_data('中国证券报')
    sfi_s = SFI()
    sfi_s.set_data_df(df)
    sfi_s.run(do_file_reg().format('中国证券报'))


if __name__ == "__main__":
    import shutil
    import os

    PATH = '/Users/mac/PycharmProjects/Google-V3/stata/result'
    try:
        shutil.rmtree(PATH)
        os.mkdir(PATH)
    except Exception as e:
        print(e)

    start_run_csp()
    # start_run_allgzh()
