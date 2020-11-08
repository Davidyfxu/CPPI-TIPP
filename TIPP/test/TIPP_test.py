# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:48:45 2019

@author: xyf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) /
                  np.maximum.accumulate(return_list))  # 记录结束位置的下标（最小值的下标）
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置(最大值的下标)
    return ((return_list[j] - return_list[i]) / (return_list[j]))


def calc_rate(day, rate, rate_type):
    if rate_type == 0:
        return (1 + rate*day)
    elif rate_type == 1:
        return np.exp(rate*day)


name = 'zhongzheng500.xlsx'
raw_data = pd.read_excel(name)[['日期', 'rate']]
raw_data.rename(columns={'日期': 'day', 'rate': 'random_ret'}, inplace=True)

raw_data['day'] = pd.to_datetime(raw_data['day'], format='%Y%m%d')
raw_data.set_index('day', inplace=True)
year = raw_data.resample('y')
year.sum()  # 做一次无意义的运算  year才可以用来循环
data = list()
for i, j in year:
    data.append(j)

Return = list()
for i in range(len(data)):
    rate_type = 0  # 0为单利，1为复利
    test_num = 1  # 测试频次

    # 定义市场参数
    trading_year = 1
    trading_day_per_year = data[i].shape[0]
    #days_per_calendar_year = 365
    rf = 0.04  # 无风险利率
    rf_daily = rf / trading_day_per_year
    trading_day_sim = trading_year * trading_day_per_year

    # 定义TIPP参数
    is_take_profit = 0
    init_nav = 1e4  # 初始本金
    adj_period = 5  # 调整周期
    guarantee_rate = 0.8  # 保本比例
    risk_multipler = 2  # 风险乘数
    risk_trading_fee_rate = 6 / 1000  # 风险资产交易手续费率
    gaurant_adj_thresh = 0.03  # 一旦收益超过上一个保本率的 1+百分之几
    gaurant_inc = 0.02  # 就进一步提高保本率
    take_profit_thresh = 0.12
    gaurant_inc_counter = 0

    risk_asset = np.zeros(trading_day_sim)  # 风险资产
    rf_asset = np.zeros(trading_day_sim)  # 无风险资产
    min_pv_asset = np.zeros(trading_day_sim)  # 价值底线
    nav = np.zeros(trading_day_sim)  # 总资产
    nav[1] = init_nav

    # TIPP策略
    # 第1天
    min_pv_asset[1] = guarantee_rate * init_nav / \
        calc_rate(trading_day_sim, rf_daily, rate_type)  # 第1天的价值底线
    risk_asset[1] = max(0, risk_multipler * (nav[1] -
                                             min_pv_asset[1]))   # 风险资产 w/o fee
    rf_asset[1] = (nav[1] - risk_asset[1])  # 无风险资产
    risk_asset[1] = risk_asset[1] * (1 - risk_trading_fee_rate)  # 扣去手续费
    # 第2天到最后1天
    for t in range(2, trading_day_sim):
        # 未止盈
        if is_take_profit == 0:
            # 检查是否已经可以止盈
            fv = nav[t - 1] * calc_rate(trading_day_sim - t, rf_daily, rate_type) - \
                risk_trading_fee_rate * risk_asset[t-1]  # 去除所有的手续费后，看看是否满足止盈条件
            if fv/init_nav - 1 > take_profit_thresh:  # 止盈
                risk_asset[t] = 0
                rf_asset[t] = rf_asset[t-1] * calc_rate(1, rf_daily, rate_type) + (
                    1 - risk_trading_fee_rate) * risk_asset[t - 1]
                is_take_profit = 1
                nav[t] = rf_asset[t]
            else:  # 没有止盈
                # 如果已实现收益，提高保本额度
                if nav[t - 1] / init_nav > guarantee_rate * (gaurant_adj_thresh + 1):
                    guarantee_rate = guarantee_rate + gaurant_inc
                    gaurant_inc_counter = gaurant_inc_counter + 1

                min_pv_asset[t] = guarantee_rate * init_nav / \
                    calc_rate(trading_day_sim - t + 1,
                              rf_daily, rate_type)  # 价值底线
                risk_asset[t] = (1 + data[i].iloc[t-1]) * risk_asset[t-1]
                #risk_asset[t] = (1 + random_ret[0][t-1] ) * risk_asset[t -1]
                rf_asset[t] = calc_rate(
                    1, rf_daily, rate_type) * rf_asset[t - 1]
                nav[t] = risk_asset[t] + rf_asset[t]

                # 定期调整
                if (t - 1) % adj_period == 0:
                    risk_asset_b4_adj = risk_asset[t]
                    risk_asset[t] = max(
                        0, risk_multipler * (nav[t] - min_pv_asset[t]))  # 风险资产
                    rf_asset[t] = nav[t] - risk_asset[t]  # 无风险资产
                    trade_value = risk_asset_b4_adj - risk_asset[t]
                    risk_asset[t] = risk_asset[t] - \
                        abs(trade_value) * risk_trading_fee_rate  # 手续费

                # 检查是否被强制平仓
                if risk_asset[t] <= 0:
                    rf_asset[t] = nav[t] - risk_asset[t] * \
                        risk_trading_fee_rate
                    risk_asset[t] = 0
        else:
            # 止盈
            rf_asset[t] = rf_asset[t - 1] * calc_rate(1, rf_daily, rate_type)
            nav[t] = rf_asset[t]
    Return.append(nav[1:] / init_nav)

if rate_type == 0:
    print('单利计算')
elif rate_type == 1:
    print('复利计算')

annual_return = list()
annual_volatility = list()
Sharpe = list()
Maxdrawdown = list()
for k in range(10):  # 10年期间
    df_return = pd.DataFrame(Return[k])
    annual_return.append(Return[k][len(Return[k])-1])  # 记录年末收益
    volatility = (df_return.shift(1) - df_return)/df_return  # shift时间序列往后推一个
    annual_volatility.append(float(volatility.std()*np.sqrt(trading_day_sim)))
    Sharpe.append((annual_return[k]-1)/annual_volatility[k])
    Maxdrawdown.append(float(df_return.apply(MaxDrawdown)))

Results = pd.DataFrame()
Results['annual_return'] = annual_return
Results['annual_volatility'] = annual_volatility
Results['Sharpe'] = Sharpe
Results['Maxdrawdown'] = Maxdrawdown
print(Results)

# 绘每期收益图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title('净值')
plt.xlabel('时间(天)')
plt.ylabel('净值(元)')
plt.plot(range(1, len(nav)), nav[1:])
plt.savefig('中证500_TIPP.png')
