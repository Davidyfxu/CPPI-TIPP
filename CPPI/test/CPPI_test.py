import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def MaxDrawdown(return_list):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) -
                   return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return (return_list[j] - return_list[i]) / (return_list[j])


def calc_rate(day, rate, rate_type):
    if rate_type == 0:
        return (1 + rate*day)
    elif rate_type == 1:
        return np.exp(rate*day)


name = 'zhongzheng500.xlsx'
raw_data = pd.read_excel(name)[['日期', 'rate']]
raw_data.rename(columns={'日期': 'day', 'rate': 'random_ret'}, inplace=True)
#raw_data.drop(columns = '日净值',inplace = True)

raw_data['day'] = pd.to_datetime(raw_data['day'], format='%Y%m%d')
raw_data.set_index('day', inplace=True)
# raw_data.drop(labels='Unnamed:0',axis=0,inplace=True)
year = raw_data.resample('y')
year.sum()  # 做一次无意义的运算  year才可以用来循环
data = list()
for i, j in year:
    # print(str(i.year))
    data.append(j)


Return = list()
for i in range(len(data)):
    # print(data[i].shape[0])
    test_num = 1
    rate_type = 0  # 0: simple  1:compound

    # 定义市场参数
    trading_year = 1
    trading_day_per_year = data[i].shape[0]
    days_per_calendar_year = 365
    rf = 0.04  # 无风险利率
    rf_daily = rf / trading_day_per_year
    trading_day_sim = trading_year * trading_day_per_year
    init_nav = 1e4  # 初始本金
    adj_period = 5  # 调整周期
    guarantee_rate = 0.8  # 保本比例
    risk_multipler = 2  # 风险乘数
    risk_trading_fee_rate = 6 / 1000  # 风险资产交易费率
    datashape = [trading_day_sim+1, test_num]
    risk_asset = np.zeros(datashape)  # 风险资产
    rf_asset = np.zeros(datashape)  # 无风险资产
    min_pv_asset = np.zeros(datashape)  # 价值底线
    nav = np.zeros(datashape)  # 总资产
    nav[1, :] = init_nav

    # CPPI策略
    # 第1天
    min_pv_asset[1, :] = guarantee_rate * init_nav / \
        calc_rate(trading_day_sim, rf_daily, rate_type)  # 第1天的价值底线
    risk_asset[1, :] = np.maximum(np.zeros(
        test_num), risk_multipler * (nav[1, :] - min_pv_asset[1, :]))  # 风险资产 w/o fee
    rf_asset[1, :] = (nav[1, :] - risk_asset[1, :])  # 无风险资产
    risk_asset[1, :] = risk_asset[1, :] * (1 - risk_trading_fee_rate)  # 扣去手续费
    # 第二天到最后一天
    for t in range(2, trading_day_sim+1):
        min_pv_asset[t, :] = guarantee_rate * init_nav / \
            calc_rate(trading_day_sim-t+1, rf_daily, rate_type)  # 价值底线

        risk_asset[t, :] = (1 + data[i].iloc[t-1]) * risk_asset[t-1, :]
        #risk_asset[t,:] = (1+random_ret[t-1,:]) * risk_asset[t-1,:]
        rf_asset[t, :] = calc_rate(1, rf_daily, rate_type) * rf_asset[t-1, :]
        nav[t, :] = risk_asset[t, :] + rf_asset[t, :]

        # 定期调整
        if np.mod(t-1, adj_period) == 0:
            risk_asset_b4_adj = risk_asset[t, :]
            risk_asset[t, :] = np.maximum(
                np.zeros(test_num), risk_multipler * (nav[t, :] - min_pv_asset[t, :]))  # 风险资产
            rf_asset[t, :] = nav[t, :] - risk_asset[t, :]  # 无风险资产
            risk_asset[t, :] = risk_asset[t, :] - \
                abs(risk_asset_b4_adj -
                    risk_asset[t, :]) * risk_trading_fee_rate  # 手续费

        # 检查是否被强制平仓
        rf_asset[t, risk_asset[t, :] <= 0] = nav[t, risk_asset[t, :] <=
                                                 0] - risk_asset[t, risk_asset[t, :] <= 0] * risk_trading_fee_rate
        risk_asset[t, risk_asset[t, :] <= 0] = 0
    Return.append(nav[1:, 0]/init_nav)

annual_return = list()
annual_volatility = list()
Sharpe = list()
Maxdrawdown = list()
for l in range(10):
    df_return = pd.DataFrame(Return[l])
    annual_return.append(Return[l][len(Return[l])-1])
    volatility = (df_return.shift(1) - df_return)/df_return
    annual_volatility.append(float(volatility.std()*np.sqrt(trading_day_sim)))
    Sharpe.append((annual_return[l]-1)/annual_volatility[l])
    Maxdrawdown.append(float(df_return.apply(MaxDrawdown, axis=0)))

Results = pd.DataFrame()
Results['annual_return'] = annual_return
Results['annual_volatility'] = annual_volatility
Results['Sharpe'] = Sharpe
Results['Maxdrawdown'] = Maxdrawdown
print(Results)
# Results.to_excel(name+'results1.xlsx')

# 绘每期收益图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title('净值')
plt.xlabel('时间(天)')
plt.ylabel('净值(元)')
plt.plot(range(1, len(nav)), nav[1:])
