import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def getData(filename):
    """
    处理数据，获得各年的日涨跌幅

    : param filename: 输入文件，需要包含两列，日期和日涨跌
    : return: 输出dateframe格式数据，为按年分的日期和日涨跌
    """
    raw_data = pd.read_excel(filename)[['日期', 'rate']]
    raw_data.rename(columns={'日期': 'day', 'rate': 'random_ret'}, inplace=True)
    raw_data['day'] = pd.to_datetime(raw_data['day'], format='%Y%m%d')
    raw_data.set_index('day', inplace=True)

    year = raw_data.resample('y')
    year.sum()  # 做一次无意义的运算  year才可以用来循环
    data = [j for i, j in year]
    return data


def outputQuantResult(Return, nav, trading_day_sum):
    """
    输入经过该策略后的时间序列结果, 绘制收益图像

    : param Return: 收益结果数据
    : param nav: 总资产
    : param trading_day_sum: 交易日总数
    : return Results: Dataframe格式的年收益，年波动性，夏普比率，最大回撤
    """
    annual_return, annual_volatility, Sharpe, Maxdrawdown = [], [], [], []
    for i in range(len(Return)):
        df_return = pd.DataFrame(Return[i])
        annual_return.append(Return[i][len(Return[i])-1])
        volatility = (df_return.shift(1) - df_return)/df_return
        annual_volatility.append(
            float(volatility.std()*np.sqrt(trading_day_sum)))
        Sharpe.append((annual_return[i]-1)/annual_volatility[i])
        Maxdrawdown.append(float(df_return.apply(MaxDrawdown, axis=0)))

    Results = pd.DataFrame()
    Results['annual_return'] = annual_return
    Results['annual_volatility'] = annual_volatility
    Results['Sharpe'] = Sharpe
    Results['Maxdrawdown'] = Maxdrawdown

    # 绘每期收益图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('时间(天)')
    plt.ylabel('净值(元)')
    plt.plot(range(1, len(nav)), nav[1:])

    return Results


def getParameters(test_num=1, rate_type=0, trading_year=1, trading_day_per_year=255, rf=0.04, init_nav=1e4, adj_period=5, guarantee_rate=0.8, risk_multipler=2, risk_trading_fee_rate=0.006):
    """
    存储所需的固定金融参数初始化的数值

    : param params: 与量化策略所需的参数有关
    : return: 用字典存储各个金融参数
    """

    parameters = {}

    parameters["test_num"] = test_num
    parameters["rate_type"] = rate_type  # 0: simple型  1:compound型

    # 定义市场参数
    parameters["trading_year"] = trading_year
    parameters["rf"] = rf  # 无风险利率
    parameters["trading_day_per_year"] = trading_day_per_year
    parameters["rf_daily"] = rf / trading_day_per_year
    parameters["trading_day_sum"] = trading_year * trading_day_per_year
    parameters["init_nav"] = init_nav  # 初始本金
    parameters["adj_period"] = adj_period  # 调整周期
    parameters["guarantee_rate"] = guarantee_rate  # 保本比例
    parameters["risk_multipler"] = risk_multipler  # 风险乘数
    parameters["risk_trading_fee_rate"] = risk_trading_fee_rate  # 风险资产交易费率
    return parameters


def main():
    data = getData(filename='zhongzheng500.xlsx')
    Return = []
    for i in range(len(data)):

        p = getParameters(trading_day_per_year=data[i].shape[0])

        datashape = [p["trading_day_sum"]+1, p["test_num"]]
        risk_asset = np.zeros(datashape)  # 风险资产
        rf_asset = np.zeros(datashape)  # 无风险资产
        min_pv_asset = np.zeros(datashape)  # 价值底线
        nav = np.zeros(datashape)  # 总资产
        nav[1, :] = p["init_nav"]

        # CPPI策略
        # 第1天
        min_pv_asset[1, :] = p["guarantee_rate"] * p["init_nav"] / \
            calc_rate(p["trading_day_sum"], p["rf_daily"],
                      p["rate_type"])  # 第1天的价值底线
        risk_asset[1, :] = np.maximum(np.zeros(
            p["test_num"]), p["risk_multipler"] * (nav[1, :] - min_pv_asset[1, :]))  # 风险资产 w/o fee
        rf_asset[1, :] = (nav[1, :] - risk_asset[1, :])  # 无风险资产
        risk_asset[1, :] = risk_asset[1, :] * \
            (1 - p["risk_trading_fee_rate"])  # 扣去手续费
        # 第二天到最后一天
        for t in range(2, p["trading_day_sum"]+1):
            min_pv_asset[t, :] = p["guarantee_rate"] * p["init_nav"] / \
                calc_rate(p["trading_day_sum"]-t+1,
                          p["rf_daily"], p["rate_type"])  # 价值底线

            risk_asset[t, :] = (1 + data[i].iloc[t-1]) * risk_asset[t-1, :]
            rf_asset[t, :] = calc_rate(
                1, p["rf_daily"], p["rate_type"]) * rf_asset[t-1, :]
            nav[t, :] = risk_asset[t, :] + rf_asset[t, :]

            # 定期调整
            if np.mod(t-1, p["adj_period"]) == 0:
                risk_asset_b4_adj = risk_asset[t, :]
                risk_asset[t, :] = np.maximum(
                    np.zeros(p["test_num"]), p["risk_multipler"] * (nav[t, :] - min_pv_asset[t, :]))  # 风险资产
                rf_asset[t, :] = nav[t, :] - risk_asset[t, :]  # 无风险资产
                risk_asset[t, :] = risk_asset[t, :] - \
                    abs(risk_asset_b4_adj -
                        risk_asset[t, :]) * p["risk_trading_fee_rate"]  # 手续费

            # 检查是否被强制平仓
            rf_asset[t, risk_asset[t, :] <= 0] = nav[t, risk_asset[t, :] <=
                                                     0] - risk_asset[t, risk_asset[t, :] <= 0] * p["risk_trading_fee_rate"]
            risk_asset[t, risk_asset[t, :] <= 0] = 0
        Return.append(nav[1:, 0]/p["init_nav"])

    Results = outputQuantResult(Return, nav, p["trading_day_sum"])
    Results.to_excel("temp.xlsx")


if __name__ == "__main__":
    main()
