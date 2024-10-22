import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si

class BouchardSornettOptionPricing:
    def __init__(self, df: pd.DataFrame):
        """ 初始化 Bouchard-Sornett 类，传入包含价格的 DataFrame """
        self.prices = df['close'].astype(float)  # 确保价格序列是浮点数类型
        self.n = len(self.prices)
        self.epsilon = 1e-10  # 用于防止除零的小常数
        self.returnrate = None  # 收益率序列
        self.detrended = None  # 去趋势收益率序列
        self.pdf_series = None  # 概率密度函数序列

    def calculate_returnrate(self, interval: int = 1) -> pd.Series:
        """ 根据给定的间隔计算收益率 """
        self.returnrate = (self.prices.shift(-interval) - self.prices) / (self.prices + self.epsilon)
        return self.returnrate

    def detrended_returnrate(self) -> pd.Series:
        """ 计算去趋势的收益率序列 """
        returnrate = self.calculate_returnrate()
        self.detrended = returnrate - returnrate.mean()
        return self.detrended

    def calculate_pdf(self, ifplot: bool = False, nbin: int = None) -> pd.Series:
        """ 计算基于去趋势收益率的概率密度函数 (PDF) """
        self.detrended_returnrate()  # 确保已经计算了去趋势收益率
        Rmax = self.detrended.max()
        Rmin = self.detrended.min()
        if nbin is None:
            nbin = int(np.sqrt(self.n))  # 默认的 bin 数量
        step = (Rmax - Rmin) / nbin
        counts = []
        for i in range(nbin):
            R_i = Rmin + i * step
            count = ((self.detrended >= R_i) & (self.detrended < R_i + step)).sum()
            counts.append(count)
        count_series = pd.Series(counts, index=[Rmin + i * step for i in range(nbin)])
        self.pdf_series = count_series / self.n

        if ifplot:
            plt.figure(figsize=(10, 6))
            plt.bar(count_series.index, count_series.values / self.n, width=step, align='edge', alpha=0.6, color='b')
            plt.title('概率密度函数 (PDF)')
            plt.xlabel('去趋势收益率')
            plt.ylabel('概率')
            plt.grid(True)
            plt.show()

        return self.pdf_series

    def price_option(self, S0, X_range, T=21, r=0.00):
        """ 使用 Bouchard-Sornett 方法计算期权价格 """
        # 如果还未计算，先计算去趋势收益率和 PDF
        if self.pdf_series is None:
            self.calculate_pdf()

        # 基于公式 (6.20) 计算期权价格
        option_prices = []
        for X in X_range:
            payoff_sum = 0
            for k, prob in self.pdf_series.items():
                payoff = max(S0 * (1 + k) - X, 0)
                payoff_sum += payoff * prob
            V_0 = np.exp(-r * T / 252) * payoff_sum  # 折现后的预期支付
            option_prices.append(V_0)

        return option_prices

# Black-Scholes 公式实现
def black_scholes(S, X, T, r, sigma):
    """ 使用 Black-Scholes 公式计算期权价格 """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - X * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price

# 读取并清理数据
df_cleaned = pd.read_csv('data.csv')  # 请确保文件路径正确，data.csv 为包含日期和收盘价的文件
df_cleaned = df_cleaned[['date', 'close']].dropna()  # 保留所需的列并移除空值
df_cleaned['close'] = df_cleaned['close'].astype(float)  # 确保价格数据为浮点数
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])  # 将日期列转换为日期类型
df_cleaned = df_cleaned.sort_values('date')  # 按日期排序

# 使用清理后的数据进行初始化
SZ_option_pricing = BouchardSornettOptionPricing(df_cleaned)

# 定义初始股价 (S0) 和行权价格范围 (X_range)
S0 = df_cleaned['close'].iloc[-1]  # 数据集中的最后一个收盘价作为 S0
X_range = np.linspace(S0 * 0.8, S0 * 1.2, 50)  # 行权价格范围为 S0 的 80% 到 120%

# 使用 Bouchard-Sornett 方法计算期权价格
option_prices_bs = SZ_option_pricing.price_option(S0, X_range)

# Black-Scholes 参数
T = 21 / 252  # 21天到期时间，转换为年
r = 0.00  # 无风险利率
sigma = 0.2  # 假设年化波动率

# 使用 Black-Scholes 方法计算期权价格
bs_option_prices = [black_scholes(S0, X, T, r, sigma) for X in X_range]

# 绘制两种方法的期权价格对比图，并添加参数标注
plt.figure(figsize=(10, 6))
plt.plot(X_range, option_prices_bs, label='Option Price (Bouchard-Sornett)', color='blue')
plt.plot(X_range, bs_option_prices, label='Option Price (Black-Scholes)', color='red', linestyle='--')

# 图表标题和标签
plt.title("Option Price (V_0) vs Strike Price (X)\n" +
          f"T={T*252:.0f} days, r={r:.2f}, sigma={sigma:.2f}")
plt.xlabel("Strike Price (X)")
plt.ylabel("Option Price (V_0)")
plt.grid(True)
plt.legend()

# 标注参数值
plt.text(X_range[40], max(option_prices_bs) * 0.75,
         f"Parameters:\nT = {T*252:.0f} days\nr = {r:.2f}\nsigma = {sigma:.2f}",
         fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.show()
