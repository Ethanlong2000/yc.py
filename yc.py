import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.interpolate import interp1d  # 进行插值画图

plt.style.use('ggplot')
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

x2 = [0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
x3 = [1, 2, 3, 4, 5, 6, 7, 8]

# Average number of students in higher education per 100,000 population
y1 = [931, 1146, 1298, 1420, 1613, 1816, 1924, 2042, 2128, 2189, 2253, 2335, 2418, 2488, 2524, 2530, 2576, 2658, 2857]
# Number of schools on the QS University Rankings liang
y2 = [23, 25, 27, 27, 33, 39, 40, 40]
# Average education expenditure
y3 = [926.1945290, 871.7649663, 790.9739311, 757.2847394, 722.5957561, 806.0438000, 874.3799671, 1020.5126175,
      1085.1793724, 1328.6736655, 1774.4465911,
      2685.0260000, 2532.9807000, 2679.0620000, 2947.0650000, 3750.2782105, 4034.3550261, 4243.3768986, 4441.3656606]
# Ratio of higher education schools per 10,000 people
y4 = [95.98282, 108.67788, 120.09874, 133.16614, 137.04916, 142.03335, 144.40433, 170.40406, 172.72387, 175.85073,
      178.79541, 180.34918, 183.06485, 184.89275, 186.23329, 187.74725, 189.26968, 190.84407, 191.99314]
# Teacher-student ratio (number of teachers = 1)
y5 = [168.5, 172.8, 172.3, 172.7, 173.3, 174.2, 175.2, 175.3, 176.8, 177.3, 175.2, 175.6, 179.5]

# y4=[0.009598282,0.010867788,0.012009874,0.013316614,0.013704916,0.014203335,0.014440433,0.017040406
#     ,0.017272387,0.017585073,0.017879541,0.018034918,0.018306485,0.018489275,0.018623329,0.018774725,
#     0.018926968,0.019084407,0.019199314]
# y5=[18.22,19,17,16.22,16.85,17.93,17.28,17.23,17.27,17.33,17.42,17.52,17.53,17.68,17.73,17.07,17.52,17.56,17.95]
xlable = "TIME"
ylable1 = "Average number of students in higher education per 100,000 population"
ylable2 = "Number of schools on the QS University Rankings liang"
ylable3 = "Average education expenditure"
ylable4 = "Ratio of higher education schools per 10,000 people"
ylable5 = "Teacher-student ratio (number of teachers = 1)"


# 作出数据的散点图
def Data_plot(x, y, xlable, ylable):
    plt.scatter(x, y)
    plt.xlabel(xlable, fontproperties=font)
    plt.ylabel(ylable, fontproperties=font)
    plt.show()
# plt.title(title, fontproperties=font)
#回归分析
def Get_lr(x, y, model):
    message0 = f'The one-dimensional linear regression equation is: \ty={model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f}*x'
    n = len(x)
    y_prd = model.predict(x)
    Regression = sum((y_prd - np.mean(y)) ** 2)  # 回归
    Residual = sum((y - y_prd) ** 2)  # 残差
    R_square = Regression / (Regression + Residual)  # 相关性系数R^2
    F = (Regression / 1) / (Residual / (n - 2))  # F 分布
    pf = stats.f.sf(F, 1, n - 2)
    message1 = (f'Correlation coefficient(R^2)： {R_square[0]:.3f}；\n')
    return print(message0 + '\n' + message1 )

    #
    # message1 = (f'Correlation coefficient(R^2)： {R_square[0]:.3f}；\n' +
    #             f'Regression analysis(SSR)： {Regression[0]:.3f}；\t'
    #             f'Residuals(SSE)：{Residual[0]:.3f}；\n' +
    #             f'           F ： {F[0]:.3f}；\tpf ： {pf[0]}')
    # L_xx = n * np.var(x)
    # sigma = np.sqrt(Residual / (n - 2))
    # t = model.coef_ * np.sqrt(L_xx) / sigma
    # pt = stats.t.sf(t, n - 2)
    # message2 = f'           t ： {t[0][0]:.3f}；\tpt ： {pt[0][0]}'
    # return print(message0 + '\n' + message1 + '\n' + message2)

#残差图
def Residual_plot(x, y, model, font):
    ycout = 'No outliers'
    n = len(x)
    Y_predict = model.predict(x)
    e = y - Y_predict
    sigama = np.std(e)
    L_xx = n * np.var(x)
    H_ii = 1 / n + (x - np.mean(x)) / L_xx
    SRE = e / (sigama * np.sqrt(1 - H_ii))
    if sum(SRE > 3)[0]:
        ycout = x[SRE > 3], y[SRE > 3]
        message = 'Outlier： ' + str(ycout)
    else:
        message = 'Outlier：  ' + ycout

    mx = max(x)[0] + 1
    plt.scatter(x, e, c='green', s=6)
    plt.plot([0, mx], [2 * sigama, 2 * sigama], 'k--', c='orange')
    plt.plot([0, mx], [-2 * sigama, -2 * sigama], 'k--', c='orange')
    plt.plot([0, mx], [3 * sigama, 3 * sigama], 'k--', c='red')
    plt.plot([0, mx], [-3 * sigama, -3 * sigama], 'k--', c='red')
    plt.annotate(message, xy=(0, np.ceil(3 * sigama + 1)), xycoords='data', fontproperties=font)
    plt.xlim(0, mx)
    plt.ylim(-np.ceil(3 * sigama + 2), np.ceil(3 * sigama + 2))
    plt.show()
    return print(message)


# pic(x3,y2,xlable,ylable2)
# pic(x1,y4,xlable,ylable4)
# pic(x1,y3,xlable,ylable3)

X = np.array(x1).reshape(-1, 1)  # xiugai
Y = np.array(y5).reshape(-1, 1)  # xiugai

lr = LinearRegression()
lr.fit(X, Y)
Y_predict = lr.predict(X)
Get_lr(X, Y, lr)

# 画对比图

linear_interp = interp1d(x1, Y_predict.transpose()[0], kind='linear')  # xiugai
computed = np.linspace(min(x1), max(x1), 50)  # xiugai
linear_results = linear_interp(computed)

plt.scatter(x1, y1, s=8, c='green', label='orignal ')  # xiugai
plt.scatter(X, Y_predict, c='orange', s=9, label='predict ')
plt.plot(computed, linear_results, label='linear_interp', alpha=0.7, c='yellow')
plt.xlabel(xlable, fontproperties=font)
plt.ylabel(ylable1, fontproperties=font)  # xiugai
# plt.title('火灾损失',fontproperties = font)
plt.ylim(900, 2900)  # xiugai
plt.legend(loc='upper left')
plt.show()

# 画残差图
Residual_plot(X, Y, lr,font)