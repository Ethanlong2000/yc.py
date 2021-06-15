import matplotlib.pyplot as plt
plt.style.use('ggplot')
## 解决中文字符显示不全
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
import numpy as np
from sklearn.linear_model import LinearRegression

xlable = "TIME"
ylable1 = "Average number of students in higher education per 100,000 population"
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
y = [931, 1146, 1298, 1420, 1613, 1816, 1924, 2042, 2128, 2189, 2253, 2335, 2418, 2488, 2524, 2530, 2576, 2658, 2857]

def get_lr_stats(x, y, model):

    message0 = f'一元线性回归方程为: \ty={model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f}*x'
    from scipy import stats
    n     = len(x)
    y_prd = model.predict(x)
    Regression = sum((y_prd - np.mean(y))**2) # 回归
    Residual   = sum((y - y_prd)**2)          # 残差
    R_square   = Regression / (Regression + Residual) # 相关性系数R^2
    F          = (Regression / 1) / (Residual / ( n - 2 ))  # F 分布
    pf         = stats.f.sf(F, 1, n-2)
    message1 = (f'相关系数(R^2)： {R_square[0]:.3f}；\n' +
                f'回归分析(SSR)： {Regression[0]:.3f}；\t残差(SSE)：{Residual[0]:.3f}；\n' +
                f'           F ： {F[0]:.3f}；\tpf ： {pf[0]}')
    ## T
    L_xx  =  n * np.var(x)
    sigma =  np.sqrt(Residual / (n-2))
    t     =  model.coef_ * np.sqrt(L_xx) / sigma
    pt    =  stats.t.sf(t, n-2)
    message2 = f'           t ： {t[0][0]:.3f}；\tpt ： {pt[0][0]}'
    return print(message0 +'\n' +message1 + '\n'+message2)


x_in = np.array(x).reshape(-1,1)
y_in = np.array(y).reshape(-1,1)
lreg = LinearRegression()
lreg.fit(x_in, y_in)
y_prd = lreg.predict(x_in)
get_lr_stats(x_in, y_in, lreg)


from scipy.interpolate import interp1d # 进行插值画图
linear_interp = interp1d(x, y_prd.transpose()[0], kind='linear')
computed = np.linspace(min(x),max(x) , 50)
linear_results = linear_interp(computed)

plt.scatter(x, y,s = 8,c='green', label = 'orignal ')
plt.scatter(x_in, y_prd,c = 'orange', s = 9 ,  label = 'predict ')
plt.plot(computed, linear_results , label = 'linear_interp', alpha = 0.7, c = 'yellow')
plt.xlabel(xlable,fontproperties = font)
plt.ylabel(ylable1,fontproperties = font)
# plt.title('火灾损失',fontproperties = font)
plt.ylim(900,2900)
plt.legend(loc = 'upper left')
plt.show()