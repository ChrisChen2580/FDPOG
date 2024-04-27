import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def f(x, r):
    return np.abs(x / (x + r) - x)

x = np.linspace(0, 4, 1000) # 生成从0到10之间的1000个等间距的x值
r_list = [0.2, 0.5, 1]      # 不同的r值列表

for r in r_list:
    y = f(x, r)
    plt.plot(x, y, label=r'$\gamma$={}'.format(r)) # 绘制函数曲线

plt.xlim(0, 4) # 设置x轴范围从0到10
plt.ylim(0, 4)  # 设置y轴范围从0到1
plt.xlabel('a')
plt.ylabel('F(a)')
# plt.xticks(fontsize=20) # 设置x轴刻度字号为20
# plt.yticks(fontsize=20) # 设置y轴刻度字号为20
plt.legend()
# plt.show() # 显示图像

# 设置保存的PDF文件的字体和其他参数
rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tight_layout()
plt.savefig('plot.pdf')