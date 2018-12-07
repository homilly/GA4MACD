import random
import math
import talib
import matplotlib.pyplot as plt

pop_size = 10  # 种群数量
max_value = 10  # 基因中允许出现的最大值
chrom_length = 4  # 染色体长度
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率
results = [[]]  # 存储每一代的最优解，N个二元组
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度


def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)

    return pop[1:]


def b2d(b, max_value, chrom_length):
    t = 0
    for j in range(len(b)):
        t += b[j] * (math.pow(2, j))
    t = t * max_value / (math.pow(2, chrom_length) - 1)
    return t


def decodechrom(pop, chrom_length):
    temp = []
    for i in range(len(pop)):
        t = 0
        for j in range(chrom_length):
            t += pop[i][j] * (math.pow(2, j))
        temp.append(t)
    return temp


def calobjValue(pop, chrom_length, max_value):
    instruments = ['600519.SHA']
    start_date = '2014-01-01'  # 起始时间
    end_date = '2018-01-01'  # 结束时间

    temp1 = []
    obj_value = []
    temp1 = decodechrom(pop, chrom_length)
    for i in range(len(temp1)):
        print(temp1[i])

        def initialize(context):

            context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))  # 设置手续费
            # 使用MACD需要设置长短均线和macd平均线的参数
            context.short = temp1[i]
            context.long = 26
            context.smoothperiod = 9
            context.observation = 50

        def handle_data(context, data):

            if context.trading_day_index < context.observation:
                return

            sid = context.symbol(instruments[0])
            # 读取历史数据
            prices = data.history(sid, 'price', context.observation, '1d')
            # 用Talib计算MACD取值，得到三个时间序列数组，分别为macd, signal 和 hist
            macd, signal, hist = talib.MACD(np.array(prices), context.short, context.long, context.smoothperiod)

            # 计算现在portfolio中股票的仓位
            cur_position = context.portfolio.positions[sid].amount

            # 策略逻辑
            # 卖出逻辑(下穿)
            if macd[-1] - signal[-1] < 0 and macd[-2] - signal[-2] > 0:
                # 进行清仓
                if cur_position > 0 and data.can_trade(sid):
                    context.order_target_value(sid, 0)

            # 买入逻辑(上穿)
            if macd[-1] - signal[-1] > 0 and macd[-2] - signal[-2] < 0:
                if cur_position == 0 and data.can_trade(sid):
                    # 满仓入股
                    context.order_target_percent(sid, 1)

        m = M.trade.v2(
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            initialize=initialize,
            handle_data=handle_data,
            order_price_field_buy='open',
            order_price_field_sell='open',
            capital_base=float("1.0e6"),
            benchmark='000300.INDX',
        )
        obj_value.append(m.raw_perf.read_df().tail(1).algorithm_period_return.sum())
    return obj_value


def calfitValue(obj_value):
    print(obj_value)
    fit_value = []
    c_min = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + c_min > 0):
            temp = c_min + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


def best(pop, fit_value):
    px = len(pop)
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, px):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


def crossover(pop, pc):
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


def cumsum(fit_value):
    for i in range(len(fit_value) - 2, -1, -1):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value) - 1] = 1


def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # 计算累计概率
    cumsum(newfit_value)
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # 转轮盘选择法
    while newin < pop_len:
        if (ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


pop = geneEncoding(pop_size, chrom_length)

print(pop)
temp1 = decodechrom(pop, chrom_length)
print(temp1)

for i in range(pop_size):
    obj_value = calobjValue(pop, chrom_length, max_value)  # 个体评价
    fit_value = calfitValue(obj_value)  # 淘汰
    best_individual, best_fit = best(pop, fit_value)  # 第一个存储最优的解, 第二个存储最优基因
    results.append([best_fit, b2d(best_individual, max_value, chrom_length)])
    selection(pop, fit_value)  # 新种群复制
    crossover(pop, pc)  # 交配
    mutation(pop, pm)  # 变异

results = results[1:]
results.sort()
print(results[-1])
print(best_individual)
print(best_fit)
print(obj_value[1])

print (results)
print ("y = %f, x = %f" % (results[-1][0], results[-1][1]))

X = []
Y = []
for i in range(10):
    X.append(i)
    t = results[i][0]
    Y.append(t)

plt.plot(X, Y)
plt.show()