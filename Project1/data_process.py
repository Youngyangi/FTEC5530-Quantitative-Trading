import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hangseng = pd.read_csv('data/HSI.csv')
ha = pd.read_csv('data/HA.csv')
rya = pd.read_csv('data/RYAAY.csv')
sky = pd.read_csv('data/SKYW.csv')


def ploting():
    stock_dict = {
        "HA": ha,
        "RYAAY": rya,
        "SKYW": sky
    }
    fig = plt.figure()
    fig.tight_layout()
    plt.subplot(2, 2, 1)
    plt.plot(hangseng['Date'], hangseng['Close'], label='HangSeng Closed Price')
    plt.xlabel("2019.1.1 - 2019.12.31")
    plt.ylabel("price")
    plt.title('HangSeng Closed Price', fontsize=10, color='blue')

    for i, x in enumerate(stock_dict.keys()):
        plt.subplot(2, 2, i+2)
        plt.plot(stock_dict[x]['Date'], stock_dict[x]['Adj Close'])
        plt.title(f"{x} Adj Close Price", fontsize=10, color='red')
        plt.xlabel("2019.1.1 - 2019.12.31")
        plt.ylabel("price")
    fig.savefig('figure', dpi=500, bbox_inches='tight')
    # plt.show()

# ploting()

# Stocks have more records than the HSI, remove those
label1 = set(hangseng['Date'].tolist())
label2 = set(ha['Date'].tolist())
label = label1 & label2


def remove_extra(data, label):
    id = []
    for i, x in enumerate(data['Date'].tolist()):
        if x not in label:
            id.append(i)
    return data.drop(id, axis=0)


hangseng = remove_extra(hangseng, label)
ha = remove_extra(ha, label)
rya = remove_extra(rya, label)
sky = remove_extra(sky, label)

hangseng_price = np.array(hangseng['Close'].tolist())[:, np.newaxis]
ha_price = np.array(ha['Adj Close'].tolist())[:, np.newaxis]
rya_price = np.array(rya['Adj Close'].tolist())[:, np.newaxis]
sky_price = np.array(sky['Adj Close'].tolist())[:, np.newaxis]

matrix = np.concatenate([hangseng_price, ha_price, rya_price, sky_price], axis=1)
# print(matrix)

print(np.cov(matrix.T))


def get_return_array(data):
    array = []
    for i in range(len(data)-1):
        score = (data[i+1] - data[i]) / data[i]
        array.append(score)
    return np.array(array)[:, np.newaxis]


def get_logreturn_array(data):
    array = []
    for i in range(len(data) - 1):
        score = np.log(data[i+1]/data[i])
        array.append(score)
    return np.array(array)[:, np.newaxis]


def get_return_cov(data, log=False):
    arrays = []
    if log:
        for x in data:
            arrays.append(get_logreturn_array(x))
        matrix = np.concatenate(arrays, axis=1)
    else:
        for x in data:
            arrays.append(get_return_array(x))
        matrix = np.concatenate(arrays, axis=1)

    return np.cov(matrix.T)


data = [hangseng['Close'].tolist(), ha['Adj Close'].tolist(),
        rya['Adj Close'].tolist(), sky['Adj Close'].tolist()]

print(get_return_cov(data, log=True))
print(get_return_cov(data, log=False))
