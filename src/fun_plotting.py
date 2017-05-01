import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


lr = [0.00001, 0.0001, 0.001]

transfer_pos_precision = [0.2485, 0.2594, 0.1071]
random_pos_precision = [0.4455, 0.6627, 0.3026]

transfer_pos_recall = [0.6833, 0.9167, 0.7333]
random_pos_recall = [0.75, 0.9167, 0.9583]

pos_precision = [0.2485, 0.4455, 0.2594, 0.6627, 0.1071, 0.3026]
pos_recall = [0.6833, 0.75, 0.9167, 0.9167, 0.7333, 0.9583]


learning_rates = [0.00001, 0.00001, 0.00001, 0.00001,
                  0.0001, 0.0001, 0.0001, 0.0001,
                  0.001, 0.001, 0.001, 0.001]
weights = ['transfer','transfer', 'random', 'random',
         'transfer','transfer', 'random', 'random',
         'transfer','transfer', 'random', 'random']

values = [0.2485, 0.4455, 0.6833, 0.75,
          0.2594, 0.6627, 0.9167, 0.9167,
          0.1071, 0.3026, 0.7333, 0.9583]




plt.plot(lr, transfer_pos_precision, color='red', label='transfer')
plt.plot(lr, random_pos_precision)

plt.plot(lr, transfer_pos_recall)
plt.plot(lr, random_pos_recall)

plt.xlabel('learning rate')


plt.show()


# precision = np.array([learning_rates, weights, values, units])
# precision = np.transpose(precision)
#
# pre = pd.DataFrame(data=precision, columns=['learning rate', 'weights', 'values', 'dummy'])
# # gammas = sns.load_dataset("gammas")
#
# sns.tsplot(pre, time='learning rate', value='values', condition='weights', unit='dummy')
# sns.plt.show()