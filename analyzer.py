import matplotlib.pyplot as plt
from formatter import formatter

data = formatter(32, time_series=False, ignore_nan=True)

# data.hist(bins=1000)
data.boxplot(column='sensor3')
# plt.matshow(data.corr())

plt.show()
