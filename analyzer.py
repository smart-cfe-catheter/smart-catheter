import matplotlib.pyplot as plt
from formatter import formatter

data = formatter(32, time_series=False, ignore_nan=True)

data.hist()
plt.show()
