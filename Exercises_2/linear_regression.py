import statsmodels.api as sm
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import abline_plot

def dataToReg(y, x):
    regX = sm.add_constant(x)
    model = sm.OLS(y, regX)
    tres = model.fit()
    
    print(tres.summary())

    f = np.poly1d(tres.params[::-1])

    spacing = np.linspace(min(x), max(x), max(x))

    plt.plot(f(spacing).astype(np.int))

data = pandas.read_csv('auto.csv', sep=',', header=0, index_col=False, dtype={
    'mpg': np.float32,  
    'cylinders': np.float32,
    'displacement': np.float32,
    'horsepower': np.float32,
    'weight': np.float32,
    'acceleration': np.float32,
    'model_year': np.int32,
    'origin': np.int32
    })

values = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
colors = ['blue', 'magenta', 'green', 'red', 'orange', 'purple', 'cyan']

def plot_data():
    for i in range(7):
        plt.subplot(3, 3, i + 1)
        plt.scatter(data.get(values[i]).get_values(), data.get('mpg').get_values(), c=colors[i])
        plt.ylabel('mpg')
        plt.xlabel(values[i])
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(left=0.04, wspace=0.16, hspace=0.16, top=1, right=1, bottom=0.04)
    plt.show()

print("""
//////////
// Horsepower
//////////
""")

regY = data.get('mpg').get_values()

model = sm.OLS(regY, sm.add_constant(data['horsepower']))
results = model.fit()
p = results.params
print(results.summary())

# scatter-plot data
ax = data.plot(x='horsepower', y='mpg', kind='scatter')

# plot regression line
abline_plot(model_results=results, ax=ax)

#plt.show()

print("""
//////////
// All features
//////////
""")
df = data[values]

model = sm.OLS(regY, sm.add_constant(df))
results = model.fit()
print(results.summary())

#plot_data()