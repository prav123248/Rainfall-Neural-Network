
import pandas as pd

rmin = 0
rmax = 0
def standardize(val):
    return 0.8 * ((val-rmin)/(rmax-rmin)) + 0.1

#Loading semi-raw Data
data = pd.read_excel("rawData.xlsx")
#print(data.info())

#Remove any rows containing empty or non-numeric data and remove rows with outliers
for column in data:
    data = data[pd.to_numeric(data[column], errors="coerce").notnull()]
    upperQuartile = data[column].quantile(0.99)
    lowerQuartile = data[column].quantile(0.01)
    data = data[(data[column] < upperQuartile) & (data[column] > lowerQuartile)]

    #Standardizing columns
    rmin = data[column].min()
    rmax = data[column].max()
    data[column] = data[column].map(standardize)
    
#Splitting into subsets
data = data.reset_index(drop=True)
rowCount = len(data.index)
train = data.loc[:(60/100 * rowCount)+1]
validation = data.loc[(60/100 * rowCount)+1:1+(80/100 * rowCount)]
test = data.loc[(80/100 * rowCount)+1:]


#Saving column to separate files
train.to_excel("train.xlsx", index=False)
validation.to_excel("validation.xlsx", index=False)
test.to_excel("test.xlsx",index=False)

print("Finished")
