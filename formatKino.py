import pandas as pd
import numpy as np
import glob
import csv
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt

print("Data pre-procession")

csv_data = pd.DataFrame()
csvfiles = glob.glob("./res/*.csv")
for f in csvfiles:
    i = 0
    df = pd.read_csv(f, header=i, sep=";", usecols=range(1, 23))
    while df.columns.values[0] == "Unnamed: 1":
        i += 1
        df = pd.read_csv(f, header=i, sep=";", usecols=range(1, 23))

    df.rename(columns={"Ημ/νία Κλήρωσης": 'date', "Ώρα Κλήρωσης": 'time'}, inplace=True)
    for i in range(1, 23):
        df.rename(columns={"{}ος ".format(i): i}, inplace=True)

    if df.columns.values[-1] != 20:
        print("Lost {}".format(f))
        print(df.columns)
        exit(0)
    csv_data = pd.concat([csv_data, df], ignore_index=True, sort=False)


def formTime(x):
    nx = str(x).split(":")
    return int(nx[0])*60 + int(nx[1])


def formDate(x):
    nx = str(x).split("/")
    return int(nx[0]) + (int(nx[1]) - 1)*30 + int(nx[2])

def map80(l20):
    lst = [0] * 80
    for element in l20:
        lst[element - 1] = 1
    return lst


time = csv_data["time"].tolist()
time = map(lambda x: formTime(x), time)
csv_data["time"] = list(time)

date = csv_data["date"].tolist()
date = map(lambda x: formDate(x), date)
csv_data["date"] = list(date)

train_data = csv_data[["time", "date"]].values
train_data = normalize(train_data)

test_data = train_data[0:1000]

train_data = np.delete(train_data, range(0, 1000), 0)


train_labels = csv_data.drop(["time", "date"], axis=1).values
train_labels = map(lambda x: map80(x), train_labels)
train_labels = np.array(list(train_labels))

test_labels = train_labels[0:1000]

train_labels = np.delete(train_labels, range(0, 1000), 0)

np.savetxt("train_data.csv", train_data, delimiter=",", fmt='%f', comments='')
np.savetxt("test_data.csv", test_data, delimiter=",", fmt='%f', comments='')
np.savetxt("train_labels.csv", train_labels, delimiter=",", fmt='%d', comments='')
np.savetxt("test_labels.csv", test_labels, delimiter=",", fmt='%d', comments='')

print("Data ready to be processed")
