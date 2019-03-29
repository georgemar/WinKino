import pandas as pd
import glob

xlsfiles = glob.glob("./res/*.xls")
for f in xlsfiles:
    print(f)
    df = pd.DataFrame
    df = pd.read_excel(f, header=2, usecols=range(1, 23), no_index=True)
    if df.columns.values[0] != "Ημ/νία Κλήρωσης":
        df = pd.read_excel(f, header=1, usecols=range(1, 23), no_index=True)
    path = f.split(".xls")[0] + ".csv"
    df.to_csv(path, sep=";")

