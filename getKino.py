import urllib.request

for i in range(2004, 2019):
    for j in range(1, 13):
        if j < 10:
            url = "https://www.opap.gr/Excel/1100/{}/kino_{}_0{}.xls".format(i, i, j)
        else:
            url = "https://www.opap.gr/Excel/1100/{}/kino_{}_{}.xls".format(i, i, j)
        urllib.request.urlretrieve(url, './res/kino{}{}.xls'.format(i, j))
        print("Got {}-{}".format(i, j))
