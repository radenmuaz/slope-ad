import os
import urllib
_DATA = "/tmp/slope_data/"


def download(url, filename):
    if not os.path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = os.path.join(_DATA, filename)
    if not os.path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")