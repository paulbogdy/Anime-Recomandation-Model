from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import pandas as pd
import time


class StatusBar:
    def __init__(self, size, max_size):
        self.value = 0
        self.size = size
        self.max_size = max_size

    def show(self):
        active = self.value/self.max_size*self.size
        str = "[" + "#"*int(active) + "."*int(self.size-active) + "] {}:{}, {:.2f}%".format(self.value, self.max_size, 100*self.value/self.max_size)
        print('\r', str, end='')

    def increment(self):
        self.value += 1


def get_src(id):
    site = "https://myanimelist.net/anime/"
    id = str(id)
    page = urlopen(site + id).read()
    soup = BeautifulSoup(page, 'html.parser')
    content = soup.find('div', {'id': 'content'})
    return content.findAll('img')[0]['data-src']


def download_img(src, name):
    urlretrieve(src, name)


if __name__ == '__main__':
    anime_list = pd.read_csv("../Preprocessed Data/anime.csv")
    dir = 'pics/'
    bar = StatusBar(20, len(anime_list))
    for _, anime in anime_list.iterrows():
        id = anime['MAL_ID']
        name = anime['Name']
        try:
            src = get_src(id)
            download_img(src, dir+name)
        except:
            print('\r', f'Unable to download {name}')
        bar.increment()
        bar.show()
        if bar.value % 50 == 0:
            time.sleep(60)
