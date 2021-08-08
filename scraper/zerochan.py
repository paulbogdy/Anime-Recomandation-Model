from bs4 import BeautifulSoup
from urllib.request import urlopen


def adjust_name(name):
    return name.replace(" ", '+')


def get_image(url):
    page = urlopen(url).read()
    soup = BeautifulSoup(page, 'html.parser')
    menu = soup.find('div', {"id": "menu"})
    img = menu.find_all('img')[0]
    return img


if __name__ == '__main__':
    url = 'https://www.zerochan.net/'
    anime_name = 'attack+on+titan'
    url = url + anime_name
    try:
        src = get_image(url)
    except:
        print("COX")

