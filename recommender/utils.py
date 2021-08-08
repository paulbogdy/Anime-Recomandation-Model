import pickle
import re
import numpy as np
import pandas as pd


def matches(str1, str2):
    for word in re.findall(r'\w+', str1):
        ok = False
        for second_word in re.findall(r'\w+', str2):
            if word.lower() in second_word.lower():
                ok = True
                break
        if not ok:
            return False
    return True


class Recommender:
    def __init__(self):
        self.__anime_list = pd.read_csv("anime_list.csv", index_col='MAL_ID')
        self.__recommendations = pickle.load(open('recommendations.p', "rb"))

    def search_name(self, name, how_many=5):
        return self.__anime_list[self.__anime_list['Name'].apply(lambda x: matches(name, x))] \
                .sort_values(by='Name', key=lambda x: x.str.len()).head(how_many)

    def anime_id(self, name):
        anime = self.search_name(name, 1)
        if anime.empty:
            return -1
        else:
            return anime.index.values[0]

    def anime(self, anime_id):
        return self.__anime_list.loc[anime_id]

    def top_x(self, anime_id):
        result = self.anime(self.__recommendations[anime_id]).copy()
        return result
