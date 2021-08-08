import pandas as pd
import numpy as np
import re
from tensorflow.keras.losses import cosine_similarity as similarity
import pickle


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
    def __init__(self, load_dir):
        self.anime_list = pd.read_csv(load_dir + "/anime_list.csv", index_col="MAL_ID")
        self.anime_list['User Rating'] = np.nan
        with open(load_dir + "/anime_weights.npy", "rb") as f:
            self.feature_matrix = np.load(f)

    def rate(self, anime_id, rating):
        self.anime_list.loc[anime_id, 'User Rating'] = rating

    def search_name(self, name, how_many=5):
        return self.anime_list[self.anime_list['Name'].apply(lambda x: matches(name, x))]\
            .sort_values(by='Name', key=lambda x: x.str.len()).head(how_many)

    def anime_id(self, name):
        return self.search_name(name, 1).index.values[0]

    def anime(self, anime_id):
        return self.anime_list.loc[anime_id]

    def similar(self, searched_name, how_many=15):
        anime = self.search_name(searched_name, 1)
        anime_id = anime.index.values[0]
        distance = similarity(self.feature_matrix, self.feature_matrix[anime_id])
        viewed_anime = self.anime_list.dropna().index.values
        if anime_id not in viewed_anime:
            viewed_anime = np.append(viewed_anime, [anime_id])
        top_x = pd.Series(distance).sort_values().drop(labels=viewed_anime).head(how_many)
        series = top_x.index.values
        return series
        """
        result = pd.DataFrame(columns=['Name', 'Genres', 'Score', 'Description'])
        for i, anime_id in enumerate(series):
            row = self.anime_list.loc[anime_id]
            new_row = {"Name": row['Name'],
                       "Genres": row['Genres'],
                       "Score": row['Score'],
                       "No. Episodes": row['Episodes'],
                       "Description": row['Description']}
            result.loc[i] = new_row
        return result
        """


def recommend_all():
    recommender = Recommender("..")
    recommendations = {}
    for anime_id in recommender.anime_list.index.values:
        distance = similarity(recommender.feature_matrix, recommender.feature_matrix[anime_id])
        viewed_anime = recommender.anime_list.dropna().index.values
        if anime_id not in viewed_anime:
            viewed_anime = np.append(viewed_anime, [anime_id])
        top_x = pd.Series(distance).sort_values().drop(labels=viewed_anime).head(15)
        recommendations[anime_id] = top_x.index.values
    return recommendations


if __name__ == '__main__':
    """
    solutions = recommend_all()
    pickle.dump(solutions, open('../recommender/recommendations.p', "wb"))
    """
    sol = pickle.load(open('../recommender/recommendations.p', "rb"))
    recommender = Recommender("..")
    print(recommender.anime(sol[63]))


