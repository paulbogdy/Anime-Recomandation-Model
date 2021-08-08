import pandas as pd


def preprocess(raw_dir, proc_dir):
    """
        Function for preprocessing the raw data
        Due to the fact that the dataset is very large and I am going to use Collaborative filtering,
        irrelevant data should be removed

        Irrelevant data chosen:
            - Ratings of users that watched less than 2 episodes of that anime
            - Ratings of users that didn't rate at leas 100 other anime (that is for collaborative filtering)
            - Columns of ratings that gave the status and the number of episodes watched
            - Only kept anime columns: Name, Id, Score, Nr. Episodes, Genre and Description
    """

    print("Reading Raw Data...", end=' ')
    anime = pd.read_csv(raw_dir + "/anime.csv")
    anime_synopsis = pd.read_csv(raw_dir + '/anime_with_synopsis.csv')
    ratings = pd.read_csv(raw_dir + "/animelist.csv")
    print("Done")

    print("Selecting only TV anime...", end=' ')
    tv_anime = anime[anime['Type'] == 'TV'].copy()
    print('Done')

    print("Selecting the English anime name...", end=' ')
    tv_anime['Name'] = tv_anime.apply(lambda x: x['English name'] if x['English name'] != 'Unknown' else x['Name'],
                                      axis=1).copy()
    print("Done")

    print("Keeping only the relevant properties of anime...", end=' ')
    relevant_anime = tv_anime[['MAL_ID', 'Name', 'Score', 'Genres', 'Episodes']].copy()
    synopsis = pd.Series(anime_synopsis.sypnopsis.values, index=anime_synopsis.MAL_ID).to_dict()
    relevant_anime['Description'] = relevant_anime['MAL_ID'].apply(lambda x: synopsis[x])
    print("Done")

    """
    I've decided to keep the ratings where the person watched only one or 0 episodes
    because it seems that the model gets better results with this data,
    I believe it's because people don't always select the number of episodes watched.
    
    print("Removing irrelevant ratings (less than 2 episodes watched)...", end=' ')
    ratings.query('watched_episodes > 1', inplace=True)
    print("Done")
    """

    print("Removing ratings of invalid anime...", end=' ')
    valid_id_dict = {}
    for id in ratings['anime_id'].unique():
        valid_id_dict[id] = False
    for id in relevant_anime['MAL_ID'].values:
        valid_id_dict[id] = True
    ratings_mask = ratings['anime_id'].map(valid_id_dict)
    ratings = ratings[ratings_mask]
    print("Done")

    print("Removing inactive Users (less than 100 anime rated)...", end=' ')
    n_ratings = ratings['user_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(n_ratings[n_ratings >= 100].index)].copy()
    ratings.drop(['watching_status', 'watched_episodes'], axis=1, inplace=True)
    print("Done")

    print("Removing anime with no ratings...", end=' ')
    valid_id_dict = {}
    for id in relevant_anime['MAL_ID'].values:
        valid_id_dict[id] = False
    for id in ratings['anime_id'].unique():
        valid_id_dict[id] = True
    relevant_anime = relevant_anime[relevant_anime['MAL_ID'].map(valid_id_dict)]
    print("Done")

    print("Saving data...", end=' ')
    relevant_anime.to_csv(proc_dir + '/anime.csv', index=False)
    ratings.to_csv(proc_dir + '/ratings.csv', index=False)
    print("Done")

