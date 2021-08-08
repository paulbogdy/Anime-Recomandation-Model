# Anime-Recomandation-Model

Website available at: https://anime-recommender-5k.herokuapp.com

Dataset used for this project: https://www.kaggle.com/hernan4444/anime-recommendation-database-2020

The Project is composed of 4 modules: Data analysis and preprocessing (processing), Model creation and training (model),
Website (recommender), Picture scraper from myanimelist (scraper)

The goal of the Project is to find 15 similar anime for any given anime (only works on TV anime)

##Solution:

The proposed solution is using collaborative filtering, as there are 70m ratings from 200k+ users for 5k anime.
I've used matrix factorization: The idea is simple, we want to reduce the initial sparse matrix of 70m ratings (A), into
2 matrices (B, C) so that B*C = A, for doing that we want to minimize the cost of the error, where error means
how bad the factorization approximation is.

After that we now have 2 matrices, where one represents vectors of features for users, and one for anime. We only
care about the one for anime, as we want to find similarities between those. That is done by using cosine
similarity between vectors (it's better than using euclidean distance).

There is also a scraper, to get an image for each anime, because the front end definitely looks better this way :)    

