import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dot, Add, Activation, BatchNormalization, Flatten, Input, Dense


def RecommenderNet(num_users, num_anime):
    """
    For this problem, I used a collaborative filtering approach:
        -There are a lot of ratings, so it makes sense to let the model learn from only those
        -The idea is to use matrix factorization, splitting the matrix of ratings (user, anime)
        into 2 feature matrices, (user, feature), (feature, anime), and when multiplying those 2,
        there should be a good approximation of the matrix of ratings we have from the beginning
        -I also used 2 more biases embeddings, for a more unbiased presentation of anime features

    :param num_users: integer, the number of unique user id's in the ratings
    :param num_anime: integer, the number of unique anime id's in the ratings
    :return: keras.model, the final model
    """
    embedding_size = 32

    user = Input(name='user', shape=[1])
    anime = Input(name='anime', shape=[1])

    user_embedding = Embedding(name='user_embedding',
                               input_dim=num_users,
                               output_dim=embedding_size,
                               embeddings_initializer="he_normal",
                               embeddings_regularizer=keras.regularizers.l2(1e-6))(user)
    user_bias = Embedding(name='user_bias',
                          input_dim=num_users,
                          output_dim=1)(user)

    anime_embedding = Embedding(name='anime_embedding',
                                input_dim=num_anime,
                                output_dim=embedding_size,
                                embeddings_initializer="he_normal",
                                embeddings_regularizer=keras.regularizers.l2(1e-6))(anime)
    anime_bias = Embedding(name='anime_bias',
                           input_dim=num_anime,
                           output_dim=1)(anime)

    x = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    x = Add(name='added')([x, anime_bias, user_bias])
    x = Flatten()(x)

    x = Dense(1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=[user, anime], outputs=x)
    model.compile(loss='binary_crossentropy', metrics=["mae", "mse"], optimizer='Adam')
    return model
