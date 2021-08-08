from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from . import RecommenderNet
import pandas as pd
import matplotlib.pyplot as plt


def callbacks():
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00005

    rampup_epochs = 2
    sustain_epochs = 0
    exp_decay = .8

    def lrfn(epoch):
        if epoch < rampup_epochs:
            return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            return max_lr
        else:
            return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

    lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

    my_callbacks = [
        lr_callback,
    ]
    return my_callbacks


def train():
    anime = pd.read_csv("Preprocessed Data/anime.csv")
    ratings = pd.read_csv("Preprocessed Data/ratings.csv")

    user_ids = ratings['user_id'].unique().tolist()
    user2encoded = {x: i for i, x in enumerate(user_ids)}
    encoded2user = {i: x for i, x in enumerate(user_ids)}

    anime_ids = ratings['anime_id'].unique().tolist()
    anime2encoded = {x: i for i, x in enumerate(anime_ids)}
    encoded2anime = {i: x for i, x in enumerate(anime_ids)}

    ratings['user'] = ratings['user_id'].map(user2encoded)
    ratings['anime'] = ratings['anime_id'].map(anime2encoded)

    num_users = len(user_ids)
    num_anime = len(anime_ids)

    model = RecommenderNet(num_users, num_anime)

    max_rating = ratings['rating'].max()
    min_rating = ratings['rating'].min()

    ratings = ratings.sample(frac=1)
    x = ratings[['user', 'anime']].values
    y = ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    batch_size = 8192
    history = model.fit(
        x=[x[:, 0], x[:, 1]],
        y=y,
        batch_size=batch_size,
        epochs=8,
        verbose=1,
        callbacks=callbacks()
    )
    return model, history


def training_plot(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()
