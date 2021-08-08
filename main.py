from processing import preprocess
from model.training import train, training_plot


if __name__ == '__main__':
    # It takes a lot of time this way, I really recommend to use Google Collab, with TPU accelerator

    preprocess('Raw Data', 'Preprocessed Data')
    model, history = train()
    training_plot(history)
