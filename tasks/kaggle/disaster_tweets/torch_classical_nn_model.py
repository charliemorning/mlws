import torch

from models.trainer import SupervisedNNModelTrainConfig
from models.torch.trainer import PyTorchTrainer
from models.torch.nlp.network.fast_text import FastTextConfig, FastText
from models.torch.nlp.network.cnn import PoolingType, TextCNNModelConfig, TextCNN
from models.torch.nlp.network.rnn import TextRNNModelConfig, TextRNN

from tasks.kaggle.disaster_tweets import prepare_data_for_cnn_and_rnn


def fast_text_model(word_index, embedding_matrix):

    config = FastTextConfig(
        max_feature=len(word_index) + 1,
        embedding_size=300,
        dim_out=1
    )

    return FastText(config)


def cnn_model(word_index, embedding_matrix):

    cnn_model_config = TextCNNModelConfig(
        max_features=len(word_index) + 1,
        max_seq_length=180,
        embedding_size=300,
        filters=[100, 100, 100],
        kernel_size=[3, 4, 5],
        padding=[0, 0, 0],
        dilation=[1, 1, 1],
        stride=[1, 1, 1],
        dim_out=1,
        pooling_type=PoolingType.AVG_POOLING,
        global_pooling=True,
        without_pretrained=False,
        freeze_pretrained=True
    )

    model = TextCNN(
        cnn_model_config=cnn_model_config,
        embedding_matrix=embedding_matrix
    )

    return model


def rnn_model(word_index, embedding_matrix):

    model_config = TextRNNModelConfig(
        max_features=len(word_index) + 1,
        max_seq_length=128,
        embedding_size=300,
        dim_out=1
    )

    model = TextRNN(model_config=model_config, embedding_matrix=embedding_matrix)

    return model


def main():
    xs_word_index_train, ys_train, xs_word_index_eval, ys_eval, embedding_matrix, word_index\
        = prepare_data_for_cnn_and_rnn()

    config = SupervisedNNModelTrainConfig(
        learning_rate=0.005,
        epoch=1000,
        train_batch_size=10240,
        eval_batch_size=10240,
        device="cpu",
        binary_out=True
    )

    model = rnn_model(word_index, embedding_matrix)

    trainer = PyTorchTrainer(
        model=model,
        train_config=config
    )

    trainer.fit(train_data=(xs_word_index_train, ys_train), eval_data=(xs_word_index_eval, ys_eval))


if __name__ == "__main__":
    main()