import argparse

from playground.nlp.framework import Transformer as Transformer_

from train.cross_valid import CVFramework
from nlp.trainer import SupervisedNNModelTrainConfig
from nlp.diagrams.neruel_network.trainer import PyTorchTrainer
from nlp.diagrams.neruel_network.classical.fast_text import FastTextConfig, FastText
from nlp.diagrams.neruel_network.classical.cnn import PoolingType, TextCNNModelConfig, TextCNN
from nlp.diagrams.neruel_network.classical.rnn import TextRNNModelConfig, TextRNN
from nlp.diagrams.neruel_network.classical.transformer import TransformerConfig, Transformers
from tasks.kaggle.disaster_tweets.data_helper import prepare_data_for_cnn_and_rnn


def fast_text_model(vocab_size, embedding_matrix):

    config = FastTextConfig(
        max_feature=vocab_size + 1,
        embedding_size=300,
        dim_out=1
    )

    return FastText(config)


def cnn_model(vocab_size, embedding_matrix):

    cnn_model_config = TextCNNModelConfig(
        max_features=vocab_size + 1,
        max_seq_length=180,
        embedding_size=300,
        filters=[100, 100, 100],
        kernel_size=[3, 4, 5],
        padding=[0, 0, 0],
        dilation=[1, 1, 1],
        stride=[1, 1, 1],
        dim_out=1,
        pooling_type=PoolingType.MAX_POOLING,
        global_pooling=True,
        without_pretrained=False,
        freeze_pretrained=True
    )

    model = TextCNN(
        cnn_model_config=cnn_model_config,
        embedding_matrix=embedding_matrix
    )

    return model


def rnn_model(vocab_size, embedding_matrix):

    model_config = TextRNNModelConfig(
        max_features=vocab_size + 1,
        max_seq_length=128,
        embedding_size=300,
        dim_out=1
    )

    model = TextRNN(model_config=model_config, embedding_matrix=embedding_matrix)

    return model


def transformer_model(word_index, embedding_matrix):
    config = TransformerConfig(len(word_index) + 1)
    return Transformers(config, embedding_matrix)


def transformer_model_(word_index):
    return Transformer_(
        len(word_index) + 1,
        2
    )


def main(data_home, model_path):
    dataset, embedding_matrix = prepare_data_for_cnn_and_rnn(data_home, model_path)

    config = SupervisedNNModelTrainConfig(
        learning_rate=0.005,
        epoch=1000,
        train_batch_size=128,
        eval_batch_size=128,
        binary_out=True
    )

    model = fast_text_model(dataset.vocab_size(), embedding_matrix)

    trainer = PyTorchTrainer(
        model=model,
        train_config=config
    )

    cv = CVFramework(trainer)
    cv.validate(dataset)
    # loss, acc, prec, recall, f1 = trainer.fit(train_data=(xs_word_index_train, ys_train), eval_data=(xs_word_index_eval, ys_eval))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train cls models on bert.')
    parser.add_argument('data_home', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')

    args = parser.parse_args()

    data_home = args.data_home
    model_path = args.model_path
    main(data_home, model_path)
