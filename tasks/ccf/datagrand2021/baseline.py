import argparse
import os
import numpy as np

import torch

from framework.torch.layers.transformer import Transformer as Transformer_

from train.cross_valid import CVFramework
from train.trainer import SupervisedNNModelTrainConfig
from train.torch.trainer import PyTorchTrainer
from train.torch.nlp.network.fast_text import FastTextConfig, FastText
from train.torch.nlp.network.cnn import PoolingType, TextCNNModelConfig, TextCNN
from train.torch.nlp.network.rnn import TextRNNModelConfig, TextRNN
from train.torch.nlp.network.transformer import TransformerConfig, Transformers
from tasks.ccf.datagrand2021.data_helper import *
from util.model import predict
from util.model import rebuild_embedding_index, build_embedding_matrix


TRAIN_DATA_FILENAME = "datagrand_2021_train.csv"
TEST_DATA_FILENAME = "datagrand_2021_test.csv"
MODEL_FILENAME = "word2vec.skipgram.unigram.300d.txt"


def fast_text_model(vocab_size, dim_out):

    config = FastTextConfig(
        max_feature=vocab_size + 1,
        embedding_size=100,
        dim_out=dim_out
    )

    return FastText(config)


def cnn_model(vocab_size, embedding_matrix, dim_out):

    cnn_model_config = TextCNNModelConfig(
        max_features=vocab_size + 1,
        max_seq_length=128,
        embedding_size=300,
        filters=[100, 100, 100],
        kernel_size=[3, 4, 5],
        padding=[0, 0, 0],
        dilation=[1, 1, 1],
        stride=[1, 1, 1],
        dim_out=dim_out,
        pooling_type=PoolingType.MAX_POOLING,
        global_pooling=True,
        without_pretrained=True

    )

    model = TextCNN(
        cnn_model_config=cnn_model_config,
        embedding_matrix=embedding_matrix
    )

    return model


def rnn_model(vocab_size, embedding_matrix, dim_out):

    model_config = TextRNNModelConfig(
        max_features=vocab_size + 1,
        max_seq_length=128,
        embedding_size=300,
        dim_out=dim_out,
        without_pretrained=True
    )

    model = TextRNNAtt(model_config=model_config, embedding_matrix=embedding_matrix)

    return model


def main(data_home):

    word_embeddings = load_word_embeddings(os.path.join(data_home, MODEL_FILENAME))

    train_dataset = load_labelled_data_as_dataset(os.path.join(data_home, TRAIN_DATA_FILENAME))
    print(train_dataset.stats())
    print(train_dataset.label_dist())

    # vocab = Vocabulary([word_embeddings.keys()])
    # iv, ivc, oov, oovc = vocab.check_coverage(train_dataset.get_vocab())
    # print("iv count: %d; oov count: %d" % (ivc, oovc))
    # print("oov rate: %f" % (oovc / float(ivc + oovc)))

    embedding_index = rebuild_embedding_index(word_embeddings, train_dataset.get_vocab().word_index())
    embedding_matrix = build_embedding_matrix(word_embeddings, train_dataset.get_vocab().word_index())

    # emb_mean, emb_std = embedding_matrix.mean(), embedding_matrix.std()
    # embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    # embedding_matrix = None

    config = SupervisedNNModelTrainConfig(
        learning_rate=0.01,
        epoch=10,
        train_batch_size=128,
        eval_batch_size=128,
        binary_out=False,
        patience=5
    )

    model = rnn_model(train_dataset.vocab_size(), embedding_matrix, len(np.unique(train_dataset.get_labels())))
    print(model)

    trainer = PyTorchTrainer(
        model=model,
        train_config=config
    )

    # cv = CVFramework(trainer)
    # cv.validate(train_dataset)

    xs_train = ((train_dataset.get_sequences(), train_dataset.get_sequence_lengths()), train_dataset.get_labels())
    trainer.fit(xs_train)

    test_dataset = load_test_data_as_dataset(os.path.join(data_home, TEST_DATA_FILENAME), train_dataset.get_vocab(), train_dataset.get_label_encoder())

    print(test_dataset.vocab_coverage())
    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    print(test_dataset.get_oov())
    print(test_dataset.stats())

    xs_test = (test_dataset.get_sequences(), test_dataset.get_sequence_lengths())
    _, preds = trainer.predict(xs_test)
    preds = train_dataset.get_label_encoder().decode(preds)
    make_submissions(os.path.join(data_home, "submission.csv"), preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train cls models on bert.')
    parser.add_argument('data_home', type=str)
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')

    args = parser.parse_args()

    data_home = args.data_home
    main(data_home)
