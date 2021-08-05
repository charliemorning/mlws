import argparse

from framework.torch.layers.transformer import Transformer as Transformer_

from train.cross_valid import CVFramework
from train.trainer import SupervisedNNModelTrainConfig
from train.torch.trainer import PyTorchTrainer
from train.torch.nlp.network.fast_text import FastTextConfig, FastText
from train.torch.nlp.network.cnn import PoolingType, TextCNNModelConfig, TextCNN
from train.torch.nlp.network.rnn import TextRNNModelConfig, TextRNN
from train.torch.nlp.network.transformer import TransformerConfig, Transformers
from tasks.ccf.datagrand2021.data_helper import load_labelled_data_as_dataset



def fast_text_model(vocab_size):

    config = FastTextConfig(
        max_feature=vocab_size + 1,
        embedding_size=300,
        dim_out=35
    )

    return FastText(config)


def main(data_home):
    dataset = load_labelled_data_as_dataset(data_home)

    config = SupervisedNNModelTrainConfig(
        learning_rate=0.005,
        epoch=1000,
        train_batch_size=128,
        eval_batch_size=128,
        binary_out=False
    )

    model = fast_text_model(dataset.vocab_size())

    trainer = PyTorchTrainer(
        model=model,
        train_config=config
    )

    cv = CVFramework(trainer)
    cv.validate(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train cls models on bert.')
    parser.add_argument('data_home', type=str)
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')

    args = parser.parse_args()

    data_home = args.data_home
    main(data_home)
