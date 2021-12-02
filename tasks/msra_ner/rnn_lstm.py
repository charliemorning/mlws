from tasks.msra_ner import prepare_for_data
from nlp.diagrams.neruel_network.classical.rnn_crf import LSTMSeqCls, SeqClsModelConfig
from nlp.diagrams.neruel_network.trainer import SupervisedNNModelTrainConfig, PyTorchTrainer


def main():

    xs_seq_train, ys_index_train, xs_seq_test, ys_index_test, word_index, label_index = prepare_for_data()

    train_config = SupervisedNNModelTrainConfig(
        learning_rate=0.005,
        epoch=80,
        train_batch_size=256,
        eval_batch_size=256
    )

    model_config = SeqClsModelConfig(
        50000,
        dim_out=7
    )

    model = LSTMSeqCls(config=model_config)


    trainer = PyTorchTrainer(
        model=model,
        train_config=train_config,
        loss_fn=model.loss_fn
    )

    trainer.fit(train_data=(xs_seq_train, ys_index_train), eval_data=(xs_seq_test, ys_index_test))


if __name__ == '__main__':
    main()