import tensorflow as tf

from models.torch.trainer import SupervisedNNModelTrainConfig, KerasTrainer
from models.torch.nlp.text_cnn import TextCNNModelConfig


class TextCNNTrainer(KerasTrainer):

    def __init__(
            self,
            train_config: SupervisedNNModelTrainConfig,
            cnn_model_config: TextCNNModelConfig,
            embedding_matrix=None,
            embedding_trainable=False,
            metric_callbacks=None
    ):
        super(TextCNNTrainer, self).__init__(train_config)
        self.cnn_model_config = cnn_model_config

        input_ = tf.keras.layers.Input(shape=(cnn_model_config.max_seq_length,))

        if embedding_matrix is None:
            x = tf.keras.layers.Embedding(cnn_model_config.max_features, cnn_model_config.embedding_size)(input_)
        else:
            x = tf.keras.layers.Embedding(
                cnn_model_config.max_features + 1,
                cnn_model_config.embedding_size,
                weights=[embedding_matrix],
                trainable=embedding_trainable)(input_)

        x = tf.keras.layers.SpatialDropout1D(0.5)(x)

        x = tf.keras.layers.Conv1D(cnn_model_config.filters, 3, activation='relu', padding='same')(x)

        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        if cnn_model_config.dim_out == 1:
            output = tf.keras.layers.Dense(cnn_model_config.dim_out, activation="sigmoid")(x)
        else:
            output = tf.keras.layers.Dense(cnn_model_config.dim_out, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_, outputs=output)

        if cnn_model_config.dim_out == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'

        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=metric_callbacks)

        self.model = model

        print(self.model.summary())
