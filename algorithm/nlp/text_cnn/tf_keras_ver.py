import tensorflow as tf

from algorithm.nlp.framework import KerasTrainFramework


class TextCNN(KerasTrainFramework):

    def __init__(self
                 , config
                 , vocab_size: int
                 , input_size: int
                 , embed_size: int
                 , filters: int
                 , kernel_size: int
                 , output_dim: int
                 , embedding_matrix=None
                 , embedding_trainable=False
                 , metric_callbacks=None
                 ):

        KerasTrainFramework.__init__(self, config)

        input_ = tf.keras.layers.Input(shape=(input_size,))

        if embedding_matrix is None:
            x = tf.keras.layers.Embedding(vocab_size, embed_size)(input_)
        else:
            x = tf.keras.layers.Embedding(vocab_size + 1, embed_size, weights=[embedding_matrix], trainable=embedding_trainable)(input_)

        x = tf.keras.layers.SpatialDropout1D(0.5)(x)

        x = tf.keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(x)

        x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dropout(0.5)(x)

        if output_dim == 1:
            output = tf.keras.layers.Dense(output_dim, activation="sigmoid")(x)
        else:
            output = tf.keras.layers.Dense(output_dim, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_, outputs=output)

        if output_dim == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'

        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=metric_callbacks)

        self.model = model

        print(self.model.summary())
