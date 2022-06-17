import tensorflow as tf
from tensorflow import keras
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPool1D,
    Dropout,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


class DGCNN(tf.keras.Model):
    def __init__(
        self,
        graphs,
        layer_sizes=[1024, 1024, 1024, 512],
        k=20,
    ):
        super(DGCNN, self).__init__()

        # First we create the base DGCNN model that includes the graph convolutional and SortPooling layers.
        self.dgcnn_model = DeepGraphCNN(
            layer_sizes=layer_sizes,
            activations=["tanh", "tanh", "tanh", "tanh"],
            k=k,  # the number of rows for the output tensor (k = 10, 20)
            bias=False,
            generator=PaddedGraphGenerator(graphs),
        )

        self.conv_layer1 = Conv1D(
            filters=256, kernel_size=sum(layer_sizes), strides=sum(layer_sizes)
        )
        self.pooling_layer1 = MaxPool1D(pool_size=2)
        self.conv_layer2 = Conv1D(filters=512, kernel_size=5, strides=1)
        self.flatten_layer = Flatten()
        self.dense_layer1 = Dense(units=1024, activation="relu")
        self.dropout_layer = Dropout(rate=0.25)
        self.prediction = Dense(units=2, activation="softmax")

    def call(self):
        _, x_out = self.dgcnn_model.in_out_tensors()

        graphs_cnn_vector = x_out
        # Next, we add the convolutional, max pooling, and dense layers.
        x_out = self.conv_layer1(x_out)
        x_out = self.pooling_layer1(x_out)
        x_out = self.conv_layer2(x_out)
        x_out = self.flatten_layer(x_out)
        x_out = self.dense_layer1(x_out)
        x_out = self.dropout_layer(x_out)
        prediction = self.dgcnn_model.prediction(x_out)

        return prediction, graphs_cnn_vector
