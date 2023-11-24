import tensorflow as tf


def main():
    mnist = tf.keras.datasets.mnist
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xtrain, xtest = xtrain / 255.0, xtest / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ]
    )
    predictions = model(xtrain[:1]).numpy()
    print(tf.nn.softmax(predictions).numpy())

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=5)
    model.evaluate(xtest,  ytest, verbose=2)


if __name__ == '__main__':
    main()
