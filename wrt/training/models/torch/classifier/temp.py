def convnet(input_shape, num_classes):
    seq_layers = [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),

        layers.Dropout(0.2),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(512, activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
        ]
    model = Sequential(seq_layers)
    return model