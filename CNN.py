def create_network(input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     input_shape=input_shape))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', padding='same', ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(1000))
    model.add(Dense(200))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    return Model(input, x)
