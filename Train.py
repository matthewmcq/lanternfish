import tensorflow as tf

def train(model, y_train, y_true, epochs=10, batch_size=1):

    # I feel like these could go in a config file or kwargs
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    model.summary()

    # print the model summary
    

    # print(f"Training data {train_data}")
    # print(f"Training data shape: {train_data}")

    # batch the data
    # train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # print(f"Training data after batching: {train_data}")

    # train the model
    model.fit(x=y_train, y=y_true,  epochs=epochs, batch_size =batch_size, shuffle=True, validation_split=0.2)
    return model