import tensorflow as tf

def train(model, train_data, epochs=10, batch_size=1):

    # I feel like these could go in a config file or kwargs
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.RootMeanSquaredError()]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # print the model summary
    

    # print(f"Training data {train_data}")

    # batch the data
    train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # print(f"Training data after batching: {train_data}")

    # train the model
    model.fit(train_data, epochs=epochs)

    return model