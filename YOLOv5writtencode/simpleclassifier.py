import tensorflow as tf
import tensorflow.keras.layers as layer

model = tf.keras.Sequential([layer.Flatten(input_shape=(1920,1080)),
                            layer.Conv2D(filters=32,kernel_size=10,strides=3,padding='same',activation='relu'),
                            layer.MaxPool2D(pool_size=(4,4),strides=2),
                            layer.Conv2D(filters=64,kernel_size=10,strides=3,padding='same',activation='relu'),
                            layer.MaxPool2D(pool_size=(4,4),strides=2),
                            layer.Dropout(0.1),
                            layer.Dense(14,activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels,  epochs = 100)

model.save('model/save.h5')