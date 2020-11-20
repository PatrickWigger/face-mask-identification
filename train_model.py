from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Label encoding for building model
from sklearn.preprocessing import LabelEncoder


def train_model(model, data, labels, test_size, epochs):
    # Create label encoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y=labels)
    labels = to_categorical(y=labels)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)

    # Create Image Data Generator
    img_data_generator = ImageDataGenerator(zoom_range=0.1,
                                            rotation_range=25,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            shear_range=0.15,
                                            horizontal_flip=True,
                                            fill_mode="nearest")

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the head of the network
    model.fit(img_data_generator.flow(x=X_train, y=y_train, batch_size=1),
              steps_per_epoch=len(X_train),
              validation_data=(X_test, y_test),
              validation_steps=len(X_test),
              epochs=epochs)

    return model
