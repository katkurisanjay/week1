
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

warnings.filterwarnings('ignore')

train_path = "DATASET/TRAIN"
test_path = "DATASET/TEST"

def load_images():
    # visualization
    x_data = []
    y_data = []
    for category in glob.glob(train_path+'/*'):
        for file in tqdm(glob.glob(category+'/*')):
            img_array = cv2.imread(file)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x_data.append(img_array)
            y_data.append(category.split('/')[-1])
    data = pd.DataFrame({'image':x_data, 'label':y_data})

    data.shape

    colors = ['#800080', '#008080']
    plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', colors=colors, startangle= 90, explode=[0.05, 0.05])
    plt.show()

    colors = ['#FFD700', '#000080']
    plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', colors=colors, startangle= 180, explode=[0.05, 0.05])
    plt.show()

    colors = ['#FFD2DC', '#F08080']
    plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', colors=colors, startangle= 270, explode=[0.05, 0.05])
    plt.show()

    colors = ['#40E0D0', '#FF69B4']
    plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', colors=colors, startangle= 360, explode=[0.05, 0.05])
    plt.show()

    plt.figure(figsize=(20, 15))
    for i in range(9):
        plt.subplot(4, 3, (i%12)+1)
        index = np.random.randint(15000)
        plt.title('index is of {0}'.format(data.label[index]))
        plt.imshow(data.image[index])
        plt.tight_layout()

    """## CNN - Convolutional Neural Network

    """


    # Define the CNN model
    model = Sequential()

    # Add convolutional and pooling layers
    model.add(Conv2D(32, (3, 3),
    input_shape=(224, 224, 3)))  # Input shape for 224x224 RGB images
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    # Flatten the features for the dense layers
    model.add(Flatten())

    # Fully connected dense layers with dropout
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(2))  # Assuming two output classes
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    batch_size = 64

    # Print model summary
    model.summary()


    # Define image data generators
    train_datagen = ImageDataGenerator(rescale= 1./225)

    # Define image data generators
    test_datagen = ImageDataGenerator(rescale= 1./225)

    # Define dataset paths
    train_path = "DATASET/TRAIN"  # Update with actual path if needed
    test_path = "DATASET/TEST"    # Update with actual path if needed

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical"
    )

    # hist = model.fit(
    #     train_generator,
    #     epochs = 10,
    #     validation_data = test_generator
    #     )
    hist = model.fit(train_generator,
                    validation_data=(test_generator),
                    epochs=10,
                    batch_size=64)

    plt.figure(figsize=(10,6))
    plt.plot(hist.history["accuracy"], label="Train Accuracy")
    plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(hist.history["loss"], label="Training Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    def predict_fun(img):
        plt.figure(figsize=(6,4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        img = cv2.resize(img,(224,224))
        img = np.reshape(img, [-1, 224, 224,3])
        result = np.argmax(model.predict(img))
        if result == 0:
            print("the image is Recyclable waste")
        elif result == 1:
                print("the image show is Organic waste")

    test_img = cv2.imread(r"C:\Users\Sanjay\OneDrive\Desktop\waste classification\DATASET\TEST\O\O_12577.jpg")
    predict_fun(test_img)

    test_img = cv2.imread(r"C:\Users\Sanjay\OneDrive\Desktop\waste classification\DATASET\TEST\R\R_11110.jpg")
    predict_fun(test_img)

