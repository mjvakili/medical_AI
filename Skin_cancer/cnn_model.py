from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers, models
from keras import Model



def vgg_model(IMG_WIDTH, IMG_HEIGHT, retrain = False):
    """
    VGG model + 1 fully connected layer + 1 softmax layer for land use classification
    """
    baseModel = VGG16(weights="imagenet", include_top=False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    #If retrain is set to False, freeze the VGG16 layers:
    if retrain == False:
        for layer in baseModel.layers:
	          layer.trainable = False
    x = baseModel.output
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(7, activation="softmax")(x)

    # Configure the model
    model = Model(baseModel.input, x)
    
    return model
