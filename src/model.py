import tensorflow as tf 
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

def build_model(num_class=3):
    """ 
    Construction d'un modèle basé sur ResNet50v2
    """

    # Socle du modèle
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    # Freezing (gèle du socle pour ne pas ecraser)
    base_model.trainable = False

    # Greffe d'une nouvelle tête
    inputs = Input(shape=(224,224,3))

    # On fait passer l'image dans le socle
    x = base_model(inputs, training=False)

    # On resume l'info (pooling)
    x = GlobalAveragePooling2D()(x)

    #Couche intermediaire
    x = Dense(128, activation="relu")(x)

    # Couche de sortie
    # Prise de décision finale (label 1,2,3) softmax pour proba proche dde 100%
    outputs = Dense(num_class, activation="softmax")(x)

    # assemblage du modèle
    model = Model(inputs, outputs)

    return model