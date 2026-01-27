import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_data, test_data, img_size=(224,224), batch_size=32):
    """ 
    Transforme les listes (chemin, label) en générateur keras
    """
    # conversion des dataframe
    train_df = pd.DataFrame(train_data, columns=['filepath', 'label'])
    train_df['filepath'] = train_df['filepath'].astype(str)
    train_df['label'] = train_df['label'].astype(int)

    test_df = pd.DataFrame(test_data, columns=['filepath', 'label'])
    test_df['filepath'] = test_df['filepath'].astype(str)
    test_df['label'] = test_df['label'].astype(int)

    # Data normalisation (rescale en divisant par 255)
    datagen = ImageDataGenerator(rescale=1./255)
    print(f"Préparateur des generateurs keras (batchsize : {batch_size})")

    # Création du flux d'entrainement
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=True
    )

    # Création du flux de test
    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=True
    )

    return train_gen, test_gen