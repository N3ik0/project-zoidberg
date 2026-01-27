# Loader zoidberg
from pathlib import Path

class Loader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        #Images
        self.train_normal_paths = []
        self.train_pneumonia_paths = []
        self.test_normal_paths = []
        self.test_pneumonia_paths = []

        if not self.data_dir.exists():
            print(f"Erreur : le dossier {self.data_dir} n'éxiste pas")
            

    def load_paths(self):
        """ Récupère la liste de tous les fichiers sans les ouvrirs """
        train_normal_dir = self.data_dir / "raw" / "train" / "NORMAL"
        train_pneumonia_dir = self.data_dir / "raw" / "train" / "PNEUMONIA"
        test_normal_dir = self.data_dir / "raw" / "test" / "NORMAL"
        test_pneumonia_dir = self.data_dir / "raw" / "test" / "PNEUMONIA"

        self.train_normal_paths = list(train_normal_dir.rglob("*.jpeg"))
        self.train_pneumonia_paths = list(train_pneumonia_dir.rglob("*.jpeg"))
        self.test_normal_paths = list(test_normal_dir.rglob("*.jpeg"))
        self.test_pneumonia_paths = list(test_pneumonia_dir.rglob("*.jpeg"))

        print(f"J'ai trouvé {len(self.train_normal_paths)} images d'entrainement saines")
        print(f"J'ai trouvé {len(self.train_pneumonia_paths)} images d'entrainement pneumonie")
        print(f"J'ai trouvé {len(self.test_normal_paths)} images de tests normales")
        print(f"J'ai trouvé {len(self.test_pneumonia_paths)} images de tests malades")

        return self.train_normal_paths, self.train_pneumonia_paths, self.test_normal_paths, self.test_pneumonia_paths

    def create_dataset(self):
        """ 
        Création des labels pour le dataset 
        0 = Sain
        1 = Pneumonie bacterienne
        2 = Pneumonie virale
        """
        self.load_paths()

        train_normal_data = [(path, 0) for path in self.train_normal_paths]
        test_normal_data = [(path, 0) for path in self.test_normal_paths]

        train_pneumonia_data = []
        test_pneumonia_data = []

        for path in self.train_pneumonia_paths:
            filename = path.name.lower()

            if "bacteria" in filename:
                label = 1
            elif "virus" in filename:
                label = 2
            else:
                continue

            train_pneumonia_data.append((path, label))

        for path in self.test_pneumonia_paths:
            filename = path.name.lower()

            if "bacteria" in filename:
                label = 1
            elif "virus" in filename:
                label = 2
            else: 
                continue

            test_pneumonia_data.append((path, label))

        full_train_dataset = train_normal_data + train_pneumonia_data
        full_test_dataset = test_normal_data + test_pneumonia_data

        import random
        random.shuffle(full_train_dataset)
        random.shuffle(full_test_dataset)

        print(f"Dataset train : {len(full_train_dataset)} images")
        print(f"Dataset test : {len(full_test_dataset)}")

        return full_train_dataset, full_test_dataset
