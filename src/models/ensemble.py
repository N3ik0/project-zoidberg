"""
Module d'ensemble : fait voter plusieurs modèles pour un diagnostic plus fiable.
Stratégie : soft voting pondéré (moyenne pondérée des probabilités softmax).
"""
import torch
import torch.nn.functional as F
from src.models.registry import get_model


class EnsemblePredictor:
    """
    Charge N modèles et produit un diagnostic par soft voting pondéré.

    En médical, un seul modèle peut "se tromper avec confiance".
    L'ensemble réduit ce risque : le diagnostic n'est validé que
    si la majorité des modèles converge.
    """

    def __init__(self, model_configs, device, confidence_threshold=0.7, weights=None):
        """
        Args:
            model_configs: Liste de dicts {"name": str, "weights_path": str, "num_classes": int}
            device: Device (cuda/cpu)
            confidence_threshold: Seuil en-dessous duquel on marque "à revoir"
            weights: Liste de poids pour chaque modèle (si None, poids égaux)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.models = []
        self.model_names = []

        for config in model_configs:
            model = get_model(config["name"], config["num_classes"])
            model.load_state_dict(
                torch.load(config["weights_path"], map_location=device, weights_only=True)
            )
            model.to(device)
            model.eval()
            self.models.append(model)
            self.model_names.append(config["name"])

        # Poids de chaque modèle (normalisés)
        if weights is not None:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(self.models)
            self.weights = [1.0 / n] * n

    @classmethod
    def from_accuracies(cls, model_configs, accuracies, device, confidence_threshold=0.7):
        """
        Crée un EnsemblePredictor avec des poids basés sur l'accuracy de chaque modèle.

        Args:
            model_configs: Liste de dicts config
            accuracies: Liste des accuracies correspondantes (0-100)
            device: Device
            confidence_threshold: Seuil de confiance
        """
        return cls(model_configs, device, confidence_threshold, weights=accuracies)

    def predict(self, image_tensor):
        """
        Prédit la classe d'une image en faisant voter tous les modèles (pondéré).

        Args:
            image_tensor: Image en tensor (1, C, H, W), déjà transformée

        Returns:
            dict avec :
                - "classe": index de la classe prédite
                - "probas": probabilités moyennes pondérées par classe
                - "confiance": probabilité de la classe choisie
                - "fiable": True si confiance >= seuil
                - "details": liste des résultats individuels par modèle
        """
        image_tensor = image_tensor.to(self.device)

        all_probas = []
        individual_results = []

        with torch.no_grad():
            for i, model in enumerate(self.models):
                output = model(image_tensor)
                probas = F.softmax(output, dim=1)
                all_probas.append(probas * self.weights[i])

                conf, cls = torch.max(probas, dim=1)
                individual_results.append({
                    "name": self.model_names[i],
                    "classe": cls.item(),
                    "confiance": conf.item(),
                    "probas": probas.squeeze().cpu().tolist(),
                    "poids": self.weights[i],
                })

        # Weighted soft voting : somme pondérée (les poids sont normalisés)
        mean_probas = torch.stack(all_probas).sum(dim=0)  # (1, num_classes)
        confiance, classe = torch.max(mean_probas, dim=1)

        return {
            "classe": classe.item(),
            "probas": mean_probas.squeeze().cpu().tolist(),
            "confiance": confiance.item(),
            "fiable": confiance.item() >= self.confidence_threshold,
            "details": individual_results,
        }

    def predict_tta(self, image_pil, tta_transforms):
        """
        Prédit avec Test-Time Augmentation : applique N transforms différentes
        et moyenne les probabilités pour un résultat plus robuste.

        Args:
            image_pil: Image PIL (pas encore transformée)
            tta_transforms: Liste de transforms Compose

        Returns:
            dict identique à predict(), avec les probas moyennées sur les TTA passes
        """
        all_tta_probas = []

        for transform in tta_transforms:
            tensor = transform(image_pil).unsqueeze(0).to(self.device)
            result = self.predict(tensor)
            all_tta_probas.append(result["probas"])

        # Moyenne des probabilités sur toutes les passes TTA
        import numpy as np
        mean_probas = np.mean(all_tta_probas, axis=0)
        classe = int(np.argmax(mean_probas))
        confiance = float(mean_probas[classe])

        # Recalculer les détails pour la dernière passe (informatif)
        last_result = self.predict(tta_transforms[0](image_pil).unsqueeze(0).to(self.device))

        return {
            "classe": classe,
            "probas": mean_probas.tolist(),
            "confiance": confiance,
            "fiable": confiance >= self.confidence_threshold,
            "details": last_result["details"],
            "tta_passes": len(tta_transforms),
        }

    def evaluate(self, loader):
        """
        Évalue l'ensemble sur un DataLoader complet.

        Returns:
            all_preds: liste des prédictions
            all_labels: liste des vrais labels
        """
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                batch_probas = []

                for i, model in enumerate(self.models):
                    output = model(images)
                    probas = F.softmax(output, dim=1)
                    batch_probas.append(probas * self.weights[i])

                # Weighted soft voting par batch
                mean_probas = torch.stack(batch_probas).sum(dim=0)
                _, preds = torch.max(mean_probas, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return all_preds, all_labels
