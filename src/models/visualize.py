import torch
import warnings
from torchinfo import summary

# Importe ton modèle depuis le fichier où tu l'as défini (ex: model.py)
# Si tu l'as mis dans predict.py, ajuste l'import
from model import GuitarCRNN 

def visualize_text_summary(model: torch.nn.Module, input_shape: tuple):
    """
    Méthode 1 : Affiche un tableau détaillé dans le terminal.
    Idéal pour vérifier le nombre de paramètres et les dimensions à chaque couche.
    """
    print("\n" + "="*50)
    print("🧠 ANALYSE DE L'ARCHITECTURE DU MODÈLE (TORCHINFO)")
    print("="*50 + "\n")
    
    # torchinfo fait passer un faux tenseur dans le modèle pour analyser les flux
    summary(
        model, 
        input_size=input_shape,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        col_width=18,
        row_settings=("var_names",)
    )

def export_to_onnx(model: torch.nn.Module, input_shape: tuple, filepath: str = "guitar_crnn.onnx"):
    """
    Méthode 2 : Exporte le modèle dans le format standard ONNX.
    Ce fichier pourra être ouvert dans un visualiseur graphique.
    """
    print("\n" + "="*50)
    print(f"📦 EXPORT DU MODÈLE VERS ONNX : {filepath}")
    print("="*50 + "\n")
    
    # 1. Créer un faux tenseur d'entrée (Dummy input)
    # [Batch, Channels, Time, Frequencies]
    dummy_input = torch.randn(*input_shape)
    
    # Ignorer les warnings liés aux opérations dynamiques du RNN lors de l'export
    warnings.filterwarnings("ignore")
    
    # 2. Exporter le graphe de calcul
    torch.onnx.export(
        model,                      # Le modèle en mémoire
        dummy_input,                # L'entrée factice
        filepath,                   # Le nom du fichier de sortie
        export_params=True,         # Stocker les poids (même s'ils ne sont pas encore entraînés)
        opset_version=12,           # Version de compatibilité ONNX
        do_constant_folding=True,   # Optimisation du graphe
        input_names=['CQT_Input'],  # Nommer l'entrée pour que le schéma soit lisible
        output_names=['Onset_Probabilities', 'Frame_Probabilities'], # Nommer les sorties
        # Rendre l'axe du temps dynamique (car une musique peut faire 10s ou 3 minutes)
        dynamic_axes={
            'CQT_Input': {2: 'time_steps'},
            'Onset_Probabilities': {1: 'time_steps'},
            'Frame_Probabilities': {1: 'time_steps'}
        }
    )
    print("✅ Export réussi ! Vous pouvez maintenant visualiser le fichier.")

if __name__ == "__main__":
    # Instanciation du modèle (avec nos 216 bins de CQT et 88 notes possibles)
    model = GuitarCRNN(n_bins=216, n_classes=88)
    
    # Définition de la forme d'entrée (Input Shape)
    # Batch Size = 1
    # Channels = 1 (Mono)
    # Time Steps = 100 frames (correspond à environ 1 seconde d'audio selon notre hop_length)
    # Frequencies = 216 bins (notre résolution CQT)
    dummy_shape = (1, 1, 100, 216)
    
    # 1. Afficher le résumé dans la console
    visualize_text_summary(model, dummy_shape)
    
    # 2. Générer le fichier pour le diagramme visuel
    export_to_onnx(model, dummy_shape, filepath="src/models/guitar_crnn.onnx")