import torch
import torch.nn as nn

class GuitarCRNN(nn.Module):
    """
    Réseau de neurones CRNN pour la transcription polyphonique de guitare.
    Approche 'Onsets and Frames' simplifiée.
    """
    
    def __init__(self, n_bins: int = 216, n_classes: int = 88):
        """
        Args:
            n_bins (int): Nombre de fréquences en entrée (Notre CQT a 216 bins).
            n_classes (int): Nombre de notes possibles (88 pour couvrir tout un piano/guitare).
        """
        super(GuitarCRNN, self).__init__()
        
        # --- 1. Bloc CNN (Extraction des caractéristiques spatiales/fréquentielles) ---
        # On utilise des convolutions 2D pour repérer les motifs harmoniques
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), # Réduit la dimension fréquentielle, garde le temps
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Dropout(0.25)
        )
        
        # Calcul de la taille de sortie du CNN pour la passer au RNN
        # 216 / 2 / 2 = 54 bins restants * 64 channels = 3456
        cnn_out_features = (n_bins // 4) * 64
        
        # --- 2. Bloc RNN (Modélisation temporelle) ---
        # Le Bi-GRU lit la musique en avant et en arrière pour un meilleur contexte
        self.rnn = nn.GRU(
            input_size=cnn_out_features, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- 3. Têtes de Prédiction (Multi-Task Learning) ---
        # Tête 1 : Détection des attaques (Onsets)
        self.onset_head = nn.Sequential(
            nn.Linear(256 * 2, 128), # *2 car Bi-directionnel
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Sigmoid() # Sortie entre 0.0 et 1.0 (Probabilité)
        )
        
        # Tête 2 : Détection des notes maintenues (Frames)
        self.frame_head = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Passe avant (Forward pass).
        Args:
            x (torch.Tensor): Entrée de forme [Batch, 1, Temps, n_bins]
        Returns:
            Tuple: (Probabilités Onsets, Probabilités Frames)
        """
        # 1. Passage dans le CNN
        x = self.cnn(x) 
        
        # 2. Reshape pour le RNN : [Batch, Channels, Temps, Freqs] -> [Batch, Temps, Features]
        batch_size, channels, time_steps, freqs = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, time_steps, channels * freqs)
        
        # 3. Passage dans le RNN
        rnn_out, _ = self.rnn(x)
        
        # 4. Prédictions séparées
        onsets = self.onset_head(rnn_out)
        frames = self.frame_head(rnn_out)
        
        return onsets, frames