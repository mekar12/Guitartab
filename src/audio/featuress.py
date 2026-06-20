import librosa
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    Classe en charge de l'extraction des caractéristiques temps-fréquence
    (Spectrogrammes) à partir d'un signal audio prétraité.
    Optimisé pour la détection de notes de guitare (CQT).
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512, bins_per_octave: int = 36):
        """
        Initialise l'extracteur.
        
        Args:
            sr (int): Fréquence d'échantillonnage (doit matcher le preprocessor).
            hop_length (int): Nombre d'échantillons entre chaque trame temporelle.
                512 à 22050Hz donne une résolution d'environ 23ms (excellent pour le rythme).
            bins_per_octave (int): Résolution fréquentielle. 36 = 3 "cases" par demi-ton.
                Avoir une résolution plus fine que le demi-ton aide l'IA à gérer 
                les bends, les vibratos ou les guitares légèrement désaccordées.
        """
        self.sr = sr
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        
        # Plage de la guitare : Mi grave (E2) ~82.41 Hz
        self.fmin = librosa.note_to_hz('E2') 
        # 6 octaves pour couvrir jusqu'aux harmoniques aiguës
        self.n_bins = 6 * self.bins_per_octave 

    def compute_cqt(self, y: np.ndarray) -> np.ndarray:
        """
        Calcule la Transformée à Q Constant (CQT) du signal.
        
        Args:
            y (np.ndarray): Signal audio temporel (1D).
            
        Returns:
            np.ndarray: Spectrogramme CQT en échelle décibel (2D : [Fréquences, Temps]).
        """
        print(f"🎵 Calcul de la CQT (Résolution: {self.bins_per_octave} bins/octave)...")
        
        # Calcul de la CQT complexe
        C = librosa.cqt(
            y, 
            sr=self.sr, 
            hop_length=self.hop_length, 
            fmin=self.fmin, 
            n_bins=self.n_bins, 
            bins_per_octave=self.bins_per_octave
        )
        
        # Conversion de l'amplitude (module de la matrice complexe) en Décibels
        # log_CQT est ce que le réseau de neurones va ingérer.
        log_CQT = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        print(f"✅ CQT calculée. Shape (Fréquences x Trames temporelles) : {log_CQT.shape}")
        return log_CQT

    def plot_cqt(self, log_CQT: np.ndarray, title: str = "Constant-Q Transform (Guitar Range)"):
        """
        Affiche le spectrogramme CQT (utile pour le débogage visuel).
        """
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            log_CQT, 
            sr=self.sr, 
            hop_length=self.hop_length, 
            x_axis='time', 
            y_axis='cqt_note', 
            fmin=self.fmin, 
            cmap='magma' # Palette de couleurs contrastée, idéale pour les onsets
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()