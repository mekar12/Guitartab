import librosa
import numpy as np
import soundfile as sf
import os
import warnings

# Ignorer les avertissements mineurs de librosa liés aux formats MP3
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

class AudioPreprocessor:
    """
    Classe en charge du prétraitement des signaux audio pour la transcription MIR.
    Standardise les taux d'échantillonnage, les canaux et l'amplitude.
    """
    
    def __init__(self, target_sr: int = 22050, mono: bool = True, top_db: int = 30):
        """
        Initialise le préprocesseur avec les paramètres standards.
        
        Args:
            target_sr (int): Fréquence d'échantillonnage cible (22050 Hz recommandé).
            mono (bool): Si True, convertit le signal en mono.
            top_db (int): Seuil en décibels pour la suppression des silences.
        """
        self.target_sr = target_sr
        self.mono = mono
        self.top_db = top_db

    def process(self, file_path: str) -> tuple[np.ndarray, int]:
        """
        Charge et prétraite un fichier audio.
        
        Args:
            file_path (str): Chemin vers le fichier audio (WAV, MP3, etc.)
            
        Returns:
            tuple: (signal_audio_traité (np.ndarray), frequence_echantillonnage (int))
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier audio est introuvable : {file_path}")

        print(f"🎵 Prétraitement en cours : {os.path.basename(file_path)}")

        # 1 & 2. Chargement, conversion Mono et Rééchantillonnage
        # Librosa gère nativement le resampling et le mono lors du load.
        audio, sr = librosa.load(file_path, sr=self.target_sr, mono=self.mono)

        # 3. Suppression des silences (Trimming)
        # Sépare le signal actif du silence absolu en début et fin de piste
        audio_trimmed, index = librosa.effects.trim(audio, top_db=self.top_db)
        
        # 4. Normalisation Peak
        # Assure que l'amplitude maximale absolue du signal est exactement 1.0
        if len(audio_trimmed) > 0:
            max_val = np.max(np.abs(audio_trimmed))
            if max_val > 0:
                audio_normalized = audio_trimmed / max_val
            else:
                audio_normalized = audio_trimmed
        else:
            audio_normalized = audio_trimmed

        print(f"✅ Terminé : {len(audio_normalized)} échantillons à {self.target_sr} Hz.")
        
        return audio_normalized, self.target_sr

    def save(self, audio: np.ndarray, sr: int, output_path: str):
        """
        Sauvegarde le signal prétraité sur le disque (utile pour débugger).
        """
        sf.write(output_path, audio, sr)
        print(f"💾 Fichier sauvegardé : {output_path}")