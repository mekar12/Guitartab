import os
import pandas as pd
from typing import List, Tuple
# On importe le modèle d'inférence pré-entraîné de Spotify
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

class PitchTracker:
    """
    Classe en charge de l'inférence Deep Learning pour la transcription audio.
    Utilise le modèle Basic Pitch pour extraire les événements musicaux (polyphoniques).
    """

    def __init__(self):
        """
        Initialise le traqueur. Le modèle pré-entraîné est chargé automatiquement
        lors du premier appel à predict().
        """
        print("🧠 Initialisation du module IA de Pitch Tracking (Basic Pitch)...")
        self.model_path = ICASSP_2022_MODEL_PATH

    def extract_notes(self, audio_path: str) -> pd.DataFrame:
        """
        Analyse un fichier audio et détecte toutes les notes jouées.

        Args:
            audio_path (str): Chemin vers le fichier audio (prétraité idéalement).

        Returns:
            pd.DataFrame: Un tableau contenant [start_time, end_time, pitch, velocity]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier introuvable pour l'inférence : {audio_path}")

        print(f"🎵 Analyse neuronale en cours sur : {os.path.basename(audio_path)}")
        
        # Le modèle retourne 3 éléments :
        # 1. model_output : les probabilités brutes (frames, onsets, contours)
        # 2. midi_data : un objet PrettyMIDI structuré
        # 3. note_events : une liste de tuples (start_t, end_t, pitch_midi, amplitude, pitch_bends)
        _, _, note_events = predict(audio_path)

        if not note_events:
            print("⚠️ Aucune note détectée par l'IA.")
            return pd.DataFrame()

        # Nettoyage et structuration des données en DataFrame Pandas pour faciliter 
        # le traitement mathématique à l'étape suivante (le mapping du manche)
        notes_data = []
        for note in note_events:
            start_time, end_time, pitch_midi, velocity, _ = note
            notes_data.append({
                "start_time": round(start_time, 3), # Arrondi à la milliseconde
                "end_time": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "pitch_midi": int(pitch_midi),      # Ex: 64 = Mi (E4)
                "velocity": round(velocity, 2)      # Confiance de l'IA (0.0 à 1.0)
            })

        # On trie chronologiquement pour respecter l'ordre d'exécution du guitariste
        df_notes = pd.DataFrame(notes_data).sort_values(by="start_time").reset_index(drop=True)
        
        print(f"✅ Inférence terminée : {len(df_notes)} notes détectées.")
        return df_notes