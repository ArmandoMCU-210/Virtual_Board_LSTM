"""
Módulo Autocompleter basado en server.py
Proporciona la clase Autocompleter para sugerir la siguiente palabra usando un modelo LSTM entrenado.
"""
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Autocompleter:
    def __init__(self,
                 model_path: str = 'model/autocomplete_es.h5',
                 tokenizer_path: str = 'model/tokenizer.pkl',
                 seq_length: int = 3):
        """
        Inicializa el Autocompleter.
        :param model_path: Ruta al archivo .h5 del modelo entrenado.
        :param tokenizer_path: Ruta al archivo .pkl del tokenizer.
        :param seq_length: Número de palabras de contexto a considerar.
        """
        self.seq_length = seq_length
        # Cargar modelo LSTM
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise IOError(f"No se pudo cargar el modelo en '{model_path}': {e}")
        # Cargar tokenizer
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        except Exception as e:
            raise IOError(f"No se pudo cargar el tokenizer en '{tokenizer_path}': {e}")
        # Mapeo inverso índice->palabra
        self.index_word = {index: word for word, index in self.tokenizer.word_index.items()}

    def suggest(self, text: str) -> str:
        """
        Sugiere la siguiente palabra basada en el contexto dado.
        :param text: Texto ingresado hasta el momento.
        :return: Palabra sugerida.
        """
        # Normalizar y tokenizar texto completo
        words = text.lower().split()
        # Convertir a secuencia de índices
        seq = self.tokenizer.texts_to_sequences([' '.join(words)])[0]
        # Asegurar longitud fija con padding pre
        seq_padded = pad_sequences([seq], maxlen=self.seq_length, padding='pre', truncating='pre')
        # Predicción
        preds = self.model.predict(seq_padded, verbose=0)[0]
        next_idx = int(np.argmax(preds))
        # Devolver palabra o cadena vacía si no existe
        return self.index_word.get(next_idx, '')
