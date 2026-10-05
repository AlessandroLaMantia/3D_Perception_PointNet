"""Verifiche locali: pesi reali, preprocessing e coerenza delle viste interne."""
import unittest
import sys
from pathlib import Path

# Rende importabili i moduli del progetto anche eseguendo lo script direttamente.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from inferenza import analizza, carica_modello, leggi_campione, prepara_punti


class TestInferenza(unittest.TestCase):
    # Scopo: Carica una volta pesi reali e campione test condivisi dalle verifiche.
    @classmethod
    def setUpClass(cls):
        cls.model = carica_modello()
        cls.points, cls.label = leggi_campione('test', 0)

    # Controlla forma, centratura, raggio e ripetibilita del campionamento.
    # Scopo: Verifica forma, centro, raggio, ripetibilità e gestione degli input non validi.
    def test_preprocessing(self):
        points = prepara_punti(self.points)
        self.assertEqual(points.shape, (1024, 3))
        np.testing.assert_allclose(points.mean(axis=0), 0, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(points, axis=1).max()), 1, places=6)
        np.testing.assert_array_equal(points, prepara_punti(self.points))
        with self.assertRaises(ValueError):
            prepara_punti(np.array([[np.nan, 0, 0]]))
        with self.assertRaises(IndexError):
            leggi_campione('test', -1)

    # Confronta le viste con il forward reale e verifica invarianza alla permutazione dei punti.
    # Scopo: Confronta le viste col forward reale e controlla l’invarianza agli stessi punti riordinati.
    def test_inspection_matches_real_forward(self):
        result = analizza(self.model, self.points)
        captured = {}

        # Scopo: Cattura l’ingresso reale di fc1 per confrontarlo col descrittore mostrato.
        def before_classifier(module, args):
            captured['descriptor'] = args[0].detach().numpy().copy()

        handle = self.model.fc1.register_forward_pre_hook(before_classifier)
        try:
            with torch.inference_mode():
                tensor = torch.from_numpy(result['input'].T.copy()).unsqueeze(0)
                logits, transform, _ = self.model(tensor)
        finally:
            handle.remove()
        np.testing.assert_allclose(result['probabilities'], logits.softmax(dim=1)[0].numpy())
        np.testing.assert_allclose(result['global_features'], captured['descriptor'][0])
        np.testing.assert_allclose(result['aligned'], (transform[0].numpy() @ result['input'].T).T)
        self.assertAlmostEqual(float(result['probabilities'].sum()), 1, places=6)
        self.assertEqual(int(result['max_counts'].sum()), int((result['global_features'] > 0).sum()))
        # PointNet deve essere invariante alla permutazione dell'ordine dei punti.
        with torch.inference_mode():
            permuted, _, _ = self.model(tensor.flip(dims=[2]))
        torch.testing.assert_close(logits, permuted, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
