"""Controlli del protocollo, senza inventare risultati del dataset."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from esperimento import split_horses, fit_head, metrics


class ProtocolloTest(unittest.TestCase):
    # Evita che copie identiche di una mesh compaiano sia nel training sia nel test.
    # Scopo: Controlla che file identici non finiscano sia nel supporto sia nel test.
    def test_split_disgiunto_e_deduplicato(self):
        records = [dict(id=str(i), mesh_sha256=str(i // 2)) for i in range(60)]
        train, test = split_horses(records, 10)
        self.assertEqual(len(train), 20)
        self.assertEqual(len(test), 10)
        self.assertFalse({r['mesh_sha256'] for r in train} & {r['mesh_sha256'] for r in test})
        self.assertEqual((train, test), split_horses(records, 10))

    # Verifica il caso one-shot e controlla che la testa originale resti invariata.
    # Scopo: Verifica one-shot, ripetibilità e mancata modifica della testa di partenza.
    def test_un_esempio_non_modifica_originale(self):
        torch.manual_seed(1)
        original = torch.nn.Linear(256, 40)
        before = original.weight.detach().clone()
        support = torch.randn(1, 256)
        replay = torch.randn(40, 256)
        a = fit_head(original, support, replay, torch.arange(40), 11, 3, .001)
        b = fit_head(original, support, replay, torch.arange(40), 11, 3, .001)
        torch.testing.assert_close(original.weight, before)
        torch.testing.assert_close(a.weight, b.weight)
        self.assertEqual(a.out_features, 41)
        self.assertFalse(torch.equal(a.weight[:40], before))

    # Una risposta costante puo avere recall alta ma deve fallire sulle altre metriche.
    # Scopo: Dimostra che la sola recall non basta per valutare la nuova classe.
    def test_rispondere_sempre_horse_non_basta(self):
        labels = np.arange(41)
        result = metrics(np.full(41, 40), labels)
        self.assertEqual(result['horse_recall'], 1)
        self.assertEqual(result['old_accuracy'], 0)
        self.assertAlmostEqual(result['accuracy'], 1/41)
        self.assertAlmostEqual(result['horse_precision'], 1/41)


if __name__ == '__main__':
    unittest.main()
