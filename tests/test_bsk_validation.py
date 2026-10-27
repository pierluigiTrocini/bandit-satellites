"""Test d'integrazione con BSK-RL (richiedono bsk e bsk-rl; si saltano se mancano). Durano alcune decine di secondi.

Verificano che il livello astratto sia coerente con BSK-RL dove deve esserlo (geometria) e fissano per iscritto dove
non lo e' (tempo fra immagini: BSK-RL e' molto piu' lento del modello di transizione neutro)."""
import unittest

import _path  # noqa: F401
import numpy as np

try:
    import validate_aeossp as V
    HAVE_BSK = True
except Exception:                                  # pragma: no cover
    HAVE_BSK = False


@unittest.skipUnless(HAVE_BSK, "BSK-RL non disponibile")
class TestAgainstBsk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = V.compare_geometry(hours=6.0, n_targets=150)

    def test_numerical_j2_matches_bsk_trajectory(self):
        nj = self.geo["providers"]["NumericalJ2"]
        self.assertLess(nj["pos_err_km_max"], 15.0)                  # misurato: ~10 km, dovuto a orientazione terrestre e armoniche superiori
        self.assertEqual(nj["unmatched"], 0)
        self.assertLess(nj["d_open_mean_abs_s"], 3.0)
        self.assertLessEqual(abs(nj["windows"] - nj["windows_bsk"]), max(2, 0.02 * nj["windows_bsk"]))

    def test_analytic_provider_is_much_worse_and_documented_as_such(self):
        # gli elementi osculanti letti come medi cambiano il periodo dello ~0.1%: la deriva e' enorme in confronto
        kj, nj = self.geo["providers"]["KeplerJ2"], self.geo["providers"]["NumericalJ2"]
        self.assertGreater(kj["pos_err_km_max"], 10 * nj["pos_err_km_max"])

    def test_bsk_is_slower_than_the_neutral_transition_model(self):
        rows, n_img = V.measure_slew(hours=4.0, n_targets=400)
        self.assertGreater(len(rows), 15)
        self.assertGreater(np.median(rows[:, 1]), 30.0)              # fra due immagini consecutive passano decine di secondi
        from aeossp.model import SlewModel
        neutral = SlewModel().transition(np.radians(rows[:, 0]))
        self.assertGreater(float(np.mean(rows[:, 1] - neutral)), 10.0)   # il modello neutro (5 s + manovra) e' ottimista
        fit = V.fit_slew(rows)
        self.assertTrue(10.0 < fit["settle_s"] < 120.0, fit)
        self.assertLess(fit["rms_s"], 40.0)
        self.assertLess(abs(fit["bias_s"]), 5.0)                     # il modello calibrato non ha bias sistematico


if __name__ == "__main__":
    unittest.main()
