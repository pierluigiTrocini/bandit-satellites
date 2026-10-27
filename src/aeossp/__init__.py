"""aeossp: modello astratto dello scheduling di satelliti agili di osservazione della Terra (AEOSSP).

Livelli (vedi docs/architettura-simulatore.md):
  geometry   propagazione orbitale, sistemi di riferimento, finestre di accesso (VTW)
  model      parametri dei satelliti, tempo di transizione (manovra + stabilizzazione), budget di risorse
  instance   l'istanza del problema (task candidati, finestre di osservazione, classi di bracci)
  schedule   verifica di un piano rispetto ai vincoli (2), (5), (12), (13) del paper e calcolo del profitto
  solvers    riferimenti: programmazione dinamica, greedy, MILP; oracolo del regret
  env        ambiente a round per gli algoritmi di online learning (bandit combinatorio con feedback ritardato)
  trace      esportazione verso il formato di traccia v2 del viewer

Dipendenze: solo numpy e scipy (nessuna dipendenza da BSK-RL/Basilisk, tranne l'adattatore opzionale `bsk_adapter`).
"""
