# Findings draft (auto-generated - edit before publishing)


- Target: downloads_30d >= 50. Base rate on the test set: 15.8%.
- Final model (LightGBM, calibrated) on the newest 1,116 models: PR-AUC 0.43 (vs 0.16 for guessing), ROC-AUC 0.74.
- The top 10% of models by predicted probability are adopted 2.9x more often than average (46% vs 16%).
- Strongest drivers: Builder type, Model card wording (topics), No. of language tags, Model size (parameters), No. of files.
- Adoption rate by number of documentation sections (0-5): 0: 13%, 1: 13%, 2: 29%, 3: 14%, 4: 20%, 5: 27%.
