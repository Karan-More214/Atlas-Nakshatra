# Findings draft (auto-generated - edit before publishing)


- Target: downloads_30d >= 10. Base rate on the test set: 61.8%.
- Final model (LightGBM, calibrated) on the newest 1,818 models: PR-AUC 0.87 (vs 0.62 for guessing), ROC-AUC 0.80.
- The top 10% of models by predicted probability are adopted 1.6x more often than average (98% vs 62%).
- Strongest drivers: GGUF (local-run) weights, Links in card, Builder type, Model size (parameters), No. of files.
- Adoption rate by number of documentation sections (0-5): 0: 60%, 1: 76%, 2: 87%, 3: 56%, 4: 72%, 5: 83%.
