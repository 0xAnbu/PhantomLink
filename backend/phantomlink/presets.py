"""Training presets. Each one changes real hyper-parameters and states what we expect."""

GENERATOR_PRESETS = {
    "fast": {
        "label": "⚡ Fast",
        "expectation": "Modelo diminuto (1 capa, 32 dims). Entrena en segundos; los dominios "
                       "tendrán estructura básica pero poca coherencia.",
        "config": {"n_layer": 1, "n_head": 2, "n_embd": 32, "block_size": 64, "dropout": 0.0,
                   "learning_rate": 3e-3, "batch_size": 32, "iterations": 400, "grad_clip": 1.0,
                   "warmup_frac": 0.05, "eval_interval": 50, "seed": 42},
    },
    "balanced": {
        "label": "⚖️ Balanced",
        "expectation": "Configuración de referencia (2 capas, 4 cabezas, 64 dims, ~107k "
                       "parámetros). La validation loss debería bajar de forma estable.",
        "config": {"n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 64, "dropout": 0.1,
                   "learning_rate": 1e-3, "batch_size": 64, "iterations": 2000, "grad_clip": 1.0,
                   "warmup_frac": 0.05, "eval_interval": 100, "seed": 42},
    },
    "break_it": {
        "label": "💥 Break it",
        "expectation": "Learning rate 1000× mayor (1.0), sin warmup ni gradient clipping. En "
                       "nuestras pruebas la loss explota (≈3,7 → >100) con normas de gradiente "
                       "de decenas, y después se queda atascada en ≈3,3 (el modelo sano llega a "
                       "≈2,7): genera sopas de letras.",
        "config": {"n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 64, "dropout": 0.0,
                   "learning_rate": 1.0, "batch_size": 32, "iterations": 600, "grad_clip": 0.0,
                   "warmup_frac": 0.0, "eval_interval": 25, "seed": 42},
    },
}

CLASSIFIER_PRESETS = {
    "fast": {
        "label": "⚡ Fast",
        "expectation": "20 árboles poco profundos con 10k dominios: entrena en unos "
                       "segundos. Medido: F1 test ≈0,78 (casi igual que Balanced).",
        "config": {"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 5,
                   "class_weight": "balanced", "threshold": 0.5, "train_size": 10000, "seed": 42},
    },
    "balanced": {
        "label": "⚖️ Balanced",
        "expectation": "100 árboles, profundidad 12, 30k dominios: la configuración de "
                       "referencia. Medido: F1 test ≈0,785, AUC ≈0,86.",
        "config": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2,
                   "class_weight": "balanced", "threshold": 0.5, "train_size": 30000, "seed": 42},
    },
    "break_it": {
        "label": "💥 Break it",
        "expectation": "Árboles sin límite de profundidad, hojas de 1 muestra, sin pesos de "
                       "clase y solo 2k dominios: sobreajuste. Medido: accuracy train ≈0,94 "
                       "frente a ≈0,83 en validación; F1 test cae a ≈0,75.",
        "config": {"n_estimators": 30, "max_depth": None, "min_samples_leaf": 1,
                   "class_weight": None, "threshold": 0.5, "train_size": 2000, "seed": 42},
    },
}
