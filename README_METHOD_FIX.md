
# SACU V2 Methodology Fix

The old evaluation used y_true to compute agent influence weights. Because those
weights were used to stitch the final prediction, the hidden target affected
the prediction itself.

SACU V2 removes y_true completely from the inference function.

Influence is now based on:
- 50% observed-sensor consistency,
- 35% local wave-equation residual,
- 15% normalized gate entropy.

The three score families are normalized across agents before combination.

y_true remains valid only for:
- supervised training loss,
- post-prediction evaluation metrics.

First run:
    python scripts\00_prepare_minimal_source.py

Then validate:
    python scripts\01_validate_no_target_leakage.py --model "PATH_TO_MODEL.keras"

Important:
An old model may be used only to test the corrected inference path.
Do not use revised paper results until SACU V2 is retrained using TrainerV2.
