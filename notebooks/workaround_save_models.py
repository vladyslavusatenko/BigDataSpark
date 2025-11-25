# Workaround for Windows Hadoop issues when saving ML Pipeline models
# Use this code instead of cell 34 in the notebook

# Cell 34 - FIXED VERSION

# Save individual feature sets for specific use cases

# 1. User-level features (for segmentation)
print("Saving user features...")
rfm_features.write \
    .format("delta") \
    .mode("overwrite") \
    .save("../data/processed/delta/user_features")

print("✓ User features saved")

# 2. Product-level features (for recommendations)
print("Saving product features...")
product_features.write \
    .format("delta") \
    .mode("overwrite") \
    .save("../data/processed/delta/product_features")

print("✓ Product features saved")

# 3. Encoding pipeline model - WORKAROUND
# Instead of saving the pipeline, save the stages separately
print("Saving encoding pipeline (workaround)...")

# Save as pickle (Python native, no Hadoop needed)
import pickle
import os

models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

# Save the encoding stages info (for recreation)
encoding_info = {
    'categorical_cols': categorical_cols,
    'stages': {
        'indexers': [(stage.getInputCol(), stage.getOutputCol()) for stage in encoding_model.stages if hasattr(stage, 'getInputCol') and '_index' in stage.getOutputCol()],
        'encoders': [(stage.getInputCol(), stage.getOutputCol()) for stage in encoding_model.stages if hasattr(stage, 'getInputCol') and '_vec' in stage.getOutputCol()]
    }
}

with open(f"{models_dir}/encoding_pipeline_config.pkl", 'wb') as f:
    pickle.dump(encoding_info, f)

print("✓ Encoding pipeline config saved (as pickle)")
print(f"  Location: {models_dir}/encoding_pipeline_config.pkl")

# 4. Scaler model - WORKAROUND
print("Saving scaler model...")

# Save scaler parameters
scaler_info = {
    'inputCol': scaler_model.getInputCol(),
    'outputCol': scaler_model.getOutputCol(),
    'mean': scaler_model.mean.toArray().tolist() if hasattr(scaler_model, 'mean') else None,
    'std': scaler_model.std.toArray().tolist()
}

with open(f"{models_dir}/scaler_model_config.pkl", 'wb') as f:
    pickle.dump(scaler_info, f)

print("✓ Scaler config saved (as pickle)")
print(f"  Location: {models_dir}/scaler_model_config.pkl")

print("\n" + "="*60)
print("WORKAROUND APPLIED")
print("="*60)
print("Models saved using pickle instead of Spark MLWriter")
print("This avoids Hadoop native library issues on Windows")
print("\nTo recreate the pipelines:")
print("  1. Load config files with pickle")
print("  2. Rebuild Pipeline with same stages")
print("  3. Or simply re-run the encoding cells")
print("="*60)
