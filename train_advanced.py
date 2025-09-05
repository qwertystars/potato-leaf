import os
import argparse
import numpy as np
from advanced_model import AdvancedLeafDiseaseModel, create_data_generators
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import joblib

def train_model(model_type="densenet", epochs=50, batch_size=16):
    """Train advanced model with selected architecture"""

    print(f"🚀 Training {model_type.upper()} model for leaf disease classification...")

    # Check if dataset exists
    dataset_dir = "datasets"
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found!")
        return None

    # Create model
    model_trainer = AdvancedLeafDiseaseModel(model_type=model_type)
    model = model_trainer.create_model()
    model_trainer.compile_model()

    print(f"✅ Created {model_type} model with {model.count_params():,} parameters")

    # Create data generators
    train_gen, val_gen = create_data_generators(dataset_dir, batch_size=batch_size)

    print(f"📊 Training data: {train_gen.samples} images")
    print(f"📊 Validation data: {val_gen.samples} images")
    print(f"📊 Classes: {list(train_gen.class_indices.keys())}")

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/best_{model_type}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print(f"🔥 Starting training for {epochs} epochs...")

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_type}_leaf_disease_model.h5"
    model.save(model_path)

    # Save class indices for prediction
    class_info = {
        'class_indices': train_gen.class_indices,
        'model_type': model_type,
        'input_shape': (224, 224, 3)
    }
    joblib.dump(class_info, f"models/{model_type}_class_info.pkl")

    print(f"✅ Model saved as {model_path}")
    print(f"✅ Class info saved as models/{model_type}_class_info.pkl")

    # Print training summary
    final_acc = max(history.history['val_accuracy'])
    print(f"\n🎯 Best validation accuracy: {final_acc:.4f}")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train advanced leaf disease model')
    parser.add_argument('--model', type=str, default='densenet',
                       choices=['densenet', 'vgg16', 'resnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')

    args = parser.parse_args()

    # Train model
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    if model:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model files saved in models/ directory")
        print(f"🎯 Use --model {args.model} to train this architecture again")
