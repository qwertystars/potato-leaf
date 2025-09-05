import os
import numpy as np
from feature_extractor import extract_features
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

X, y = [], []
classes = ["early_blight", "late_blight", "healthy"]

print("🚀 Training improved SVM with enhanced feature extraction...")

for label in classes:
    class_dir = f"datasets/{label}"
    if not os.path.exists(class_dir):
        print(f"❌ Directory {class_dir} not found!")
        continue

    image_files = []
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            image_files.append(fname)

    print(f"\nProcessing {label} class - found {len(image_files)} images:")

    for i, fname in enumerate(image_files):
        try:
            img_path = os.path.join(class_dir, fname)
            print(f"  ✅ Processing {fname} ({i+1}/{len(image_files)})")
            feat = extract_features(img_path)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"  ❌ Error processing {fname}: {e}")
            continue

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ No images were successfully processed!")
    exit(1)

print(f"\n✅ Successfully processed {len(X)} images total")
print(f"📊 Feature vector dimensions: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"🏷️ Classes found: {list(le.classes_)}")

# Split data for better evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📈 Training samples: {len(X_train)}")
print(f"📊 Test samples: {len(X_test)}")

# Create pipeline with scaling and hyperparameter tuning
print("🔍 Performing hyperparameter optimization...")

# Define parameter grid for optimization
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

# Create pipeline
pipeline = make_pipeline(
    StandardScaler(),
    SVC(probability=True, random_state=42)
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,  # 3-fold CV due to small dataset
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"🎯 Best parameters: {grid_search.best_params_}")
print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")

# Get best model
best_clf = grid_search.best_estimator_

# Evaluate on test set
test_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"📊 Test accuracy: {test_accuracy:.4f}")
print(f"\n📈 Detailed classification report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# Train final model on all data
print("🔄 Training final model on all data...")
final_clf = make_pipeline(
    StandardScaler(),
    SVC(
        **{k.split('__')[1]: v for k, v in grid_search.best_params_.items() if k.startswith('svc__')},
        probability=True,
        random_state=42
    )
)
final_clf.fit(X, y_encoded)

# Save the improved model
os.makedirs("models", exist_ok=True)
joblib.dump({"clf": final_clf, "le": le}, "models/svm_model_2048.pkl")
print("✅ Improved SVM trained and saved successfully to models/svm_model_2048.pkl")

# Print training summary
unique, counts = np.unique(y, return_counts=True)
print(f"\n📋 Final training data summary:")
for class_name, count in zip(unique, counts):
    print(f"  • {class_name}: {count} images")

print(f"\n🎯 Final model performance:")
print(f"  • Test accuracy: {test_accuracy:.1%}")
print(f"  • Best hyperparameters: {grid_search.best_params_}")
print(f"  • Feature scaling: ✅ Applied")
print(f"  • Cross-validation: ✅ 3-fold")
