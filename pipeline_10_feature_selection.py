import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import os
from pathlib import Path
import random

def main():
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Load the dataset
    data_file = Path('data/features/protein_features.csv')
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Assign fold numbers to each sample
    num_folds = 5
    df['fold'] = df.index % num_folds

    # Get feature columns (excluding 'accession_id', 'label', 'fold')
    feature_columns = [col for col in df.columns if col not in ['accession_id', 'label', 'fold']]

    # Store selected features from each fold
    fold_selected_features = []

    # Perform cross-validation as per the user's requirement
    for i in range(num_folds):
        # Define fold numbers for training, validation, and testing
        train_folds = [(i + j) % num_folds for j in range(3)]  # Folds i, i+1, i+2
        val_fold = (i + 3) % num_folds
        test_fold = (i + 4) % num_folds

        # Split the data
        train_data = df[df['fold'].isin(train_folds)]
        val_data = df[df['fold'] == val_fold]
        test_data = df[df['fold'] == test_fold]

        X_train = train_data[feature_columns]
        y_train = train_data['label']
        X_val = val_data[feature_columns]
        y_val = val_data['label']
        X_test = test_data[feature_columns]
        y_test = test_data['label']

        # Train the Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        rf.fit(X_train, y_train)

        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        })

        # Select top 10 features
        top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
        selected_features = top_features['feature'].tolist()
        fold_selected_features.append(selected_features)

        # Save the feature importances for this fold
        output_dir = Path('data/features/selected_features')
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_num = i + 1  # Folds numbered from 1 to 5
        output_file = output_dir / f'fold_{fold_num}_feature_importances.csv'
        feature_importance_df.to_csv(output_file, index=False)

        print(f"Fold {fold_num}: Top 20 selected features saved to {output_file}")

    # Aggregate selected features from all folds
    all_selected_features = [feature for fold_features in fold_selected_features for feature in fold_features]
    feature_counter = Counter(all_selected_features)

    # Get the most common features across all folds
    common_features = feature_counter.most_common(20)
    # final_selected_features = [feature for feature, count in common_features]

    # Save the final selected features
    final_features_df = pd.DataFrame(common_features, columns=['feature', 'count'])
    final_output_file = output_dir / 'final_top_20_features.csv'
    final_features_df.to_csv(final_output_file, index=False)

    print(f"Final top 20 selected features saved to {final_output_file}")

if __name__ == "__main__":
    main()
