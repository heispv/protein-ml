import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from pathlib import Path
import joblib  # For saving the trained model

def main():
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    
    # Load the dataset
    data_file = Path('data/features/norm_protein_features.csv')
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Load selected features
    selected_features_file = Path('data/features/selected_features/final_top_20_features.csv')
    if not selected_features_file.exists():
        raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")
    
    selected_features_df = pd.read_csv(selected_features_file)
    selected_features = selected_features_df['feature'].tolist()
    
    # Ensure that selected features are in the dataframe
    for feature in selected_features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in the dataset.")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Assign fold numbers to each sample
    num_folds = 5
    df['fold'] = df.index % num_folds
    
    # Prepare data
    X = df[selected_features]
    y = df['label']
    folds = df['fold']
    
    # Store results
    results = []
    best_hyperparams = {'C': [], 'gamma': []}
    
    # Define initial hyperparameter grid
    C_values = np.linspace(1, 10, 8)
    gamma_values = [0.01, 0.03, 0.065, 0.1, 0.13, 0.165, 0.2, 0.25, 0.3]  # Fixed typo: 0,65 to 0.065
    
    print("=== Step 1: Initial 5-Fold Hyperparameter Tuning ===")
    for i in range(num_folds):
        # Define fold numbers for training, validation, and testing
        train_folds = [(i + j) % num_folds for j in range(3)]  # Folds i, i+1, i+2
        val_fold = (i + 3) % num_folds
        test_fold = (i + 4) % num_folds
        
        # Split the data
        train_idx = df[df['fold'].isin(train_folds)].index
        val_idx = df[df['fold'] == val_fold].index
        test_idx = df[df['fold'] == test_fold].index
        
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        best_mcc = -1
        best_C = None
        best_gamma = None
        
        # Grid search over hyperparameters
        for C in C_values:
            for gamma in gamma_values:
                # Create the SVM model
                svm = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', random_state=random_seed)
                
                # Train on training set
                svm.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred_val = svm.predict(X_val)
                
                # Compute MCC
                mcc = matthews_corrcoef(y_val, y_pred_val)
                
                # Update best hyperparameters
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_C = C
                    best_gamma = gamma
        
        # Store the best hyperparameters
        best_hyperparams['C'].append(best_C)
        best_hyperparams['gamma'].append(best_gamma)
        
        # Train final model with best hyperparameters on training set
        svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf', class_weight='balanced', random_state=random_seed)
        svm.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_test = svm.predict(X_test)
        
        # Compute metrics on test set
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        
        # Collect results
        result = {
            'fold': i + 1,
            'best_C': best_C,
            'best_gamma': best_gamma,
            'validation_mcc': best_mcc,
            'test_mcc': test_mcc,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        results.append(result)
        print(f"Fold {i+1}: Best C={best_C}, gamma={best_gamma}, Validation MCC={best_mcc:.4f}, Test MCC={test_mcc:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Determine the min and max of best C and gamma
    min_C = min(best_hyperparams['C'])
    max_C = max(best_hyperparams['C'])
    min_gamma = min(best_hyperparams['gamma'])
    max_gamma = max(best_hyperparams['gamma'])
    
    print("\n=== Step 2: Refined Hyperparameter Tuning on Entire Training Dataset ===")
    print(f"Refined C range: {min_C} to {max_C}")
    print(f"Refined gamma range: {min_gamma} to {max_gamma}")
    
    # Define refined hyperparameter grid
    # To ensure that min and max are included, expand the range slightly if needed
    # For example, use log-spaced values for better coverage
    refined_C_values = np.linspace(min_C, max_C, 8)
    refined_gamma_values = np.linspace(min_gamma, max_gamma, 8)
    
    # Initialize variables to store the best hyperparameters from the refined grid search
    final_best_mcc = -1
    final_best_C = None
    final_best_gamma = None
    
    # Perform grid search on the entire dataset
    for C in refined_C_values:
        for gamma in refined_gamma_values:
            # Create the SVM model
            svm = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', random_state=random_seed)
            
            # Train on the entire dataset
            svm.fit(X, y)
            
            # Perform cross-validation to evaluate MCC
            y_pred = cross_val_predict(svm, X, y, cv=num_folds, n_jobs=-1)
            mcc = matthews_corrcoef(y, y_pred)
            
            # Update best hyperparameters
            if mcc > final_best_mcc:
                final_best_mcc = mcc
                final_best_C = C
                final_best_gamma = gamma
    
    print(f"Refined Best C={final_best_C}, gamma={final_best_gamma}, Cross-Validated MCC={final_best_mcc:.4f}")
    
    # Train final model with refined hyperparameters on the entire dataset
    final_svm = SVC(C=final_best_C, gamma=final_best_gamma, kernel='rbf', class_weight='balanced', random_state=random_seed)
    final_svm.fit(X, y)
    
    # Save the final trained SVM model for later use
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = output_dir / 'final_svm_model.joblib'
    joblib.dump(final_svm, model_file)
    print(f"Final trained SVM model saved to {model_file}")
    
    # Save all results to CSV
    # Save initial 5-fold results
    initial_results_file = output_dir / 'svm_initial_5fold_results.csv'
    results_df.to_csv(initial_results_file, index=False)
    print(f"Initial 5-Fold Results saved to {initial_results_file}")
    
    # Save final refined hyperparameters
    refined_results = {
        'final_best_C': final_best_C,
        'final_best_gamma': final_best_gamma,
        'final_cross_validated_MCC': final_best_mcc
    }
    refined_results_df = pd.DataFrame([refined_results])
    refined_results_file = output_dir / 'svm_refined_hyperparameters.csv'
    refined_results_df.to_csv(refined_results_file, index=False)
    print(f"Refined Hyperparameters saved to {refined_results_file}")
    
    print("=== Hyperparameter Tuning and Model Saving Completed ===")

if __name__ == '__main__':
    main()
