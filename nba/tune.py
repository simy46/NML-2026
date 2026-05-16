import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import os
import gc
import copy
import yaml
import argparse
import optuna
import pandas as pd
import torch

from training import NBATrainer 

def main():
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization.")
    parser.add_argument(
        "--config", 
        type=str,  
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=30, 
        help="Number of Optuna trials to run."
    )
    args = parser.parse_args()

    # Load base configuration
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    model_name = base_cfg.get("MODEL_NAME", "Unknown_Model")
    
    def objective(trial):
        trial_cfg = copy.deepcopy(base_cfg)
        
        if "wandb" in trial_cfg:
            trial_cfg["wandb"]["enabled"] = False

        # Sample hyperparameters based on ranges in yaml
        opt_config = trial_cfg.get("optimization", {})
        if not opt_config:
            raise ValueError("No 'optimization' block found in config.")

        for param_name, search_space in opt_config.items():
            param_type = search_space.get("type", "float")
            section = search_space.get("section")
            
            if param_type == "float":
                val = trial.suggest_float(
                    param_name, 
                    search_space["low"], 
                    search_space["high"], 
                    log=search_space.get("log", False)
                )
            elif param_type == "int":
                val = trial.suggest_int(
                    param_name, 
                    search_space["low"], 
                    search_space["high"],
                    log=search_space.get("log", False)
                )
            elif param_type == "categorical":
                val = trial.suggest_categorical(param_name, search_space["choices"])
            else:
                continue
            
            # Apply the sampled value to the correct section in the config
            if section and section in trial_cfg:
                trial_cfg[section][param_name] = val
            else:
                # Fallback: search common sections if not specified
                for s in ["training", "model", "data"]:
                    if s in trial_cfg and param_name in trial_cfg[s]:
                        trial_cfg[s][param_name] = val
                        break
        
        trial_trainer = NBATrainer(trial_cfg)
        
    
        history = trial_trainer.train_with_early_stopping(
            test_size=0.2, 
            max_epoch=50, 
            patience=5,      
            min_delta=1e-4
        )
        
        if not history:
            return float('inf')
            
        # The objective is to minimize the best test loss achieved during the run
        best_val_loss = min(epoch_metrics["test_loss"] for epoch_metrics in history)

        del trial_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return best_val_loss

    print(f"Starting Optuna Hyperparameter Optimization Study for {model_name}")
    
    save_dir = os.path.join("saved_models", model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    db_path = os.path.join(save_dir, "optimization_study.db")
    csv_path = os.path.join(save_dir, "optimization_results.csv")
    
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"{model_name}_study", 
        storage=storage_name, 
        direction="minimize",
        load_if_exists=True 
    )
    study.optimize(objective, n_trials=args.trials)
    
    # Export results to CSV
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print("Optimization Complete!")
    print(f"Database saved to: {db_path}")
    print(f"Results CSV saved to: {csv_path}")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Loss): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()