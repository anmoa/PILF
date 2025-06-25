import os
import sys

import argparse
import importlib.util
import torch
from datetime import datetime
from collections import defaultdict

from run_experiment import setup_experiment, get_dataloaders, execute_training_loop, Tee
from utils.training import validate
from utils.plotting import plot_metrics

def merge_metrics(main_metrics, new_metrics):
    for key, value in new_metrics.items():
        main_metrics[key].extend(value)

def run_rehearsal_experiment(config_path: str, checkpoint_path: str, rehearsal_freq: int = 4):
    spec = importlib.util.spec_from_file_location(name="config", location=config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    train_cfg = config.train_config
    log_dir = os.path.join(train_cfg['output_dir'], 'log')
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.basename(config_path).replace('.py', '').replace('_', '-')
    log_file_name = f"{current_time}-{config_name}-rehearsal.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    original_stdout = sys.stdout
    log_file = open(log_file_path, 'w')
    sys.stdout = Tee(original_stdout, log_file)
    print(f"Logging output to: {log_file_path}")

    try:
        task_suffix = "-rehearsal"
        config_name = os.path.basename(config_path).replace('.py', '').replace('_', '-')
        model, optimizer, loss_fn, pi_monitor, device, model_name = setup_experiment(
            config, base_model_name=config_name, model_name_suffix=task_suffix, checkpoint_path=checkpoint_path
        )
        
        # Load dataloaders
        svhn_train_loader, svhn_val_loader = get_dataloaders('SVHN', train_cfg['batch_size'])
        cifar_train_loader, cifar_val_loader = get_dataloaders('CIFAR10', train_cfg['batch_size'])

        # Initialize metric collectors
        full_step_metrics = defaultdict(list)
        full_epoch_metrics = defaultdict(list)
        
        global_step = 0
        start_epoch = 21 # Assuming stage 1 (20 epochs) is done
        num_rehearsal_cycles = 2
        svhn_epochs_per_cycle = 4
        rehearsal_epochs = 1
        overtime_epochs = 2

        print("\n" + "="*50)
        print(" " * 10 + f"STAGE 2 (Rehearsal): {num_rehearsal_cycles}x({svhn_epochs_per_cycle} SVHN + {rehearsal_epochs} CIFAR) + {overtime_epochs} SVHN Overtime")
        print("="*50 + "\n")

        for i in range(num_rehearsal_cycles):
            print(f"\n--- Cycle {i+1}/{num_rehearsal_cycles} ---")
            # Train on SVHN
            print(f"\n--- Training on SVHN for {svhn_epochs_per_cycle} epochs ---")
            
            model, global_step, step_metrics, epoch_metrics = execute_training_loop(
                model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
                device=device, model_name=model_name,
                train_loader=svhn_train_loader,
                val_loaders={"CIFAR10_Val": cifar_val_loader, "SVHN_Val": svhn_val_loader},
                epochs=svhn_epochs_per_cycle,
                use_pilr=(config.model_config.get('model_type') == 'pilr_moe'),
                accumulation_steps=train_cfg['accumulation_steps'],
                output_dir=train_cfg['output_dir'],
                start_epoch=start_epoch,
                global_step=global_step,
                train_fn_kwargs=train_cfg.get('train_fn_kwargs', {})
            )
            merge_metrics(full_step_metrics, step_metrics)
            merge_metrics(full_epoch_metrics, epoch_metrics)
            start_epoch += svhn_epochs_per_cycle
            
            # Save checkpoint after SVHN training block
            checkpoint_dir = os.path.join(train_cfg['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = f"{model_name}-epoch_{start_epoch - 1}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nCheckpoint saved to: {checkpoint_path}")

            # Rehearsal on CIFAR-10
            print(f"\n--- Rehearsal on CIFAR-10 for {rehearsal_epochs} epoch ---")
            model, global_step, step_metrics, epoch_metrics = execute_training_loop(
                model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
                device=device, model_name=model_name,
                train_loader=cifar_train_loader,
                val_loaders={"CIFAR10_Val": cifar_val_loader, "SVHN_Val": svhn_val_loader},
                epochs=rehearsal_epochs,
                use_pilr=(config.model_config.get('model_type') == 'pilr_moe'),
                accumulation_steps=train_cfg['accumulation_steps'],
                output_dir=train_cfg['output_dir'],
                start_epoch=start_epoch,
                global_step=global_step,
                train_fn_kwargs=train_cfg.get('train_fn_kwargs', {})
            )
            merge_metrics(full_step_metrics, step_metrics)
            merge_metrics(full_epoch_metrics, epoch_metrics)
            start_epoch += rehearsal_epochs

            # Save checkpoint after rehearsal block
            checkpoint_dir = os.path.join(train_cfg['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = f"{model_name}-epoch_{start_epoch - 1}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nCheckpoint saved to: {checkpoint_path}")

        # --- Overtime Training ---
        if overtime_epochs > 0:
            print(f"\n--- Overtime Training on SVHN for {overtime_epochs} epochs ---")
            model, global_step, step_metrics, epoch_metrics = execute_training_loop(
                model=model, optimizer=optimizer, loss_fn=loss_fn, pi_monitor=pi_monitor,
                device=device, model_name=model_name,
                train_loader=svhn_train_loader,
                val_loaders={"CIFAR10_Val": cifar_val_loader, "SVHN_Val": svhn_val_loader},
                epochs=overtime_epochs,
                use_pilr=(config.model_config.get('model_type') == 'pilr_moe'),
                accumulation_steps=train_cfg['accumulation_steps'],
                output_dir=train_cfg['output_dir'],
                start_epoch=start_epoch,
                global_step=global_step,
                train_fn_kwargs=train_cfg.get('train_fn_kwargs', {})
            )
            merge_metrics(full_step_metrics, step_metrics)
            merge_metrics(full_epoch_metrics, epoch_metrics)
            
            # Save final checkpoint after overtime
            checkpoint_dir = os.path.join(train_cfg['output_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = f"{model_name}-epoch_{start_epoch + overtime_epochs - 1}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nCheckpoint saved to: {checkpoint_path}")

        # --- FINAL PLOTTING & EVALUATION ---
        plot_metrics_kwargs = {}
        use_pilr = config.model_config.get('model_type') == 'pilr_moe'
        train_fn_kwargs = train_cfg.get('train_fn_kwargs', {})
        if use_pilr:
            if train_fn_kwargs.get('pilr_mode', 'gate') == 'gate':
                if 'pilr_surprise_values' in full_epoch_metrics:
                    plot_metrics_kwargs['pilr_surprise_values'] = full_epoch_metrics['pilr_surprise_values']
                    plot_metrics_kwargs['pilr_decisions'] = full_epoch_metrics['pilr_decisions']
            elif train_fn_kwargs.get('pilr_mode') == 'lr_scheduler':
                if 'train_lr_mod' in full_epoch_metrics:
                    plot_metrics_kwargs['lr_mod_values'] = full_epoch_metrics['train_lr_mod']

        plot_metrics(full_step_metrics, full_epoch_metrics, train_cfg['output_dir'], model_name=model_name, **plot_metrics_kwargs)
        print(f"\nPlots for full experiment saved to: {os.path.abspath(train_cfg['output_dir'])}")

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Training completed. Log saved to: {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Run a continual learning experiment with rehearsal.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to a model checkpoint from stage 1.")
    parser.add_argument('--rehearsal_freq', type=int, default=4, help="Frequency of rehearsal epochs.")
    args = parser.parse_args()
    run_rehearsal_experiment(args.config, args.checkpoint_path, args.rehearsal_freq)

if __name__ == "__main__":
    main()
