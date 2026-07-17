import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
import wandb
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decision_transformer import DecisionTransformer, DTDataset


def train(config=None):
    """
    Train the Decision Transformer on the collected CARLA dataset.
    """

    # Default config
    if config is None:
        config = {
            'dataset_path':   '/scratch/prj/inf_offroad_auto_nav/data/processed/dataset.pkl',
            'save_dir':       '/scratch/prj/inf_offroad_auto_nav/models/checkpoints',
            'state_dim':      8,
            'action_dim':     3,
            'hidden_dim':     128,
            'n_layers':       3,
            'n_heads':        1,
            'context_length': 20,
            'dropout':        0.1,
            'max_timestep':   500,
            'batch_size':     64,
            'lr':             1e-4,
            'weight_decay':   0.1,
            'n_epochs':       100,
            'early_stop':     10,
            'seed':           0,
        }

    # Reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialise W&B
    wandb.init(
        project='dt-offroad-navigation',
        name=f"dt_seed{config['seed']}",
        config=config
    )

    # Load dataset
    print(f"Loading dataset from {config['dataset_path']}...")
    with open(config['dataset_path'], 'rb') as f:
        dataset = pickle.load(f)

    train_episodes = dataset['train']
    val_episodes   = dataset['val']
    norm_stats     = dataset['norm_stats']

    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes:   {len(val_episodes)}")

    # Create datasets and dataloaders
    train_dataset = DTDataset(
        train_episodes,
        context_length=config['context_length']
    )
    val_dataset = DTDataset(
        val_episodes,
        context_length=config['context_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialise model
    model = DecisionTransformer(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        context_length=config['context_length'],
        dropout=config['dropout'],
        max_timestep=config['max_timestep']
    ).to(device)

    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # Optimiser and loss
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config['n_epochs']
    )

    # Training loop
    os.makedirs(config['save_dir'], exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(
        config['save_dir'],
        f"dt_best_seed{config['seed']}.pt"
    )

    for epoch in range(config['n_epochs']):

        # ── Training ──────────────────────────────────────────────
        model.train()
        train_losses = []

        for batch in train_loader:
            states, actions, rtg, timesteps = [
                x.to(device) for x in batch
            ]

            # Forward pass
            action_preds = model(states, actions, rtg, timesteps)

            # MSE loss on all timesteps
            loss = criterion(action_preds, actions)

            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimiser.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                states, actions, rtg, timesteps = [
                    x.to(device) for x in batch
                ]
                action_preds = model(states, actions, rtg, timesteps)
                loss = criterion(action_preds, actions)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step()

        print(
            f"Epoch {epoch+1:4d}/{config['n_epochs']} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f}"
        )

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         scheduler.get_last_lr()[0]
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'optim_state': optimiser.state_dict(),
                'val_loss':    val_loss,
                'config':      config,
                'norm_stats':  norm_stats
            }, best_model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['early_stop']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model saved:   {best_model_path}")

    wandb.finish()
    return model, best_val_loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',           type=int,   default=0)
    parser.add_argument('--dataset_path',   type=str,
                        default='/scratch/prj/inf_offroad_auto_nav/data/processed/dataset.pkl')
    parser.add_argument('--save_dir',       type=str,
                        default='/scratch/prj/inf_offroad_auto_nav/models/checkpoints')
    parser.add_argument('--n_epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--context_length', type=int,   default=20)
    parser.add_argument('--state_dim',      type=int,   default=17)
    parser.add_argument('--action_dim',     type=int,   default=6)
    args = parser.parse_args()

    config = {
        'dataset_path':   args.dataset_path,
        'save_dir':       args.save_dir,
        'state_dim':      args.state_dim,
        'action_dim':     args.action_dim,
        'hidden_dim':     128,
        'n_layers':       3,
        'n_heads':        1,
        'context_length': args.context_length,
        'dropout':        0.1,
        'max_timestep':   1000,
        'batch_size':     args.batch_size,
        'lr':             args.lr,
        'weight_decay':   0.1,
        'n_epochs':       args.n_epochs,
        'early_stop':     10,
        'seed':           args.seed,
    }

    train(config)