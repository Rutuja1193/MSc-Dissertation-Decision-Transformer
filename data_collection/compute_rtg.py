import numpy as np
import pickle
import os


def compute_return_to_go(episodes, gamma=1.0):
    """
    Compute return-to-go for every timestep in every episode.
    RTG_t = sum of all future rewards from t onwards.
    Works backwards through each episode.

    Args:
        episodes: list of episode dicts from collect_episodes.py
        gamma: discount factor (1.0 = undiscounted, standard for DT)

    Returns:
        episodes with 'rtg' key added to each episode dict
    """
    print(f"Computing return-to-go for {len(episodes)} episodes...")

    for i, episode in enumerate(episodes):
        rewards = episode['rewards']
        T = len(rewards)

        rtg = np.zeros(T, dtype=np.float32)

        # Work backwards
        rtg[-1] = rewards[-1]
        for t in reversed(range(T - 1)):
            rtg[t] = rewards[t] + gamma * rtg[t + 1]

        episode['rtg'] = rtg

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(episodes)} episodes")

    print("Return-to-go computation complete.")
    return episodes


def normalise_dataset(episodes):
    """
    Normalise states, actions, and RTG across the full dataset.
    Stores mean and std for use at inference time.
    """
    print("Normalising dataset...")

    all_states = np.concatenate(
        [ep['states'] for ep in episodes], axis=0
    )
    all_actions = np.concatenate(
        [ep['actions'] for ep in episodes], axis=0
    )
    all_rtg = np.concatenate(
        [ep['rtg'] for ep in episodes], axis=0
    )

    state_mean = all_states.mean(axis=0)
    state_std  = all_states.std(axis=0) + 1e-8

    action_mean = all_actions.mean(axis=0)
    action_std  = all_actions.std(axis=0) + 1e-8

    rtg_mean = all_rtg.mean()
    rtg_std  = all_rtg.std() + 1e-8
    rtg_max  = all_rtg.max()

    for episode in episodes:
        episode['states']  = (
            (episode['states']  - state_mean)  / state_std
        )
        episode['actions'] = (
            (episode['actions'] - action_mean) / action_std
        )
        episode['rtg']     = (
            (episode['rtg']     - rtg_mean)    / rtg_std
        )

    normalisation_stats = {
        'state_mean':  state_mean,
        'state_std':   state_std,
        'action_mean': action_mean,
        'action_std':  action_std,
        'rtg_mean':    rtg_mean,
        'rtg_std':     rtg_std,
        'rtg_max':     rtg_max,
    }

    print(f"State dim  : {state_mean.shape[0]}")
    print(f"Action dim : {action_mean.shape[0]}")
    print(f"RTG max    : {rtg_max:.2f}")
    print(f"RTG mean   : {rtg_mean:.2f}")
    print("Normalisation complete.")

    return episodes, normalisation_stats


def prepare_dataset(
    raw_path='../data/raw/episodes_final.pkl',
    save_path='../data/processed/dataset.pkl'
):
    """
    Full pipeline: load raw episodes, compute RTG,
    normalise, and save processed dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Loading raw episodes from {raw_path}...")
    with open(raw_path, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} episodes.")

    # Step 1: compute return-to-go
    episodes = compute_return_to_go(episodes, gamma=1.0)

    # Step 2: normalise
    episodes, norm_stats = normalise_dataset(episodes)

    # Step 3: train/val/test split 80/10/10
    n = len(episodes)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    np.random.seed(42)
    indices = np.random.permutation(n)

    train_eps = [episodes[i] for i in indices[:n_train]]
    val_eps   = [episodes[i] for i in indices[n_train:n_train+n_val]]
    test_eps  = [episodes[i] for i in indices[n_train+n_val:]]

    dataset = {
        'train':      train_eps,
        'val':        val_eps,
        'test':       test_eps,
        'norm_stats': norm_stats,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n=== Dataset Prepared ===")
    print(f"Train episodes : {len(train_eps)}")
    print(f"Val episodes   : {len(val_eps)}")
    print(f"Test episodes  : {len(test_eps)}")
    print(f"Saved to       : {save_path}")

    return dataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path',  type=str,
                        default='../data/raw/episodes_final.pkl')
    parser.add_argument('--save_path', type=str,
                        default='../data/processed/dataset.pkl')
    args = parser.parse_args()

    prepare_dataset(
        raw_path=args.raw_path,
        save_path=args.save_path
    )