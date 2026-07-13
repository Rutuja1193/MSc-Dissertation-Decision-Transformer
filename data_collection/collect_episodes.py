import numpy as np
import pickle
import os
import time
import random
from carla_env import CarlaEnv


def collect_dataset(
    num_episodes=500,
    max_steps_per_episode=500,
    save_dir='../data/raw',
    save_every=50
):
    """
    Collect offline trajectory dataset from CARLA.
    Each episode logs: states, images, actions, rewards, timesteps.
    Saves incrementally every save_every episodes.
    """

    os.makedirs(save_dir, exist_ok=True)

    env = CarlaEnv(
        host='localhost',
        port=2000,
        town='Town07',
        fps=10
    )

    weather_variants = env.get_weather_variants()
    all_episodes = []
    episode_returns = []

    print(f"Starting data collection: {num_episodes} episodes")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Saving to: {save_dir}")

    for ep in range(num_episodes):

        weather = random.choice(weather_variants)

        try:
            state_vector, image = env.reset(weather=weather)
        except Exception as e:
            print(f"Episode {ep} reset failed: {e}")
            continue

        ep_states = []
        ep_images = []
        ep_actions = []
        ep_rewards = []
        ep_timesteps = []

        episode_reward = 0.0

        for t in range(max_steps_per_episode):

            action = env._get_action()

            if image is not None:
                ep_states.append(state_vector.copy())
                ep_images.append(image.copy())
                ep_actions.append(action.copy())
                ep_timesteps.append(t)

            try:
                next_state, next_image, reward, done = env.step()
            except Exception as e:
                print(f"Episode {ep} step {t} failed: {e}")
                break

            ep_rewards.append(reward)
            episode_reward += reward

            state_vector = next_state
            image = next_image

            if done:
                print(f"  Episode {ep} ended early at step {t}")
                break

        # Only save episodes with meaningful data
        min_steps = 50
        if len(ep_states) >= min_steps:
            episode = {
                'states':    np.array(ep_states,    dtype=np.float32),
                'images':    np.array(ep_images,    dtype=np.uint8),
                'actions':   np.array(ep_actions,   dtype=np.float32),
                'rewards':   np.array(ep_rewards,   dtype=np.float32),
                'timesteps': np.array(ep_timesteps, dtype=np.int32),
                'weather':   str(weather),
                'length':    len(ep_states)
            }
            all_episodes.append(episode)
            episode_returns.append(episode_reward)

            print(
                f"Episode {ep:4d} | "
                f"Steps: {len(ep_states):4d} | "
                f"Return: {episode_reward:8.2f} | "
                f"Total collected: {len(all_episodes)}"
            )
        else:
            print(
                f"Episode {ep} discarded — "
                f"too short ({len(ep_states)} steps)"
            )

        # Save checkpoint every save_every episodes
        if (ep + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                save_dir, f'episodes_checkpoint_{ep+1}.pkl'
            )
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(all_episodes, f)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final save
    final_path = os.path.join(save_dir, 'episodes_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(all_episodes, f)

    # Save statistics
    stats = {
        'num_episodes':   len(all_episodes),
        'total_timesteps': sum(ep['length'] for ep in all_episodes),
        'mean_return':    np.mean(episode_returns),
        'std_return':     np.std(episode_returns),
        'max_return':     np.max(episode_returns),
        'min_return':     np.min(episode_returns),
    }

    stats_path = os.path.join(save_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    print("\n=== Data Collection Complete ===")
    print(f"Episodes collected : {stats['num_episodes']}")
    print(f"Total timesteps    : {stats['total_timesteps']}")
    print(f"Mean return        : {stats['mean_return']:.2f}")
    print(f"Max return         : {stats['max_return']:.2f}")
    print(f"Saved to           : {final_path}")

    env.close()
    return all_episodes, stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--max_steps',    type=int, default=500)
    parser.add_argument('--save_dir',     type=str, 
                        default='../data/raw')
    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        save_dir=args.save_dir,
        save_every=50
    )