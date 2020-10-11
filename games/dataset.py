import os
import glob
import gym
import textworld.gym
from gym.envs.registration import register, spec, registry


def get_game_env(game_path, requested_infos, max_episode_steps, batch_size=1, mode="train",verbose=True):
    if verbose:
        print(game_path)
    # games
    game_file_names = glob.glob(game_path)
    if not game_file_names:
        print("No files found ...")
    env_id = textworld.gym.register_games(sorted(game_file_names), requested_infos,
                                          max_episode_steps=max_episode_steps,
                                          name='cleanup-'+mode,batch_size=batch_size)
    # env_id = make_batch(env_id, batch_size=batch_size, parallel=True)
    env = gym.make(env_id)
    game_names = [os.path.basename(game_file) for game_file in game_file_names]
    return env, sorted(game_names)


def get_evaluation_game_env(game_path, requested_infos, max_episode_steps, batch_size=1, valid_or_test="valid",verbose=True):
    assert valid_or_test in ["valid", "test"]
    # eval games
    game_file_names = []

    if os.path.isdir(game_path):
        game_file_names += glob.glob(os.path.join(game_path, "*.z8"))
        if verbose:
            print(os.path.dirname(game_path), end="")
    else:
        game_file_names.append(game_path)
        if verbose:
            print(os.path.basename(game_path), end="")

    env_id = textworld.gym.register_games(sorted(game_file_names), requested_infos,
                                                max_episode_steps=max_episode_steps,
                                                name='cleanup-eval',batch_size=batch_size)
    # env_id = make_batch(env_id, batch_size=batch_size, parallel=True)
    env = gym.make(env_id)
    return env, sorted(game_file_names)


def make_batch(env_id: str, batch_size: int, parallel: bool = False) -> str:
    """ Make an environment that runs multiple games independently.
    Arguments:
        env_id:
            Environment ID that will compose a batch.
        batch_size:
            Number of independent environments to run.
        parallel:
            If True, the environment will be executed in different processes.
    Returns:
        The corresponding gym-compatible env_id to use.
    """
    batch_env_id = "batch{}-".format(batch_size) + env_id
    env_spec = spec(env_id)
    # entry_point = 'textworld.gym.envs:BatchEnv'
    entry_point = 'textworld.gym.envs:TextworldGymEnv'
    if parallel and batch_size > 1:
        # entry_point = 'textworld.gym.envs:ParallelBatchEnv'
        entry_point = 'textworld.gym.envs:TextworldBatchGymEnv'

    register(
        id=batch_env_id,
        entry_point=entry_point,
        max_episode_steps=env_spec.max_episode_steps,
        # max_episode_seconds=env_spec.max_episode_seconds,
        nondeterministic=env_spec.nondeterministic,
        reward_threshold=env_spec.reward_threshold,
        # trials=env_spec.trials,
        # Setting the 'vnc' tag avoid wrapping the env with a TimeLimit wrapper. See
        # https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/registration.py#L122
        tags={"vnc": "foo"},
        kwargs={'env_id': env_id, 'batch_size': batch_size}
    )

    return batch_env_id

