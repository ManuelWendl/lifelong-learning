import os

import wandb


def get_wandb_checkpoint(run_id, entity):
    api = wandb.Api(overrides={"entity": entity})
    artifact = api.artifact(f"ss2r/checkpoint:{run_id}")
    download_dir = artifact.download(f"{get_state_path()}/{run_id}")
    return download_dir


def get_simulator_states(run_id, entity):
    """Download a simulator_states artifact and return the local .npz path."""
    api = wandb.Api(overrides={"entity": entity})
    artifact = api.artifact(f"ss2r/simulator_states:{run_id}")
    download_dir = artifact.download(f"{get_state_path()}/{run_id}_states")
    npz_files = [f for f in os.listdir(download_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in artifact download dir: {download_dir}")
    return os.path.join(download_dir, npz_files[0])


def get_state_path() -> str:
    log_path = os.getcwd() + "/ckpt"
    return log_path
