"""python -m rl.ladi_rl.main [hydra overrides]  (config: rl/config/ladi_rl_trainer.yaml)"""

from __future__ import annotations

import os
import socket
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_ladi_overrides(config):
    """ladi.full_window: the SDE window covers every flow step."""
    if OmegaConf.select(config, "actor_rollout_ref.ladi.full_window", default=False):
        steps = int(config.actor_rollout_ref.rollout.pipeline.num_inference_steps)
        config.actor_rollout_ref.rollout.algo.sde_window_size = steps
        config.actor_rollout_ref.rollout.algo.sde_window_range = [0, steps]
    return config


class LaDiTaskRunner:
    def run(self, config):
        import torch
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.ppo.utils import need_reference_policy

        import rl.ladi_rl  # noqa: F401
        from ladir.tokenizer import load_tokenizer

        from .dataset import LaDiRLDataset, collate_fn
        from .trainer import LaDiRayTrainer
        from .worker import LaDiActorRolloutRefWorker

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)
        if OmegaConf.select(config, "actor_rollout_ref.ladi.ar_backend", default="vllm") == "vllm":
            # the actor workers only see actor_rollout_ref
            from omegaconf import open_dict

            from .vllm_ar import ar_model_dict, ar_rollout_dict, register
            register()
            with open_dict(config):
                config.actor_rollout_ref.ladi._ar_rollout = ar_rollout_dict(config)
                config.actor_rollout_ref.ladi._ar_model = ar_model_dict(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        role = Role.ActorRolloutRef if need_reference_policy(config) else Role.ActorRollout
        role_worker_mapping = {role: ray.remote(LaDiActorRolloutRefWorker)}
        mapping = {role: "global_pool"}
        config.reward.reward_model.nnodes = config.trainer.nnodes
        config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
        pools = {"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=pools, mapping=mapping)

        tokenizer = load_tokenizer(config.actor_rollout_ref.model.path)
        train_ds = LaDiRLDataset(config.data.train_files, config.data.get("train_max_samples", -1))
        val_ds = LaDiRLDataset(config.data.val_files, config.data.get("val_max_samples", -1))
        g = torch.Generator().manual_seed(int(config.data.get("seed", None) or 1))
        sampler = torch.utils.data.RandomSampler(train_ds, generator=g) if config.data.get("shuffle", True) \
            else torch.utils.data.SequentialSampler(train_ds)
        trainer = LaDiRayTrainer(config=config, tokenizer=tokenizer, processor=None,
                                 role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager,
                                 ray_worker_group_cls=RayWorkerGroup, train_dataset=train_ds, val_dataset=val_ds,
                                 collate_fn=collate_fn, train_sampler=sampler)
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path=CONFIG_DIR, config_name="ladi_rl_trainer", version_base=None)
def main(config):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    OmegaConf.resolve(config)
    apply_ladi_overrides(config)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if not ray.is_initialized():
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(),
                                      config.ray_kwargs.get("ray_init", {}).get("runtime_env", {}))
        env = dict(runtime_env.get("env_vars", {}))
        pp = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + pp if pp else "")
        env["PYTHONUNBUFFERED"] = "1"
        runtime_env["env_vars"] = env
        init_kwargs = {**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})),
                       "runtime_env": OmegaConf.to_container(runtime_env)}
        ray.init(**init_kwargs)
    runner = ray.remote(num_cpus=1)(LaDiTaskRunner).remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
