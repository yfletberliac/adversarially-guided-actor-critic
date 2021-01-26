import os

from agac.agac import AGAC
from core.cmd_util import make_doom_env
from core.tf_util import linear_schedule

for env_id in ["VizdoomMyWayHome-v0"]:
    for seed in [123]:
        log_dir = "./logs/%s/AGAC_seed%s" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_doom_env(env_id, 1, seed, monitor_dir=log_dir)
        model = AGAC('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=2048, nminibatches=8, agac_c=linear_schedule(0.00004), beta_adv=0.00004,
                     learning_rate=0.001, ent_coef=0.01, episodic_count=False)
        model.learn(total_timesteps=10000000, tb_log_name="tb/AGAC")
