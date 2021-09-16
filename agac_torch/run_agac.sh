mpiexec -n 3 python agac_torch/run_agac_experiment.py \
    --config-path agac_torch/configs/minigrid.yaml \
    --env-name=MiniGrid-KeyCorridorS4R3-v0 \
    --seed 123