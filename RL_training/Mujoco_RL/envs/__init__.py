from gymnasium.envs.registration import register


register(
    id="PickBrick-v0",
    entry_point="envs.pick_brick_env:PickBrickEnv",
)
