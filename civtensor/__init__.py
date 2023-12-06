from gymnasium.envs.registration import register

register(
    id="civtensor/TensorBaselineEnv-v0",
    entry_point="civtensor.envs.freeciv_tensor_env:TensorBaselineEnv",
)
