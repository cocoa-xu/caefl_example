import Config

config :caefl_example, :config, [
  ps_ip: "127.0.0.1",
  ps_port: 4000,
  seed: 42,
  # 5 seconds
  timeout: 5_000
]

config :caefl_example, :agents, [
  [
    id: :agent_1,
    env: ["rain"],
    num_samples: 4000,
    train_every: 800
  ]
  # [
  #   env: ["fog"],
  #   num_samples: 4000,
  #   train_every: 200
  # ],
  # [
  #   env: ["fog", "rain"],
  #   num_samples: 4000,
  #   train_every: 400
  # ]
]
