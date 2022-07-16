defmodule CAEFLExample.Agent do
  use GenServer
  require Logger
  alias CAEFLExample.MNIST.SimulateAgent
  alias CAEFLExample.MNIST.SimulateEnvironment
  alias CAEFLExample.MNIST.SimulateEnvSensor
  alias CAEFLExample.MNIST.SimulateCameraSensor
  alias CAEFLExample.AlexNet, as: Model

  def start(env, ps_ip, ps_port, ps_timeout, num_samples, train_every, seed) do
    GenServer.start(__MODULE__, {env, ps_ip, ps_port, ps_timeout, num_samples, train_every, seed})
  end

  def start_link(env, ps_ip, ps_port, ps_timeout, num_samples, train_every, seed) do
    GenServer.start_link(__MODULE__, {env, ps_ip, ps_port, ps_timeout, num_samples, train_every, seed})
  end

  @impl true
  def init({env, ps_ip, ps_port, ps_timeout, num_samples, train_every, seed}) do
    # simulate data sensor(s), here we're using a fake camera
    {:ok, camera} = SimulateCameraSensor.start_link(env)

    # simulate environment sensor(s)
    # environment sensor(s) will be used to generate environment tags
    #   note that here we are running simulations
    #   for real usage, here we should initialise available environment sensors
    #   like temperature, humidity, geolocation and etc.
    env_sensors =
      Enum.map(env, fn env_name ->
        {:ok, sensor} =
          case env_name do
            "rain" ->
              SimulateEnvSensor.start_link(%{precipitation: [20, 100]})
            "fog" ->
              SimulateEnvSensor.start_link(%{visibility: [5, 1000]})
          end
        sensor
      end)

    # then we can initialise `CAEFL.Environment` with a list of environment sensors
    {:ok, mnist_env} = SimulateEnvironment.start_link(env_sensors)

    # now we have a very simple agent available
    # we can start the agent
    opts = [train_every: train_every, num_samples: num_samples, seed: seed]
    {:ok, agent} = SimulateAgent.start_link(
      ps_ip, ps_port, ps_timeout,
      [camera], mnist_env,
      &Model.serialize/1, &Model.deserialize/1,
      opts
    )

    # start the runloop
    runloop_pid = SimulateAgent.start_runloop(agent)
    {:ok, {agent, runloop_pid}}
  end
end
