defmodule CAEFLExample.MNIST.SimulateEnvSensor do
  use CAEFL.Sensor

  @impl CAEFL.Sensor
  def init_sensor(init_param) do
    {:ok, init_param}
  end

  @impl CAEFL.Sensor
  def read_data(simulate_param=%{precipitation: range}) do
    [lo, hi] = range
    p = Nx.to_number(Nx.random_uniform(1, lo, hi))
    {:ok, %{precipitation: p}, simulate_param}
  end

  @impl CAEFL.Sensor
  def read_data(simulate_param=%{visibility: range}) do
    [lo, hi] = range
    v = Nx.to_number(Nx.random_uniform(1, lo, hi))
    {:ok, %{visibility: v}, simulate_param}
  end
end
