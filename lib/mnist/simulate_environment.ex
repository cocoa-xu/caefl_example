defmodule CAEFLExample.MNIST.SimulateEnvironment do
  use CAEFL.Environment

  @impl CAEFL.Environment
  def environment_sensors(state = list_of_sensors) do
    {:ok, list_of_sensors, state}
  end

  @impl CAEFL.Environment
  def transform(%{precipitation: p}) when p > 10 do
    # Set `:rain` to `true` when precipitation is above 10 mm
    %{rain: true}
  end

  @impl CAEFL.Environment
  def transform(%{precipitation: _}) do
    %{rain: false}
  end

  @impl CAEFL.Environment
  def transform(%{visibility: v}) when v < 50 do
    # Set `:fog` to `true` when visibility is below 50 meters
    %{fog: true}
  end

  @impl CAEFL.Environment
  def transform(%{visibility: _}) do
    %{fog: false}
  end
end
