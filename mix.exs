defmodule CaeflExample.MixProject do
  use Mix.Project

  def project do
    [
      app: :caefl_example,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      mod: {CAEFLExample.AgentApp, []},
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:caefl, "~> 0.1.0", github: "cocoa-xu/caefl"}
    ]
  end
end
