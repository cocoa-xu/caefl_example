defmodule CAEFLExample.AgentApp do
  use Application
  alias CAEFLExample.AlexNet, as: Model

  @spec start(any, any) :: {:error, any} | {:ok, pid}
  def start(_type, _args) do
    app = Mix.Project.config()[:app]
    ps_config = Application.get_env(app, :config)
    seed = ps_config[:seed]
    ip = parse_ip(ps_config[:ps_ip])
    ps_timeout = ps_config[:timeout]

    agents_config = Application.get_env(app, :agents)
    children =
      Enum.map(agents_config, fn ac ->
        %{
          id: ac[:id],
          start: {CAEFLExample.Agent, :start_link, [
            ac[:env],
            ip,
            ps_config[:ps_port],
            ps_timeout,
            ac[:num_samples],
            ac[:train_every],
            seed
          ]},
          restart: :temporary
        }
      end)
    Supervisor.start_link(children, strategy: :one_for_one)
  end

  defp parse_ip(ip) do
    List.to_tuple(Enum.map(String.split(ip, ".", trim: true), &elem(Integer.parse(&1), 0)))
  end
end
