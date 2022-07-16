defmodule CAEFLExample.MNIST.SimulateAgent do
  use CAEFL.SimpleAgent
  alias CAEFLExample.AlexNet, as: Model

  alias __MODULE__, as: T

  @impl CAEFL.SimpleAgent
  def should_update_local_model?(self) do
    train_every = self.opts[:train_every]
    count = Enum.count(self.sampled_data)
    self =
      if count >= train_every do
        new_opts = Keyword.update!(self.opts, :num_samples, fn _ -> self.opts[:num_samples] - count end)
        %T{self | opts: new_opts}
      else
        self
      end
    {count >= train_every, self}
  end

  @impl CAEFL.SimpleAgent
  def should_stop?(self) do
    if self.opts[:num_samples] <= 0 do
      {true, :shutdown, :stop, self}
    else
      false
    end
  end

  @impl CAEFL.SimpleAgent
  def update_local_tagged_model(model_param, _env_tag, data) do
    data = List.flatten(data)

    {targets, labels} =
      Enum.reduce(data, {[], []}, fn {x, y}, {targets, labels} ->
        {[x | targets], [y | labels]}
      end)

    targets = Nx.reshape(Nx.stack(targets), {:auto, 784}, names: [:batch, :input])
    targets = Nx.to_batched_list(targets, elem(Nx.shape(targets), 0))

    labels = Nx.reshape(Nx.as_type(Nx.stack(labels), :s64), {:auto, 10}, names: [:batch, :output])
    labels = Nx.to_batched_list(labels, elem(Nx.shape(labels), 0))

    new_model = Model.train(targets, labels, Model.deserialize(model_param))
    Model.serialize(new_model)
  end

  @impl CAEFL.SimpleAgent
  def get_initial_model(opts) do
    Model.serialize(Model.init_params(opts[:seed], input_shape: 784, num_classes: 10))
  end

  @impl CAEFL.SimpleAgent
  def should_push_local_tagged_model_to_parameter_server?(_self, _model_tag, _model_param) do
    true
  end
end
