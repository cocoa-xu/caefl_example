defmodule CAEFLExample.MNIST.SimulateCameraSensor do
  use CAEFL.Sensor

  @impl CAEFL.Sensor
  def init_sensor(init_param) do
    default_backend = init_param[:default_backend] || Torchx.Backend
    Nx.global_default_backend(default_backend)

    {train_images, train_labels} = Scidata.MNIST.download()

    {images_binary, images_type, images_shape} = train_images
    train_images_binary =
      images_binary
      |> Nx.from_binary(images_type)
      |> Nx.reshape(images_shape)
      |> Nx.divide(255)

    {labels_binary, labels_type, _shape} = train_labels
    train_labels_onehot =
      labels_binary
      |> Nx.from_binary(labels_type)
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))

    {:ok, {train_images_binary, train_labels_onehot, 0}}
  end

  @impl CAEFL.Sensor
  def read_data({targets, labels, index}) do
    len = elem(Nx.shape(targets), 0)
    index =
      if index >= len do
        0
      else
        index
      end
    x = Nx.take(targets, index)
    y = Nx.take(labels, index)

    {:ok, {x, y}, {targets, labels, index + 1}}
  end
end
