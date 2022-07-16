defmodule CAEFLExample.AlexNet do
  import Nx.Defn

  @derive {Nx.Container, containers: [:weights, :num_classes]}
  defstruct weights: nil, num_classes: 0
  alias __MODULE__, as: T

  def serialize(%T{weights: weights, num_classes: num_classes}) do
    Nx.serialize({weights, num_classes})
  end

  def deserialize(binary_model) do
    {weights, num_classes} = Nx.deserialize(binary_model)
    %T{weights: weights, num_classes: num_classes}
  end

  def load_model(path) do
    deserialize(File.read!(path))
  end

  def save_model(%T{} = model, path) do
    File.write!(path, serialize(model))
  end

  def init_params(seed, opts \\ []) do
    :rand.seed(:default, seed)
    init_random_params(opts)
  end

  defn init_random_params(opts \\ []) do
    # 3 layers
    #  1. Dense(300)
    #  2. Dense(100)
    #  3. Dense(10) with softmax
    input_shape = opts[:input_shape]
    num_classes = opts[:num_classes]

    w1 = Nx.random_normal({input_shape, 300}, 0.0, 0.1, names: [:input, :layer1])
    b1 = Nx.random_normal({300}, 0.0, 0.1, names: [:layer1])
    w2 = Nx.random_normal({300, 100}, 0.0, 0.1, names: [:layer1, :layer2])
    b2 = Nx.random_normal({100}, 0.0, 0.1, names: [:layer2])
    w3 = Nx.random_normal({100, 10}, 0.0, 0.1, names: [:layer2, :output])
    b3 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    %T{weights: {w1, b1, w2, b2, w3, b3}, num_classes: num_classes}
  end

  defn softmax(logits) do
    Nx.exp(logits) /
      Nx.sum(Nx.exp(logits), axes: [:output], keep_axes: true)
  end

  defn predict(%T{weights: {w1, b1, w2, b2, w3, b3}}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Nx.logistic()
    |> Nx.dot(w3)
    |> Nx.add(b3)
    |> softmax()
  end

  defn accuracy(%T{} = model, batch_images, batch_labels) do
    Nx.mean(
      Nx.equal(
        Nx.argmax(batch_labels, axis: :output),
        Nx.argmax(predict(model, batch_images), axis: :output)
      )
      |> Nx.as_type({:s, 8})
    )
  end

  def confusion_matrix(y_true, y_pred) do
    y_true = Nx.as_type(y_true, {:s, 32})
    y_pred = Nx.as_type(y_pred, {:s, 32})

    zeros = Nx.broadcast(0, {10, 10})
    indices = Nx.stack([y_true, y_pred], axis: 1)
    updates = Nx.broadcast(1, {Nx.size(y_true)})

    Nx.indexed_add(zeros, indices, updates)
  end

  defp do_f1_score(y_true, y_pred, opts) do
    average = opts[:average]
    num_classes = opts[:num_classes]

    if average == :micro do
      y_pred
      |> Nx.equal(y_true)
      |> Nx.mean()
    else
      cm = confusion_matrix(y_true, y_pred)
      true_positive = Nx.take_diagonal(cm)

      false_positive = Nx.subtract(Nx.sum(cm, axes: [0]), true_positive)
      false_negative = Nx.subtract(Nx.sum(cm, axes: [1]), true_positive)
      precision = Nx.divide(true_positive, Nx.add(Nx.add(true_positive, false_positive), 1.0e-16))
      recall = Nx.divide(true_positive, Nx.add(Nx.add(true_positive, false_negative), 1.0e-16))

      per_class_f1 =
        Nx.divide(
          Nx.multiply(2, Nx.multiply(precision, recall)),
          Nx.add(Nx.add(precision, recall), 1.0e-16)
        )

      case average do
        nil ->
          per_class_f1

        :macro ->
          Nx.mean(per_class_f1)

        :weighted ->
          support = Nx.sum(Nx.equal(y_true, Nx.new_axis(Nx.iota({num_classes}), 1)), axes: [1])
          Nx.sum(Nx.multiply(per_class_f1, Nx.divide(support, Nx.sum(support))))
      end
    end
  end

  def f1_score(%T{} = model, batch_images, batch_labels, opts \\ []) do
    do_f1_score(
      Nx.argmax(batch_labels, axis: :output),
      Nx.argmax(predict(model, batch_images), axis: :output),
      opts
    )
  end

  defn loss(%T{} = model, batch_images, batch_labels) do
    preds = predict(model, batch_images)
    -Nx.sum(Nx.mean(Nx.log(preds) * batch_labels, axes: [:output]))
  end

  defn update(%T{weights: weights = {w1, b1, w2, b2, w3, b3}}, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3} =
      grad(weights, &loss(%T{weights: &1}, batch_images, batch_labels))

    %T{
      weights: {
        w1 - grad_w1 * step,
        b1 - grad_b1 * step,
        w2 - grad_w2 * step,
        b2 - grad_b2 * step,
        w3 - grad_w3 * step,
        b3 - grad_b3 * step
      }
    }
  end

  defn update_with_averages(
         %T{} = cur_model,
         imgs,
         tar,
         avg_loss,
         avg_accuracy,
         total,
         learning_rate
       ) do
    batch_loss = loss(cur_model, imgs, tar)
    batch_accuracy = accuracy(cur_model, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total
    {update(cur_model, imgs, tar, learning_rate), avg_loss, avg_accuracy}
  end

  def train_epoch(%T{} = cur_model, x, labels, learning_rate) do
    total_batches = Enum.count(x)

    {new_model, epoch_avg_loss, epoch_avg_acc} =
      x
      |> Enum.zip(labels)
      |> Enum.reduce({cur_model, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
        {x, tar}, {cur_model, avg_loss, avg_accuracy} ->
          update_with_averages(
            cur_model,
            x,
            tar,
            avg_loss,
            avg_accuracy,
            total_batches,
            learning_rate
          )
      end)

      epoch_avg_loss =
        epoch_avg_loss
        |> Nx.backend_transfer()
        |> Nx.to_number()

      epoch_avg_acc =
        epoch_avg_acc
        |> Nx.backend_transfer()
        |> Nx.to_number()

      {new_model, epoch_avg_loss, epoch_avg_acc}
  end

  def train(x, labels, model, opts \\ []) do
    epochs = opts[:epochs] || 5
    learning_rate = opts[:learning_rate] || 0.01

    for epoch <- 1..epochs, reduce: model do
      cur_model ->
        {time, {new_model, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [cur_model, x, labels, learning_rate])

        {_, _, f1} = run_test(cur_model, x, labels)
        f1 = Nx.to_number(f1)

        if rem(epoch, 10) == 0 do
          IO.puts(
            "Epoch #{epoch} Time: #{time / 1_000_000}s, loss: #{Float.round(epoch_avg_loss, 3)}, acc: #{Float.round(epoch_avg_acc, 3)}, macro-f1: #{Float.round(f1, 3)}"
          )
        end

        new_model
    end
  end

  def diff(%T{weights: weights_a}, %T{weights: weights_b}) do
    Tuple.to_list(weights_a)
    |> Enum.zip(Tuple.to_list(weights_b))
    |> Enum.map(fn {param_a, param_b} ->
      Nx.subtract(param_b, param_a)
    end)
    |> List.to_tuple()
  end

  def run_test(%T{num_classes: num_classes} = model, x, labels) do
    {true_label, predicted} =
      x
      |> Enum.zip(labels)
      |> Enum.reduce({[], []}, fn
        {x, target}, {true_label, predicted} ->
          pred = Nx.argmax(predict(model, x), axis: :output)
          {[Nx.argmax(target, axis: :output) | true_label], [pred | predicted]}
      end)

    y_pred =
      predicted
      |> Enum.reverse()
      |> Nx.concatenate()

    y_true =
      true_label
      |> Enum.reverse()
      |> Nx.concatenate()

    f1 = do_f1_score(y_true, y_pred, num_classes: num_classes, average: :macro)
    {y_true, y_pred, f1}
  end
end
