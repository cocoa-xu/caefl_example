defmodule CaeflExampleTest do
  use ExUnit.Case
  doctest CaeflExample

  test "greets the world" do
    assert CaeflExample.hello() == :world
  end
end
