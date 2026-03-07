defmodule ReqLLM.Providers.OpenAI.DeltaTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.OpenAI.Delta

  @base_body %{
    "model" => "gpt-5",
    "instructions" => "You are a helpful assistant.",
    "input" => [%{"role" => "user", "content" => [%{"type" => "input_text", "text" => "Hello"}]}],
    "tools" => nil,
    "tool_choice" => nil,
    "reasoning" => nil,
    "store" => false,
    "service_tier" => nil,
    "max_output_tokens" => 1000
  }

  describe "compute/3 with nil previous_response_id" do
    test "returns :full with the current body" do
      current = @base_body
      previous = @base_body

      assert {:full, ^current} = Delta.compute(current, previous, nil)
    end
  end

  describe "compute/3 with nil previous_body" do
    test "returns :full with the current body" do
      current = @base_body

      assert {:full, ^current} = Delta.compute(current, nil, "resp_abc123")
    end
  end

  describe "compute/3 when delta is possible" do
    test "returns :delta with only new input items when fields match and input is extended" do
      previous_input = [%{"role" => "user", "content" => [%{"type" => "input_text", "text" => "Hello"}]}]
      assistant_reply = [%{"role" => "assistant", "content" => [%{"type" => "output_text", "text" => "Hi there!"}]}]
      new_user_item = %{"role" => "user", "content" => [%{"type" => "input_text", "text" => "How are you?"}]}

      previous_body = %{@base_body | "input" => previous_input}
      current_body = %{@base_body | "input" => previous_input ++ assistant_reply ++ [new_user_item]}

      assert {:delta, delta_body} = Delta.compute(current_body, previous_body, "resp_abc123")

      assert delta_body["model"] == "gpt-5"
      assert delta_body["previous_response_id"] == "resp_abc123"
      assert delta_body["input"] == assistant_reply ++ [new_user_item]
      refute Map.has_key?(delta_body, "instructions")
      refute Map.has_key?(delta_body, "tools")
      refute Map.has_key?(delta_body, "tool_choice")
      refute Map.has_key?(delta_body, "reasoning")
      refute Map.has_key?(delta_body, "store")
      refute Map.has_key?(delta_body, "service_tier")
      refute Map.has_key?(delta_body, "max_output_tokens")
    end

    test "returns :delta with empty input when current input equals previous input" do
      previous_body = @base_body
      current_body = @base_body

      assert {:delta, delta_body} = Delta.compute(current_body, previous_body, "resp_abc123")

      assert delta_body["input"] == []
    end
  end

  describe "compute/3 when changed model" do
    test "returns :full" do
      previous_body = @base_body
      current_body = %{@base_body | "model" => "gpt-4o"}

      assert {:full, ^current_body} = Delta.compute(current_body, previous_body, "resp_abc123")
    end
  end

  describe "compute/3 when changed instructions" do
    test "returns :full" do
      previous_body = @base_body
      current_body = %{@base_body | "instructions" => "You are a coding assistant."}

      assert {:full, ^current_body} = Delta.compute(current_body, previous_body, "resp_abc123")
    end
  end

  describe "compute/3 when changed tools" do
    test "returns :full" do
      previous_body = @base_body

      current_body = %{
        @base_body
        | "tools" => [%{"type" => "function", "name" => "get_weather"}]
      }

      assert {:full, ^current_body} = Delta.compute(current_body, previous_body, "resp_abc123")
    end
  end

  describe "compute/3 when input is not an extension of previous" do
    test "returns :full when input items are reordered" do
      item_a = %{"role" => "user", "content" => [%{"type" => "input_text", "text" => "First"}]}
      item_b = %{"role" => "user", "content" => [%{"type" => "input_text", "text" => "Second"}]}

      previous_body = %{@base_body | "input" => [item_a, item_b]}
      current_body = %{@base_body | "input" => [item_b, item_a]}

      assert {:full, ^current_body} = Delta.compute(current_body, previous_body, "resp_abc123")
    end

    test "returns :full when current input is shorter than previous" do
      item_a = %{"role" => "user", "content" => [%{"type" => "input_text", "text" => "First"}]}
      item_b = %{"role" => "user", "content" => [%{"type" => "input_text", "text" => "Second"}]}

      previous_body = %{@base_body | "input" => [item_a, item_b]}
      current_body = %{@base_body | "input" => [item_a]}

      assert {:full, ^current_body} = Delta.compute(current_body, previous_body, "resp_abc123")
    end
  end

  describe "compute/3 with tool call and tool result as new items" do
    test "returns :delta with those items" do
      previous_input = [%{"role" => "user", "content" => [%{"type" => "input_text", "text" => "What's the weather?"}]}]

      tool_call = %{
        "type" => "function_call",
        "call_id" => "call_xyz",
        "name" => "get_weather",
        "arguments" => "{\"city\": \"London\"}"
      }

      tool_result = %{
        "type" => "function_call_output",
        "call_id" => "call_xyz",
        "output" => "Sunny, 22°C"
      }

      previous_body = %{@base_body | "input" => previous_input}
      current_body = %{@base_body | "input" => previous_input ++ [tool_call, tool_result]}

      assert {:delta, delta_body} = Delta.compute(current_body, previous_body, "resp_abc123")

      assert delta_body["input"] == [tool_call, tool_result]
      assert delta_body["previous_response_id"] == "resp_abc123"
    end
  end
end
