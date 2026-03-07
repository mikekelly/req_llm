defmodule ReqLLM.Providers.OpenAI.CompactTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "decode_compact_items_to_messages/1" do
    test "user message with content list extracts text" do
      items = [
        %{
          "role" => "user",
          "content" => [
            %{"type" => "input_text", "text" => "Hello there"}
          ]
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [%{"role" => "user", "content" => "Hello there"}]
    end

    test "user message with binary content passes through" do
      items = [
        %{"role" => "user", "content" => "Simple string content"}
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [%{"role" => "user", "content" => "Simple string content"}]
    end

    test "assistant message with content list extracts text" do
      items = [
        %{
          "role" => "assistant",
          "content" => [
            %{"type" => "output_text", "text" => "I can help you"}
          ]
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [%{"role" => "assistant", "content" => "I can help you"}]
    end

    test "assistant message with text type content extracts text" do
      items = [
        %{
          "role" => "assistant",
          "content" => [
            %{"type" => "text", "text" => "Here is my reply"}
          ]
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [%{"role" => "assistant", "content" => "Here is my reply"}]
    end

    test "function_call item produces assistant message with tool_calls" do
      items = [
        %{
          "type" => "function_call",
          "call_id" => "call_abc123",
          "name" => "get_weather",
          "arguments" => "{\"city\": \"London\"}"
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert [message] = result
      assert message["role"] == "assistant"
      assert message["content"] == ""
      assert [tool_call] = message["tool_calls"]
      assert tool_call["id"] == "call_abc123"
      assert tool_call["type"] == "function"
      assert tool_call["function"]["name"] == "get_weather"
      assert tool_call["function"]["arguments"] == "{\"city\": \"London\"}"
    end

    test "function_call_output item produces tool message" do
      items = [
        %{
          "type" => "function_call_output",
          "call_id" => "call_abc123",
          "output" => "Sunny, 22°C"
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [
               %{
                 "role" => "tool",
                 "tool_call_id" => "call_abc123",
                 "content" => "Sunny, 22°C"
               }
             ]
    end

    test "reasoning item is skipped" do
      items = [
        %{"type" => "reasoning", "encrypted_content" => "some_encrypted_data"}
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == []
    end

    test "unknown item is skipped" do
      items = [
        %{"type" => "unknown_future_type", "data" => "something"}
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == []
    end

    test "mixed list converts all items correctly" do
      items = [
        %{
          "role" => "user",
          "content" => [%{"type" => "input_text", "text" => "What's the weather?"}]
        },
        %{
          "type" => "function_call",
          "call_id" => "call_xyz",
          "name" => "get_weather",
          "arguments" => "{\"city\": \"Paris\"}"
        },
        %{
          "type" => "function_call_output",
          "call_id" => "call_xyz",
          "output" => "Rainy, 15°C"
        },
        %{
          "role" => "assistant",
          "content" => [%{"type" => "output_text", "text" => "It's rainy in Paris."}]
        },
        %{"type" => "reasoning", "encrypted_content" => "encrypted_blob"}
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert length(result) == 4

      assert Enum.at(result, 0) == %{"role" => "user", "content" => "What's the weather?"}

      tool_call_msg = Enum.at(result, 1)
      assert tool_call_msg["role"] == "assistant"
      assert [tc] = tool_call_msg["tool_calls"]
      assert tc["function"]["name"] == "get_weather"

      assert Enum.at(result, 2) == %{
               "role" => "tool",
               "tool_call_id" => "call_xyz",
               "content" => "Rainy, 15°C"
             }

      assert Enum.at(result, 3) == %{"role" => "assistant", "content" => "It's rainy in Paris."}
    end

    test "multiple text content parts are joined" do
      items = [
        %{
          "role" => "user",
          "content" => [
            %{"type" => "input_text", "text" => "Hello "},
            %{"type" => "input_text", "text" => "world"}
          ]
        }
      ]

      result = ResponsesAPI.decode_compact_items_to_messages(items)

      assert result == [%{"role" => "user", "content" => "Hello world"}]
    end

    test "empty list returns empty list" do
      assert ResponsesAPI.decode_compact_items_to_messages([]) == []
    end
  end

  describe "encode_messages_to_input/1" do
    test "converts ReqLLM.Message structs to Responses API format" do
      messages = [
        %ReqLLM.Message{
          role: :user,
          content: [%ReqLLM.Message.ContentPart{type: :text, text: "Hello"}]
        }
      ]

      result = ResponsesAPI.encode_messages_to_input(messages)

      assert [%{"role" => "user", "content" => [%{"type" => "input_text", "text" => "Hello"}]}] =
               result
    end

    test "converts assistant message with tool calls" do
      messages = [
        %ReqLLM.Message{
          role: :assistant,
          content: [],
          tool_calls: [
            ReqLLM.ToolCall.new("call_123", "my_tool", "{}")
          ]
        }
      ]

      result = ResponsesAPI.encode_messages_to_input(messages)

      assert [
               %{
                 "type" => "function_call",
                 "call_id" => "call_123",
                 "name" => "my_tool",
                 "arguments" => "{}"
               }
             ] = result
    end
  end
end
