defmodule ReqLLM.StreamingErrorEmissionTest do
  @moduledoc """
  Tests that create_lazy_stream emits {:error, reason} as a stream element
  when StreamServer.next/2 returns an error, rather than silently halting.

  This ensures consumers can see rich error information (status codes, reason text)
  instead of inferring failure from empty streams.
  """

  use ExUnit.Case, async: true

  alias ReqLLM.StreamServer

  @moduletag category: :streaming

  describe "create_lazy_stream error emission" do
    test "error from StreamServer.next is emitted as stream element before halting" do
      # Start a real StreamServer
      {:ok, server_pid} =
        StreamServer.start_link(
          provider_mod: ReqLLM.Providers.OpenAI,
          model: %LLMDB.Model{provider: :openai, id: "test"}
        )

      # Inject an error event into the stream server
      StreamServer.http_event(server_pid, {:error, :test_api_error})

      # Create a lazy stream using the same logic as Streaming.create_lazy_stream/2
      stream =
        Stream.resource(
          fn -> server_pid end,
          fn
            :halted ->
              {:halt, :halted}

            server ->
              case StreamServer.next(server, 1000) do
                {:ok, chunk} ->
                  {[chunk], server}

                :halt ->
                  {:halt, server}

                {:error, reason} ->
                  {[{:error, reason}], :halted}
              end
          end,
          fn _server -> :ok end
        )

      # Consume the stream - should get the error as a stream element
      elements = Enum.to_list(stream)

      assert [{:error, :test_api_error}] = elements
    end

    test "error with rich API Request struct is emitted as stream element" do
      {:ok, server_pid} =
        StreamServer.start_link(
          provider_mod: ReqLLM.Providers.OpenAI,
          model: %LLMDB.Model{provider: :openai, id: "test"}
        )

      # Inject an error with a rich struct
      error = ReqLLM.Error.API.Request.exception(
        reason: "No tool call found in response",
        status: 400
      )

      StreamServer.http_event(server_pid, {:error, error})

      stream =
        Stream.resource(
          fn -> server_pid end,
          fn
            :halted ->
              {:halt, :halted}

            server ->
              case StreamServer.next(server, 1000) do
                {:ok, chunk} ->
                  {[chunk], server}

                :halt ->
                  {:halt, server}

                {:error, reason} ->
                  {[{:error, reason}], :halted}
              end
          end,
          fn _server -> :ok end
        )

      elements = Enum.to_list(stream)

      assert [{:error, ^error}] = elements
    end
  end
end
