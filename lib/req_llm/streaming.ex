defmodule ReqLLM.Streaming do
  @moduledoc """
  Main orchestration for ReqLLM streaming operations.

  This module coordinates StreamServer, FinchClient, and StreamResponse to provide
  a cohesive streaming system. It serves as the entry point for all streaming
  operations and handles the complex coordination between components.

  ## Architecture

  The streaming system consists of three main components:

  - `StreamServer` - GenServer managing stream state and event processing
  - `FinchClient` - HTTP transport layer using Finch for streaming requests
  - `StreamResponse` - User-facing API providing streams and metadata tasks

  ## Transport modes

  The `:transport` option controls how streaming connects to the provider:

  - `:auto` (default) - Uses WebSocket for OpenAI providers, SSE for others.
    Falls back to SSE if the WS connection fails.
  - `:websocket` - Forces WebSocket transport. Returns `{:error, _}` on failure.
  - `:sse` - Forces SSE transport (the original HTTP streaming path).

  ## Flow

  1. `start_stream/4` creates StreamServer with provider configuration
  2. Transport layer (WS or Finch/SSE) connects and starts streaming
  3. Events are forwarded to StreamServer for processing
  4. StreamResponse provides lazy stream using `Stream.resource/3`
  5. Metadata task runs concurrently to collect usage and finish_reason
  6. Cancel function provides cleanup of all components

  ## Example

      {:ok, stream_response} = ReqLLM.Streaming.start_stream(
        ReqLLM.Providers.Anthropic,
        %LLMDB.Model{provider: :anthropic, name: "claude-3-sonnet"},
        ReqLLM.Context.new("Hello!"),
        []
      )

      # Stream tokens
      stream_response.stream
      |> ReqLLM.StreamResponse.tokens()
      |> Stream.each(&IO.write/1)
      |> Stream.run()

      # Get metadata
      usage = ReqLLM.StreamResponse.usage(stream_response)

  """

  alias ReqLLM.{Context, StreamResponse, StreamResponse.MetadataHandle, StreamServer}
  alias ReqLLM.Streaming.WebSocketManager

  require Logger

  @doc """
  Start a streaming session with coordinated StreamServer and transport.

  ## Options

    * `:transport` - Transport mode: `:auto` (default), `:websocket`, or `:sse`
    * `:timeout` - HTTP request timeout in milliseconds (default: 30_000)
    * `:metadata_timeout` - Metadata collection timeout (default: 300_000)
    * `:fixture_path` - Path for test fixture capture (testing only)
    * `:finch_name` - Finch pool name (default: ReqLLM.Finch)
    * `:provider_options` - Provider-specific options (e.g., `previous_response_id`)

  """
  @spec start_stream(module(), LLMDB.Model.t(), Context.t(), keyword()) ::
          {:ok, StreamResponse.t()} | {:error, term()}
  def start_stream(provider_mod, model, context, opts \\ []) do
    transport = Keyword.get(opts, :transport, :auto)

    case transport do
      :sse ->
        start_stream_sse(provider_mod, model, context, opts)

      :websocket ->
        start_stream_ws(provider_mod, model, context, opts)

      :auto ->
        if ws_eligible?(provider_mod, opts) do
          try do
            case start_stream_ws(provider_mod, model, context, opts) do
              {:ok, _} = success ->
                success

              {:error, reason} ->
                Logger.warning(
                  "WebSocket streaming failed (#{inspect(reason)}), falling back to SSE"
                )

                start_stream_sse(provider_mod, model, context, opts)
            end
          rescue
            e ->
              Logger.warning(
                "WebSocket streaming raised (#{inspect(e)}), falling back to SSE"
              )

              start_stream_sse(provider_mod, model, context, opts)
          end
        else
          start_stream_sse(provider_mod, model, context, opts)
        end
    end
  end

  # Only OpenAI supports WebSocket streaming via the Responses API.
  # WS always connects to api.openai.com regardless of the HTTP base_url
  # (Codex OAuth models use chatgpt.com for SSE but api.openai.com for WS).
  defp ws_eligible?(provider_mod, _opts) do
    provider_mod == ReqLLM.Providers.OpenAI
  end

  @doc false
  def build_ws_url(base_url) do
    uri = URI.parse(base_url)

    path =
      case uri.path do
        nil -> "/v1/responses"
        "" -> "/v1/responses"
        p -> if String.ends_with?(p, "/responses"), do: p, else: p <> "/responses"
      end

    URI.to_string(%{uri | path: path})
  end

  # ---------------------------------------------------------------------------
  # SSE transport (original path)
  # ---------------------------------------------------------------------------

  defp start_stream_sse(provider_mod, model, context, opts) do
    with {:ok, server_pid} <- start_stream_server(provider_mod, model, opts),
         {:ok, _http_task_pid, _http_context, _canonical_json} <-
           start_http_streaming(provider_mod, model, context, opts, server_pid) do
      build_stream_response(server_pid, model, context, opts)
    else
      {:error, reason} ->
        Logger.error("Failed to start streaming: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # ---------------------------------------------------------------------------
  # WebSocket transport
  # ---------------------------------------------------------------------------

  defp start_stream_ws(provider_mod, model, context, opts) do
    with {:ok, server_pid} <- start_stream_server(provider_mod, model, opts),
         {:ok, ws_pid} <- start_ws_manager(model, opts, server_pid),
         :ok <- send_ws_request(provider_mod, model, context, opts, ws_pid, server_pid) do
      build_stream_response(server_pid, model, context, opts)
    else
      {:error, reason} ->
        Logger.error("Failed to start WebSocket streaming: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp start_ws_manager(_model, opts, server_pid) do
    # WS always connects to api.openai.com, even for Codex OAuth models.
    # The Codex CLI does the same: WS goes to api.openai.com, SSE goes to
    # chatgpt.com/backend-api/codex. The OAuth token works for both.
    ws_url = build_ws_url("https://api.openai.com")

    # Get API key from opts (Him.LLM.Client puts the OAuth token here)
    # or fall back to standard key resolution for non-Codex usage
    api_key = Keyword.get(opts, :api_key) || ReqLLM.Keys.get!(:openai)

    case WebSocketManager.start(
           base_url: ws_url,
           api_key: api_key,
           stream_server_pid: server_pid
         ) do
      {:ok, pid} ->
        case WebSocketManager.await_connected(pid, 10_000) do
          :ok -> {:ok, pid}
          {:error, reason} -> {:error, {:ws_connect_failed, reason}}
        end

      {:error, reason} ->
        {:error, {:ws_start_failed, reason}}
    end
  end

  defp send_ws_request(provider_mod, model, context, opts, ws_pid, server_pid) do
    provider_opts = Keyword.get(opts, :provider_options, [])
    finch_name = Keyword.get(opts, :finch_name, ReqLLM.Finch)

    # Build body by calling attach_stream (which builds a Finch request),
    # then extracting and decoding the JSON body from the Finch struct.
    case provider_mod.attach_stream(model, context, opts, finch_name) do
      {:ok, %Finch.Request{body: body_json}} ->
        body_map = Jason.decode!(body_json)

        # Inject previous_response_id if provided
        body_map =
          case Keyword.get(provider_opts, :previous_response_id) do
            nil -> body_map
            id -> Map.put(body_map, "previous_response_id", id)
          end

        # Attach WS manager PID to StreamServer for lifecycle coupling
        StreamServer.attach_http_task(server_pid, ws_pid)

        case WebSocketManager.send_request(ws_pid, body_map) do
          :ok -> :ok
          {:error, reason} -> {:error, {:ws_send_failed, reason}}
        end

      {:error, reason} ->
        {:error, {:provider_build_failed, reason}}
    end
  end

  # ---------------------------------------------------------------------------
  # Shared: build StreamResponse from a running StreamServer
  # ---------------------------------------------------------------------------

  defp build_stream_response(server_pid, model, context, opts) do
    default_timeout =
      Application.get_env(
        :req_llm,
        :stream_receive_timeout,
        Application.get_env(:req_llm, :receive_timeout, 30_000)
      )

    receive_timeout = Keyword.get(opts, :receive_timeout, default_timeout)
    stream = create_lazy_stream(server_pid, receive_timeout)

    metadata_handle = start_metadata_handle(server_pid, opts)

    cancel_fn = fn -> StreamServer.cancel(server_pid) end

    stream_response = %StreamResponse{
      stream: stream,
      metadata_handle: metadata_handle,
      cancel: cancel_fn,
      model: model,
      context: context
    }

    {:ok, stream_response}
  end

  # ---------------------------------------------------------------------------
  # StreamServer setup
  # ---------------------------------------------------------------------------

  defp start_stream_server(provider_mod, model, opts) do
    server_opts = [
      provider_mod: provider_mod,
      model: model,
      fixture_path: maybe_capture_fixture(model, opts),
      high_watermark: Keyword.get(opts, :high_watermark, 500)
    ]

    genserver_opts = Keyword.take(opts, [:name, :timeout, :debug, :spawn_opt, :hibernate_after])
    all_opts = Keyword.merge(server_opts, genserver_opts)

    case StreamServer.start_link(all_opts) do
      {:ok, pid} ->
        {:ok, pid}

      {:error, reason} ->
        Logger.error("Failed to start StreamServer: #{inspect(reason)}")
        {:error, {:stream_server_failed, reason}}
    end
  end

  # Start HTTP streaming through StreamServer (SSE path)
  defp start_http_streaming(provider_mod, model, context, opts, stream_server_pid) do
    finch_name = Keyword.get(opts, :finch_name, ReqLLM.Finch)

    case StreamServer.start_http(
           stream_server_pid,
           provider_mod,
           model,
           context,
           opts,
           finch_name
         ) do
      {:ok, task_pid, http_context, canonical_json} ->
        {:ok, task_pid, http_context, canonical_json}

      {:error, {:provider_build_failed, {:http2_body_too_large, body_size, protocols}}} ->
        message = format_http2_error_message(body_size, protocols)
        Logger.error(message)
        {:error, {:http2_body_too_large, message}}

      {:error, reason} ->
        Logger.error("Failed to start HTTP streaming: #{inspect(reason)}")
        {:error, {:http_streaming_failed, reason}}
    end
  end

  defp format_http2_error_message(body_size, protocols) do
    size_kb = div(body_size, 1024)

    """
    Request body (#{size_kb}KB) exceeds safe limit for HTTP/2 connections (64KB).

    This is due to a known issue in Finch's HTTP/2 implementation:
    https://github.com/sneako/finch/issues/265

    Your current pool configuration uses: #{inspect(protocols)}

    To fix this, configure ReqLLM to use HTTP/1-only pools (recommended):

        config :req_llm,
          finch: [
            name: ReqLLM.Finch,
            pools: %{
              :default => [protocols: [:http1], size: 1, count: 8]
            }
          ]

    See the ReqLLM README section "HTTP/2 Configuration (Advanced)" for more details:
    https://github.com/agentjido/req_llm#http2-configuration-advanced
    """
  end

  defp maybe_capture_fixture(model, opts) do
    case Code.ensure_loaded(ReqLLM.Test.Fixtures) do
      {:module, mod} -> mod.capture_path(model, opts)
      {:error, _} -> nil
    end
  end

  # Create lazy stream using Stream.resource that calls StreamServer.next/2
  defp create_lazy_stream(server_pid, timeout) do
    Stream.resource(
      # start_fn: return the server pid
      fn -> server_pid end,
      # next_fn: get next chunk from server
      fn
        :halted ->
          {:halt, :halted}

        server ->
          case StreamServer.next(server, timeout) do
            {:ok, chunk} ->
              {[chunk], server}

            :halt ->
              {:halt, server}

            {:error, reason} ->
              Logger.error("Stream error: #{inspect(reason)}")
              {[{:error, reason}], :halted}
          end
      end,
      # after_fn: no-op, cleanup handled by cancel function
      fn _server -> :ok end
    )
  end

  defp start_metadata_handle(server_pid, opts) do
    default_metadata_timeout = Application.get_env(:req_llm, :metadata_timeout, 300_000)
    metadata_timeout = Keyword.get(opts, :metadata_timeout, default_metadata_timeout)

    fetch_fun = fn ->
      case StreamServer.await_metadata(server_pid, metadata_timeout) do
        {:ok, metadata} ->
          metadata

        {:error, reason} ->
          Logger.warning("Metadata collection failed: #{inspect(reason)}")
          %{error: reason}
      end
    end

    case MetadataHandle.start_link(fetch_fun) do
      {:ok, handle} -> handle
      {:error, reason} -> raise "Failed to start metadata handle: #{inspect(reason)}"
    end
  end
end
