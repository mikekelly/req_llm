defmodule ReqLLM.Streaming.WebSocketManager do
  @moduledoc """
  GenServer managing a persistent WebSocket connection to the OpenAI Responses API.

  Supports full-duplex streaming over `wss://api.openai.com/v1/responses` using
  `Mint.WebSocket`. Events received from the server are forwarded to a `StreamServer`
  via `StreamServer.http_event/2`, wrapped in SSE format so the existing SSE parsing
  pipeline handles them transparently.

  ## Warm-up

  Call `warm_up/2` before the first real request to pre-negotiate the WebSocket
  connection and obtain a `previous_response_id`. The warm-up sends a
  `response.create` with `generate: false`, which causes the server to return a
  completed response immediately (no content generated). The response ID is stored
  and used as `previous_response_id` in subsequent requests to chain conversation
  context.

  ## Connection flow

  1. `Mint.HTTP.connect/4` — opens TCP/TLS connection (HTTP/1 for WS upgrade)
  2. `Mint.WebSocket.upgrade/5` — issues HTTP upgrade request
  3. `handle_info` receives TCP messages → `Mint.WebSocket.stream/2` accumulates
     `:status`, `:headers`, `:done` responses on the upgrade ref
  4. On `:done` → `Mint.WebSocket.new/4` promotes to WebSocket, enabling send/receive

  ## Event forwarding

  Each JSON text frame from the server is wrapped as:

      "event: <type>\\ndata: <json>\\n\\n"

  and forwarded to the StreamServer as `{:data, sse_chunk}`. This lets the existing
  SSE + provider decode pipeline handle WebSocket events identically to HTTP SSE.

  ## Timeout handling

  After sending a request, an idle timer starts. If `response.completed` does not
  arrive within the timeout, the connection is treated as an error. The timeout
  defaults to 30 s, or 300 s when `thinking` is enabled in the request body.
  """

  use GenServer

  alias ReqLLM.StreamServer

  require Logger
  require Mint.HTTP

  @default_idle_timeout_ms 30_000
  @thinking_idle_timeout_ms 300_000

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc """
  Start the WebSocketManager.

  ## Options

    * `:base_url` - Base URL of the OpenAI-compatible API, e.g. `"https://api.openai.com"`.
      The scheme is replaced with `wss://` for the WebSocket connection.
    * `:api_key` - Bearer token for authentication (required).
    * `:stream_server_pid` - PID of the `StreamServer` to forward events to (required).

  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  @doc """
  Start a WebSocketManager without linking to the calling process.

  Useful when the caller wants the WS manager to fail independently (e.g.,
  in the SSE-fallback path where a WS failure should not crash the caller).
  """
  @spec start(keyword()) :: GenServer.on_start()
  def start(opts) do
    GenServer.start(__MODULE__, opts)
  end

  @doc """
  Block until the WebSocket upgrade completes and the manager is ready to
  send requests. Returns `:ok` on success or `{:error, reason}` on failure.

  If the manager is already connected, returns `:ok` immediately.
  """
  @spec await_connected(pid(), timeout()) :: :ok | {:error, term()}
  def await_connected(pid, timeout \\ 10_000) do
    GenServer.call(pid, :await_connected, timeout)
  end

  @doc """
  Send a `response.create` request over the open WebSocket connection.

  `body` is the raw request map (model, input, tools, etc.). It is merged with
  `%{"type" => "response.create"}` before serialisation.

  Returns `:ok` immediately — events arrive asynchronously via the StreamServer.
  """
  @spec send_request(pid(), map()) :: :ok | {:error, term()}
  def send_request(pid, body) do
    GenServer.call(pid, {:send_request, body})
  end

  @doc """
  Perform a warm-up round-trip: send `response.create` with `generate: false`,
  block until `response.completed`, and return the response ID.

  The warm-up events are **not** forwarded to the StreamServer.

  `body` should include at least: `model`, `tools`, `instructions`. No `input`
  field is required (an empty list is used).
  """
  @spec warm_up(pid(), map()) :: {:ok, String.t()} | {:error, term()}
  def warm_up(pid, body) do
    GenServer.call(pid, {:warm_up, body}, 60_000)
  end

  @doc """
  Close the WebSocket connection and stop the GenServer.
  """
  @spec close(pid()) :: :ok
  def close(pid) do
    GenServer.call(pid, :close)
  end

  @doc """
  Return the `response_id` from the most recently completed response, or `nil`.

  Used to chain conversation turns via the `previous_response_id` field.
  """
  @spec previous_response_id(pid()) :: String.t() | nil
  def previous_response_id(pid) do
    GenServer.call(pid, :previous_response_id)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl GenServer
  def init(opts) do
    base_url = Keyword.fetch!(opts, :base_url)
    api_key = Keyword.fetch!(opts, :api_key)
    stream_server_pid = Keyword.fetch!(opts, :stream_server_pid)

    {scheme, host, port, path} = parse_url(base_url)

    state = %{
      # Connection details
      scheme: scheme,
      host: host,
      port: port,
      path: path,
      api_key: api_key,
      # Mint state
      conn: nil,
      websocket: nil,
      request_ref: nil,
      upgrade_status: nil,
      upgrade_headers: [],
      # Session state
      stream_server_pid: stream_server_pid,
      previous_response_id: nil,
      status: :connecting,
      # Warm-up caller (GenServer.call ref parked here until response.completed arrives)
      warm_up_caller: nil,
      # Idle timeout timer reference
      idle_timer: nil,
      # Callers parked by await_connected/2
      connect_callers: []
    }

    # Initiate connection asynchronously (send message to self)
    send(self(), :connect)

    {:ok, state}
  end

  @impl GenServer
  def handle_call(:await_connected, _from, %{status: :connected} = state) do
    {:reply, :ok, state}
  end

  def handle_call(:await_connected, from, %{status: :connecting} = state) do
    {:noreply, %{state | connect_callers: [from | state.connect_callers]}}
  end

  def handle_call(:await_connected, _from, %{status: status} = state) do
    {:reply, {:error, status}, state}
  end

  @impl GenServer
  def handle_call({:send_request, body}, _from, %{status: :connected} = state) do
    frame_map = Map.put(body, "type", "response.create")

    case send_text_frame(state, frame_map) do
      {:ok, new_state} ->
        new_state = %{new_state | status: :streaming}
        new_state = start_idle_timer(new_state, body)
        {:reply, :ok, new_state}

      {:error, reason} = err ->
        {:reply, err, %{state | status: {:error, reason}}}
    end
  end

  def handle_call({:send_request, _body}, _from, %{status: status} = state) do
    {:reply, {:error, {:not_connected, status}}, state}
  end

  @impl GenServer
  def handle_call({:warm_up, body}, from, %{status: :connected} = state) do
    warm_up_body =
      body
      |> Map.put("type", "response.create")
      |> Map.put("generate", false)
      |> Map.put("store", false)
      |> Map.put_new("input", [])

    case send_text_frame(state, warm_up_body) do
      {:ok, new_state} ->
        # Park the caller; we'll reply when response.completed arrives
        new_state = %{new_state | status: :streaming, warm_up_caller: from}
        {:noreply, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:warm_up, _body}, _from, %{status: status} = state) do
    {:reply, {:error, {:not_connected, status}}, state}
  end

  @impl GenServer
  def handle_call(:close, _from, state) do
    new_state = do_close(state)
    {:stop, :normal, :ok, new_state}
  end

  @impl GenServer
  def handle_call(:previous_response_id, _from, state) do
    {:reply, state.previous_response_id, state}
  end

  # ---------------------------------------------------------------------------
  # handle_info — connection setup
  # ---------------------------------------------------------------------------

  @impl GenServer
  def handle_info(:connect, state) do
    case connect(state) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:error, reason} ->
        Logger.error("WebSocketManager: connection failed: #{inspect(reason)}")
        Enum.each(state.connect_callers, &GenServer.reply(&1, {:error, :connection_failed}))
        {:stop, {:error, :connection_failed}, %{state | status: {:error, :connection_failed}, connect_callers: []}}
    end
  end

  # ---------------------------------------------------------------------------
  # handle_info — Mint TCP/TLS messages
  # ---------------------------------------------------------------------------

  @impl GenServer
  def handle_info(message, %{conn: conn} = state)
      when Mint.HTTP.is_connection_message(conn, message) do
    case Mint.WebSocket.stream(conn, message) do
      {:ok, new_conn, responses} ->
        new_state = %{state | conn: new_conn}
        handle_mint_responses(responses, new_state)

      {:error, new_conn, reason, _responses} ->
        Logger.error("WebSocketManager: Mint stream error: #{inspect(reason)}")
        new_state = %{state | conn: new_conn, status: {:error, reason}}
        forward_error(new_state, reason)
        {:noreply, new_state}

      :unknown ->
        {:noreply, state}
    end
  end

  # ---------------------------------------------------------------------------
  # handle_info — idle timeout
  # ---------------------------------------------------------------------------

  @impl GenServer
  def handle_info(:idle_timeout, state) do
    Logger.warning("WebSocketManager: idle timeout reached")
    reason = :idle_timeout

    if state.warm_up_caller do
      GenServer.reply(state.warm_up_caller, {:error, reason})
    else
      forward_error(state, reason)
    end

    {:noreply, %{state | status: {:error, reason}, warm_up_caller: nil, idle_timer: nil}}
  end

  # ---------------------------------------------------------------------------
  # Private: connection setup
  # ---------------------------------------------------------------------------

  defp connect(%{scheme: scheme, host: host, port: port, path: path, api_key: api_key} = state) do
    mint_scheme = mint_scheme(scheme)

    with {:ok, conn} <-
           Mint.HTTP.connect(mint_scheme, host, port, protocols: [:http1]),
         {:ok, conn, ref} <-
           Mint.WebSocket.upgrade(:wss, conn, path, auth_headers(api_key),
             extensions: [Mint.WebSocket.PerMessageDeflate]
           ) do
      {:ok, %{state | conn: conn, request_ref: ref, status: :connecting}}
    else
      {:error, reason} ->
        {:error, reason}

      {:error, _conn, reason} ->
        {:error, reason}
    end
  end

  defp mint_scheme("https"), do: :https
  defp mint_scheme("http"), do: :http
  defp mint_scheme(other), do: String.to_atom(other)

  defp auth_headers(api_key) do
    [
      {"authorization", "Bearer #{api_key}"},
      {"openai-beta", "responses-websocket=v1"}
    ]
  end

  # ---------------------------------------------------------------------------
  # Private: Mint response accumulation and WebSocket promotion
  # ---------------------------------------------------------------------------

  defp handle_mint_responses([], state), do: {:noreply, state}

  defp handle_mint_responses([response | rest], state) do
    case handle_mint_response(response, state) do
      {:ok, new_state} -> handle_mint_responses(rest, new_state)
      {:stop, reason, new_state} -> {:stop, reason, new_state}
    end
  end

  defp handle_mint_response({:status, ref, status}, %{request_ref: ref} = state) do
    {:ok, %{state | upgrade_status: status}}
  end

  defp handle_mint_response({:headers, ref, headers}, %{request_ref: ref} = state) do
    {:ok, %{state | upgrade_headers: state.upgrade_headers ++ headers}}
  end

  defp handle_mint_response({:done, ref}, %{request_ref: ref} = state) do
    status = state.upgrade_status

    if status == 101 do
      case Mint.WebSocket.new(state.conn, ref, status, state.upgrade_headers) do
        {:ok, conn, websocket} ->
          Logger.debug("WebSocketManager: WebSocket upgrade complete")
          new_state = %{state | conn: conn, websocket: websocket, status: :connected}
          Enum.each(new_state.connect_callers, &GenServer.reply(&1, :ok))
          {:ok, %{new_state | connect_callers: []}}

        {:error, reason} ->
          Logger.error("WebSocketManager: WebSocket promotion failed: #{inspect(reason)}")
          {:stop, {:error, {:upgrade_failed, reason}},
           %{state | status: {:error, {:upgrade_failed, reason}}}}
      end
    else
      Logger.error("WebSocketManager: upgrade rejected with status #{status}")
      error = {:error, {:upgrade_failed, status}}
      Enum.each(state.connect_callers, &GenServer.reply(&1, error))
      {:stop, error,
       %{state | status: {:error, {:upgrade_failed, status}}, connect_callers: []}}
    end
  end

  defp handle_mint_response({:data, ref, _data}, %{request_ref: ref, websocket: nil} = state) do
    # Data received before upgrade completed (e.g., HTTP error body). Ignore —
    # the :done handler will stop the process with the non-101 status.
    {:ok, state}
  end

  defp handle_mint_response({:data, ref, data}, %{request_ref: ref} = state) do
    case Mint.WebSocket.decode(state.websocket, data) do
      {:ok, websocket, frames} ->
        new_state = %{state | websocket: websocket}
        handle_ws_frames(frames, new_state)

      {:error, websocket, reason} ->
        Logger.error("WebSocketManager: WebSocket decode error: #{inspect(reason)}")
        {:ok, %{state | websocket: websocket}}
    end
  end

  defp handle_mint_response({:error, _ref, reason}, state) do
    Logger.error("WebSocketManager: Mint error response: #{inspect(reason)}")
    forward_error(state, reason)
    {:ok, %{state | status: {:error, reason}}}
  end

  defp handle_mint_response(_other, state) do
    {:ok, state}
  end

  # ---------------------------------------------------------------------------
  # Private: WebSocket frame handling
  # ---------------------------------------------------------------------------

  defp handle_ws_frames([], state), do: {:ok, state}

  defp handle_ws_frames([frame | rest], state) do
    case handle_ws_frame(frame, state) do
      {:ok, new_state} -> handle_ws_frames(rest, new_state)
      {:stop, reason, new_state} -> {:stop, reason, new_state}
    end
  end

  defp handle_ws_frame({:ping, data}, state) do
    # Respond to server pings with a pong
    case send_frame(state, {:pong, data}) do
      {:ok, new_state} -> {:ok, new_state}
      {:error, _reason} -> {:ok, state}
    end
  end

  defp handle_ws_frame({:pong, _data}, state) do
    # Ignore pongs
    {:ok, state}
  end

  defp handle_ws_frame({:close, _code, _reason}, state) do
    Logger.debug("WebSocketManager: received close frame")
    new_state = %{state | status: :closed}

    if state.warm_up_caller do
      GenServer.reply(state.warm_up_caller, {:error, :connection_closed})
      {:stop, :normal, %{new_state | warm_up_caller: nil}}
    else
      forward_error(new_state, :connection_closed)
      {:stop, :normal, new_state}
    end
  end

  defp handle_ws_frame({:text, json_text}, state) do
    case Jason.decode(json_text) do
      {:ok, event} ->
        handle_ws_event(event, json_text, state)

      {:error, reason} ->
        Logger.warning("WebSocketManager: failed to decode JSON frame: #{inspect(reason)}")
        {:ok, state}
    end
  end

  defp handle_ws_frame({:binary, _data}, state) do
    # Binary frames are not expected from the Responses API
    {:ok, state}
  end

  # ---------------------------------------------------------------------------
  # Private: WebSocket event dispatch
  # ---------------------------------------------------------------------------

  defp handle_ws_event(%{"type" => "response.completed", "response" => response} = event,
         json_text,
         state
       ) do
    response_id = Map.get(response, "id")

    new_state =
      state
      |> cancel_idle_timer()
      |> Map.put(:previous_response_id, response_id)
      |> Map.put(:status, :connected)

    if state.warm_up_caller do
      # Warm-up mode: reply with the response_id, don't forward to StreamServer
      GenServer.reply(state.warm_up_caller, {:ok, response_id})
      {:ok, %{new_state | warm_up_caller: nil}}
    else
      # Normal streaming mode: forward to StreamServer and signal done
      forward_event(new_state, event, json_text)
      StreamServer.http_event(new_state.stream_server_pid, :done)
      {:ok, new_state}
    end
  end

  defp handle_ws_event(
         %{"type" => "error", "error" => %{"type" => "websocket_connection_limit_reached"}} =
           _event,
         _json_text,
         state
       ) do
    Logger.warning("WebSocketManager: connection limit reached")
    reason = :websocket_connection_limit_reached
    new_state = cancel_idle_timer(state)
    forward_error(new_state, reason)
    {:ok, %{new_state | status: {:error, reason}}}
  end

  defp handle_ws_event(%{"type" => type} = event, json_text, state)
       when not is_nil(type) do
    # Warm-up mode: suppress events, just accumulate
    unless state.warm_up_caller do
      forward_event(state, event, json_text)
    end

    {:ok, state}
  end

  defp handle_ws_event(_event, _json_text, state) do
    {:ok, state}
  end

  # ---------------------------------------------------------------------------
  # Private: event forwarding helpers
  # ---------------------------------------------------------------------------

  # Forward a single event to StreamServer wrapped as an SSE data chunk.
  # This allows the existing SSE parsing pipeline (StreamServer → provider decoder)
  # to handle WebSocket events identically to HTTP SSE events.
  defp forward_event(%{stream_server_pid: pid, warm_up_caller: nil}, event, json_text) do
    type = Map.get(event, "type", "unknown")
    sse_chunk = "event: #{type}\ndata: #{json_text}\n\n"
    StreamServer.http_event(pid, {:data, sse_chunk})
  end

  defp forward_event(_state, _event, _json_text), do: :ok

  defp forward_error(%{stream_server_pid: pid, warm_up_caller: nil}, reason) do
    StreamServer.http_event(pid, {:error, reason})
    StreamServer.http_event(pid, :done)
  end

  defp forward_error(_state, _reason), do: :ok

  # ---------------------------------------------------------------------------
  # Private: sending frames
  # ---------------------------------------------------------------------------

  defp send_text_frame(state, map) do
    case Jason.encode(map) do
      {:ok, json} -> send_frame(state, {:text, json})
      {:error, reason} -> {:error, {:json_encode_failed, reason}}
    end
  end

  defp send_frame(%{conn: conn, websocket: websocket, request_ref: ref} = state, frame) do
    with {:ok, websocket, data} <- Mint.WebSocket.encode(websocket, frame),
         {:ok, conn} <- Mint.WebSocket.stream_request_body(conn, ref, data) do
      {:ok, %{state | conn: conn, websocket: websocket}}
    else
      # Mint.WebSocket.encode returns {error, websocket, reason} on failure
      {:error, %Mint.WebSocket{} = _websocket, reason} ->
        {:error, reason}

      # Mint.WebSocket.stream_request_body returns {:error, conn, reason} on failure
      {:error, _conn, reason} ->
        {:error, reason}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ---------------------------------------------------------------------------
  # Private: idle timeout management
  # ---------------------------------------------------------------------------

  defp start_idle_timer(state, body) do
    state = cancel_idle_timer(state)

    timeout_ms =
      if has_thinking?(body) do
        @thinking_idle_timeout_ms
      else
        @default_idle_timeout_ms
      end

    timer_ref = Process.send_after(self(), :idle_timeout, timeout_ms)
    %{state | idle_timer: timer_ref}
  end

  defp cancel_idle_timer(%{idle_timer: nil} = state), do: state

  defp cancel_idle_timer(%{idle_timer: ref} = state) do
    Process.cancel_timer(ref)
    %{state | idle_timer: nil}
  end

  defp has_thinking?(%{"thinking" => %{"type" => "enabled"}}), do: true
  defp has_thinking?(_), do: false

  # ---------------------------------------------------------------------------
  # Private: connection teardown
  # ---------------------------------------------------------------------------

  defp do_close(state) do
    state = cancel_idle_timer(state)

    if state.conn do
      try do
        if state.websocket && state.request_ref do
          case Mint.WebSocket.encode(state.websocket, :close) do
            {:ok, _ws, data} ->
              Mint.WebSocket.stream_request_body(state.conn, state.request_ref, data)

            _ ->
              :ok
          end
        end

        Mint.HTTP.close(state.conn)
      rescue
        _ -> :ok
      end
    end

    %{state | status: :closed, conn: nil, websocket: nil}
  end

  # ---------------------------------------------------------------------------
  # Private: URL parsing
  # ---------------------------------------------------------------------------

  @doc false
  def parse_url(url) do
    uri = URI.parse(url)
    scheme = uri.scheme || "https"
    host = uri.host || "api.openai.com"
    port = uri.port || default_port(scheme)
    # Use the path from the URL, defaulting to the Responses API endpoint
    path =
      case uri.path do
        nil -> "/v1/responses"
        "" -> "/v1/responses"
        p -> p
      end

    {scheme, host, port, path}
  end

  defp default_port("https"), do: 443
  defp default_port("http"), do: 80
  defp default_port(_), do: 443
end
