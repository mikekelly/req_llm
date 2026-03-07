defmodule ReqLLM.Providers.OpenAI.Delta do
  @moduledoc """
  Computes incremental request deltas for the OpenAI Responses API.

  When using `previous_response_id`, only new input items need to be sent
  if the non-input fields (model, tools, instructions, etc.) haven't changed.
  This dramatically reduces payload sizes on multi-turn conversations.
  """

  @non_input_fields ~w(model instructions tools tool_choice reasoning store service_tier max_output_tokens)

  @doc """
  Compute whether the current request can be sent as an incremental delta
  using `previous_response_id`, or must be sent as a full request.

  Returns `{:delta, delta_body}` if only new items need to be sent,
  or `{:full, full_body}` if the complete request must be sent.
  """
  @spec compute(map(), map() | nil, String.t() | nil) :: {:delta, map()} | {:full, map()}
  def compute(current_body, _previous_body, nil), do: {:full, current_body}
  def compute(current_body, nil, _previous_response_id), do: {:full, current_body}

  def compute(current_body, previous_body, previous_response_id) do
    if fields_match?(current_body, previous_body) and input_extends?(current_body, previous_body) do
      previous_input_length = length(previous_body["input"] || [])
      new_items = Enum.drop(current_body["input"] || [], previous_input_length)

      delta_body = %{
        "model" => current_body["model"],
        "previous_response_id" => previous_response_id,
        "input" => new_items
      }

      {:delta, delta_body}
    else
      {:full, current_body}
    end
  end

  defp fields_match?(current, previous) do
    Enum.all?(@non_input_fields, fn field ->
      current[field] == previous[field]
    end)
  end

  defp input_extends?(current, previous) do
    current_input = current["input"] || []
    previous_input = previous["input"] || []

    length(current_input) >= length(previous_input) and
      Enum.zip(previous_input, current_input)
      |> Enum.all?(fn {prev, curr} -> prev == curr end)
  end
end
