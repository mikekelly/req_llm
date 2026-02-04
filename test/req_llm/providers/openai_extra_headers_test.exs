defmodule ReqLLM.Providers.OpenAIExtraHeadersTest do
  use ExUnit.Case, async: false

  alias ReqLLM.Context

  setup do
    ReqLLM.TestSupport.FakeKeys.install!()
    :ok
  end

  describe "OpenAI ChatAPI extra headers from req_http_options" do
    setup do
      {:ok, model} = ReqLLM.model("openai:gpt-4o-mini")
      context = Context.new([Context.user("test")])
      {:ok, model: model, context: context}
    end

    test "extra headers are included in streaming request", %{model: model, context: context} do
      opts = [
        api_key: "test-openai",
        req_http_options: [
          headers: [
            {"chatgpt-account-id", "test-account-123"},
            {"x-custom-header", "custom-value"}
          ]
        ]
      ]

      {:ok, finch_request} =
        ReqLLM.Providers.OpenAI.ChatAPI.attach_stream(model, context, opts, nil)

      # Verify Authorization header exists
      auth_header = Enum.find(finch_request.headers, fn {k, _v} -> k == "Authorization" end)
      assert auth_header == {"Authorization", "Bearer test-openai"}

      # Verify Content-Type header exists
      content_type_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "Content-Type" end)

      assert content_type_header == {"Content-Type", "application/json"}

      # Verify extra headers are included
      account_id_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "chatgpt-account-id" end)

      assert account_id_header == {"chatgpt-account-id", "test-account-123"}

      custom_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "x-custom-header" end)

      assert custom_header == {"x-custom-header", "custom-value"}
    end

    test "works with no extra headers", %{model: model, context: context} do
      opts = [api_key: "test-openai"]

      {:ok, finch_request} =
        ReqLLM.Providers.OpenAI.ChatAPI.attach_stream(model, context, opts, nil)

      # Should have only base headers
      assert length(finch_request.headers) >= 3

      auth_header = Enum.find(finch_request.headers, fn {k, _v} -> k == "Authorization" end)
      assert auth_header == {"Authorization", "Bearer test-openai"}

      content_type_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "Content-Type" end)

      assert content_type_header == {"Content-Type", "application/json"}
    end
  end

  describe "OpenAI ResponsesAPI extra headers from req_http_options" do
    setup do
      {:ok, model} = ReqLLM.model("openai:o1")
      context = Context.new([Context.user("test")])
      {:ok, model: model, context: context}
    end

    test "extra headers are included in streaming request", %{model: model, context: context} do
      opts = [
        api_key: "test-openai",
        req_http_options: [
          headers: [
            {"chatgpt-account-id", "test-account-456"},
            {"x-responses-custom", "responses-value"}
          ]
        ]
      ]

      {:ok, finch_request} =
        ReqLLM.Providers.OpenAI.ResponsesAPI.attach_stream(model, context, opts, nil)

      # Verify Authorization header exists
      auth_header = Enum.find(finch_request.headers, fn {k, _v} -> k == "Authorization" end)
      assert auth_header == {"Authorization", "Bearer test-openai"}

      # Verify Content-Type header exists
      content_type_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "Content-Type" end)

      assert content_type_header == {"Content-Type", "application/json"}

      # Verify extra headers are included
      account_id_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "chatgpt-account-id" end)

      assert account_id_header == {"chatgpt-account-id", "test-account-456"}

      custom_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "x-responses-custom" end)

      assert custom_header == {"x-responses-custom", "responses-value"}
    end

    test "works with no extra headers", %{model: model, context: context} do
      opts = [api_key: "test-openai"]

      {:ok, finch_request} =
        ReqLLM.Providers.OpenAI.ResponsesAPI.attach_stream(model, context, opts, nil)

      # Should have only base headers
      assert length(finch_request.headers) >= 3

      auth_header = Enum.find(finch_request.headers, fn {k, _v} -> k == "Authorization" end)
      assert auth_header == {"Authorization", "Bearer test-openai"}

      content_type_header =
        Enum.find(finch_request.headers, fn {k, _v} -> k == "Content-Type" end)

      assert content_type_header == {"Content-Type", "application/json"}
    end
  end
end
