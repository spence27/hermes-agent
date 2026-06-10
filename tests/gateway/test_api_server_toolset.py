"""Tests for hermes-api-server toolset and API server tool availability."""
import json
from unittest.mock import patch, MagicMock

import pytest


from toolsets import resolve_toolset, get_toolset, validate_toolset


class TestHermesApiServerToolset:
    """Tests for the hermes-api-server toolset definition."""

    def test_toolset_exists(self):
        ts = get_toolset("hermes-api-server")
        assert ts is not None

    def test_toolset_validates(self):
        assert validate_toolset("hermes-api-server")

    def test_toolset_includes_web_tools(self):
        tools = resolve_toolset("hermes-api-server")
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("hermes-api-server")
        expected = [
            "terminal", "process",
            "read_file", "write_file", "patch", "search_files",
            "vision_analyze", "image_generate",
            "execute_code", "delegate_task",
            "todo", "memory", "session_search", "cronjob",
        ]
        for tool in expected:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                      "browser_type", "browser_scroll", "browser_back",
                      "browser_press"]:
            assert tool in tools, f"Missing browser tool: {tool}"

    def test_toolset_includes_homeassistant_tools(self):
        tools = resolve_toolset("hermes-api-server")
        for tool in ["ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service"]:
            assert tool in tools, f"Missing HA tool: {tool}"

    def test_toolset_excludes_clarify(self):
        tools = resolve_toolset("hermes-api-server")
        assert "clarify" not in tools

    def test_toolset_excludes_send_message(self):
        tools = resolve_toolset("hermes-api-server")
        assert "send_message" not in tools

    def test_toolset_excludes_text_to_speech(self):
        tools = resolve_toolset("hermes-api-server")
        assert "text_to_speech" not in tools


class TestApiServerPlatformConfig:
    def test_platforms_dict_includes_api_server(self):
        from hermes_cli.tools_config import PLATFORMS
        assert "api_server" in PLATFORMS
        assert PLATFORMS["api_server"]["default_toolset"] == "hermes-api-server"


class TestApiServerAdapterToolset:
    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_reads_config_toolsets(self):
        """API server resolves toolsets from config like all other platforms."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # No platform_toolsets override — should fall back to hermes-api-server default
            mock_config.return_value = {}
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert isinstance(toolsets, list)
            assert len(toolsets) > 0
            assert call_kwargs.kwargs.get("platform") == "api_server"

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_respects_config_override(self):
        """User can override API server toolsets via platform_toolsets in config.yaml."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())

        with patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            # User overrides with just web and terminal
            mock_config.return_value = {
                "platform_toolsets": {"api_server": ["web", "terminal"]}
            }
            mock_agent_cls.return_value = MagicMock()

            adapter._create_agent()

            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args
            toolsets = call_kwargs.kwargs.get("enabled_toolsets")
            assert sorted(toolsets) == ["terminal", "web"]


class TestApiServerSessionMcpToolsets:
    @staticmethod
    def _runtime_server_name(session_id, server_name):
        from gateway.platforms.api_server import APIServerAdapter

        return APIServerAdapter._session_mcp_server_runtime_name(session_id, server_name)

    @staticmethod
    def _session(mcp_servers):
        return {
            "id": "session",
            "source": "api_server",
            "tool_config": json.dumps({"mcp_servers": mcp_servers}),
        }

    @staticmethod
    def _patch_agent_runtime():
        return (
            patch("gateway.run._resolve_runtime_agent_kwargs"),
            patch("gateway.run._resolve_gateway_model"),
            patch("gateway.run._load_gateway_config"),
            patch("run_agent.AIAgent"),
            patch("tools.mcp_tool.register_mcp_servers"),
        )

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_registers_session_mcp_servers(self):
        """API-server sessions can attach their own MCP servers without config.yaml edits."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        class _FakeDB:
            def get_session(self, session_id):
                assert session_id == "email-session"
                return TestApiServerSessionMcpToolsets._session(
                    {
                        "email_ops": {
                            "url": "https://email.example/mcp",
                            "headers": {"Authorization": "Bearer email-token"},
                        }
                    }
                )

        adapter = APIServerAdapter(PlatformConfig())
        adapter._session_db = _FakeDB()
        m_kwargs, m_model, m_config, m_agent_cls, m_register = self._patch_agent_runtime()
        with m_kwargs as mock_kwargs, m_model as mock_model, m_config as mock_config, \
             m_agent_cls as mock_agent_cls, m_register as mock_register:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"api_server": ["web"]}}
            mock_agent_cls.return_value = MagicMock()
            runtime_name = self._runtime_server_name("email-session", "email_ops")
            mock_register.return_value = [f"mcp_{runtime_name}_session_identity"]

            adapter._create_agent(session_id="email-session")

        mock_register.assert_called_once_with(
            {
                runtime_name: {
                    "url": "https://email.example/mcp",
                    "headers": {"Authorization": "Bearer email-token"},
                }
            }
        )
        toolsets = mock_agent_cls.call_args.kwargs["enabled_toolsets"]
        assert toolsets == [f"mcp-{runtime_name}", "web"]

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_namespaces_same_mcp_server_per_session(self):
        """Concurrent API-server sessions can reuse logical MCP names with different keys."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        class _FakeDB:
            sessions = {
                "email-session": TestApiServerSessionMcpToolsets._session(
                    {
                        "asqend": {
                            "url": "https://email.example/mcp",
                            "headers": {"Authorization": "Bearer email-token"},
                        }
                    }
                ),
                "social-session": TestApiServerSessionMcpToolsets._session(
                    {
                        "asqend": {
                            "url": "https://social.example/mcp",
                            "headers": {"Authorization": "Bearer social-token"},
                        }
                    }
                ),
            }

            def get_session(self, session_id):
                return self.sessions.get(session_id)

        adapter = APIServerAdapter(PlatformConfig())
        adapter._session_db = _FakeDB()
        m_kwargs, m_model, m_config, m_agent_cls, m_register = self._patch_agent_runtime()
        with m_kwargs as mock_kwargs, m_model as mock_model, m_config as mock_config, \
             m_agent_cls as mock_agent_cls, m_register as mock_register:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"api_server": ["web"]}}
            mock_agent_cls.return_value = MagicMock()
            email_runtime_name = self._runtime_server_name("email-session", "asqend")
            social_runtime_name = self._runtime_server_name("social-session", "asqend")
            mock_register.side_effect = [
                [f"mcp_{email_runtime_name}_session_identity"],
                [
                    f"mcp_{email_runtime_name}_session_identity",
                    f"mcp_{social_runtime_name}_session_identity",
                ],
            ]

            adapter._create_agent(session_id="email-session")
            adapter._create_agent(session_id="social-session")

        first_toolsets = mock_agent_cls.call_args_list[0].kwargs["enabled_toolsets"]
        second_toolsets = mock_agent_cls.call_args_list[1].kwargs["enabled_toolsets"]

        assert email_runtime_name != social_runtime_name
        assert first_toolsets == [f"mcp-{email_runtime_name}", "web"]
        assert f"mcp-{social_runtime_name}" not in first_toolsets
        assert second_toolsets == [f"mcp-{social_runtime_name}", "web"]
        assert f"mcp-{email_runtime_name}" not in second_toolsets
        assert mock_register.call_args_list[0].args[0][email_runtime_name]["headers"] == {
            "Authorization": "Bearer email-token"
        }
        assert mock_register.call_args_list[1].args[0][social_runtime_name]["headers"] == {
            "Authorization": "Bearer social-token"
        }

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
    def test_create_agent_fails_closed_when_session_mcp_does_not_register_toolset(self):
        """A session with required MCP config must not silently run without those tools."""
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        class _FakeDB:
            def get_session(self, session_id):
                assert session_id == "email-session"
                return TestApiServerSessionMcpToolsets._session(
                    {"asqend": {"url": "https://email.example/mcp"}}
                )

        adapter = APIServerAdapter(PlatformConfig())
        adapter._session_db = _FakeDB()
        m_kwargs, m_model, m_config, m_agent_cls, m_register = self._patch_agent_runtime()
        with m_kwargs as mock_kwargs, m_model as mock_model, m_config as mock_config, \
             m_agent_cls as mock_agent_cls, m_register as mock_register:

            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = {"platform_toolsets": {"api_server": ["web"]}}
            mock_agent_cls.return_value = MagicMock()
            mock_register.return_value = []

            with pytest.raises(RuntimeError, match="required session MCP servers"):
                adapter._create_agent(session_id="email-session")

        mock_agent_cls.assert_not_called()
