# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module declaring the singleton MCP instance.

The singleton allows other modules to register their tools with the same MCP
server using `@mcp.tool` annotations, thereby 'coordinating' the bootstrapping
of the server.
"""

import logging
import os
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp_tasks import TasksExtension
from mcp import types as mcp_types
from mcp.server.auth.provider import TokenError
from mcp.server.subscriptions import InMemorySubscriptionBus, ListenHandler
from ads_mcp.auth_storage import create_client_storage

logger = logging.getLogger(__name__)

_CLIENT_ID = os.environ.get("GOOGLE_ADS_MCP_OAUTH_CLIENT_ID")
_CLIENT_SECRET = os.environ.get("GOOGLE_ADS_MCP_OAUTH_CLIENT_SECRET")
_BASE_URL = os.environ.get("GOOGLE_ADS_MCP_BASE_URL", "http://localhost:8080")
_JWT_SIGNING_KEY = os.environ.get("GOOGLE_ADS_MCP_JWT_SIGNING_KEY")
_ALLOWED_EMAILS_ENV = "GOOGLE_ADS_MCP_ALLOWED_EMAILS"


def _allowed_emails_from_env() -> frozenset[str]:
    """Comma-separated allowlist. Empty or unset means nobody is allowed."""
    raw = os.environ.get(_ALLOWED_EMAILS_ENV, "")
    return frozenset(
        piece.strip().casefold() for piece in raw.split(",") if piece.strip()
    )


def _email_is_verified(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().casefold() == "true"


def _email_is_allowed(claims: dict[str, Any] | None, allowed: frozenset[str]) -> bool:
    if not allowed or not claims:
        return False
    email = claims.get("email")
    if not isinstance(email, str) or not _email_is_verified(claims.get("email_verified")):
        return False
    return email.strip().casefold() in allowed


class _EmailAllowlistVerifier(TokenVerifier):
    """Reject Google tokens whose verified email is not on the allowlist."""

    def __init__(self, inner: TokenVerifier, allowed_emails: frozenset[str]):
        super().__init__(required_scopes=list(inner.required_scopes or []))
        self._inner = inner
        self._allowed_emails = allowed_emails

    async def verify_token(self, token: str) -> AccessToken | None:
        access = await self._inner.verify_token(token)
        if access is None:
            return None
        if not _email_is_allowed(access.claims, self._allowed_emails):
            logger.warning("Rejected Google sign-in: account is not allowlisted")
            return None
        return access


class _AllowlistedGoogleProvider(GoogleProvider):
    """GoogleProvider that refuses every account unless the allowlist permits it."""

    def __init__(self, *, allowed_emails: frozenset[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        inner = getattr(self, "_token_validator", None)
        if inner is None or not hasattr(inner, "verify_token"):
            raise RuntimeError(
                "Refusing to start OAuth without an email allowlist hook"
            )
        self._allowed_emails = allowed_emails
        self._token_validator = _EmailAllowlistVerifier(inner, allowed_emails)
        if not allowed_emails:
            logger.warning(
                "%s is empty or unset; OAuth sign-in is disabled",
                _ALLOWED_EMAILS_ENV,
            )

    async def exchange_authorization_code(self, client: Any, authorization_code: Any) -> Any:
        code_store = getattr(self, "_code_store", None)
        code = getattr(authorization_code, "code", None)
        code_model = None
        if code_store is not None and isinstance(code, str):
            code_model = await code_store.get(key=code)
        idp_tokens = getattr(code_model, "idp_tokens", None)
        access_token = idp_tokens.get("access_token") if isinstance(idp_tokens, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise TokenError("access_denied", "Google account is not allowed")
        if await self._token_validator.verify_token(access_token) is None:
            raise TokenError("access_denied", "Google account is not allowed")
        return await super().exchange_authorization_code(client, authorization_code)


if _CLIENT_ID and _CLIENT_SECRET:
    client_storage = create_client_storage()
    provider_kwargs: dict[str, Any] = {
        "client_id": _CLIENT_ID,
        "client_secret": _CLIENT_SECRET,
        "base_url": _BASE_URL,
        "required_scopes": [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/adwords",
        ],
    }
    if _JWT_SIGNING_KEY:
        provider_kwargs["jwt_signing_key"] = _JWT_SIGNING_KEY
    if client_storage is not None:
        provider_kwargs["client_storage"] = client_storage

    auth = _AllowlistedGoogleProvider(
        allowed_emails=_allowed_emails_from_env(),
        **provider_kwargs,
    )
    mcp = FastMCP("Google Ads Server", auth=auth)
else:
    mcp = FastMCP("Google Ads Server")

mcp.add_extension(TasksExtension())


def ensure_subscriptions_listen(server: FastMCP) -> bool:
    """Register the MCP 2026 subscription stream when FastMCP omits it.

    FastMCP 4 (including 4.0.3) builds its low-level ``Server`` without the SDK's
    ``on_subscriptions_listen`` handler. Some modern clients open that stream
    while loading tools; without it, the SDK returns HTTP 404 / method not
    found, which mcp-go reports misleadingly as a missing session.

    The guard preserves a native FastMCP implementation once one is shipped.

    Returns:
        True when the compatibility handler was installed, otherwise False.
    """
    low_level_server = server._mcp_server
    if "subscriptions/listen" in low_level_server._request_handlers:
        return False

    subscription_bus = InMemorySubscriptionBus()
    low_level_server.add_request_handler(
        "subscriptions/listen",
        mcp_types.SubscriptionsListenRequestParams,
        ListenHandler(subscription_bus),
    )
    return True


ensure_subscriptions_listen(mcp)


def initialize_and_mount_tools(parent_mcp: FastMCP) -> None:
    """Loads the tools configuration and dynamically mounts the tools sub-servers."""
    from ads_mcp.config import ToolsConfig
    import importlib
    import pkgutil
    import ads_mcp.tools as tools_pkg

    # Map of category name -> FastMCP sub-server
    sub_servers = {}

    # Discover and dynamically load all tool modules
    for _, module_name, _ in pkgutil.iter_modules(tools_pkg.__path__):
        full_module_name = f"ads_mcp.tools.{module_name}"
        module = importlib.import_module(full_module_name)

        # Find any FastMCP instances defined in the module
        for attr_name in dir(module):
            attr_val = getattr(module, attr_name)
            if isinstance(attr_val, FastMCP):
                category = attr_val.name
                if (
                    category in sub_servers
                    and sub_servers[category] is not attr_val
                ):
                    raise ValueError(
                        f"Conflicting FastMCP sub-servers registered for category '{category}'."
                    )
                sub_servers[category] = attr_val

    config = ToolsConfig.load()

    for category, sub_mcp in sub_servers.items():
        if not config.is_namespace_enabled(category):
            continue

        # Filter disabled tools inside the sub-server before mounting
        tool_names = []
        for key, val in sub_mcp.local_provider._components.items():
            if key.startswith("tool:"):
                tool_names.append(val.name)

        for name in tool_names:
            if not config.is_tool_enabled(category, name):
                sub_mcp.local_provider.remove_tool(name)

        # Determine prefix/namespace
        namespace_prefix = config.get_namespace_prefix(category)

        # Mount the sub-server
        parent_mcp.mount(sub_mcp, namespace=namespace_prefix or None)


# Automatically initialize and mount tools upon import
initialize_and_mount_tools(mcp)
