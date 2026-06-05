"""Asqend container identity echo helpers.

Hosted Asqend runs Hermes in one container per org.  These helpers expose the
container-local identity that Asqend configured at process start, so callers
can prove the dashboard/OAuth surface and gateway/session surface belong to the
same org container.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

ASQEND_ORG_ID_ENV = "ASQEND_ORG_ID"
ASQEND_CONTAINER_REF_ENV = "ASQEND_HERMES_CONTAINER_REF"
ASQEND_EXPECTED_ORG_ID_HEADER = "x-asqend-org-id"
ASQEND_EXPECTED_CONTAINER_REF_HEADER = "x-asqend-hermes-container-ref"
ASQEND_IDENTITY_VERSION = "2026-06-04.asqend-hermes-container-identity"
ASQEND_IDENTITY_SOURCE = "process_env"

PROTECTED_ENV_VARS = frozenset({
    ASQEND_ORG_ID_ENV,
    ASQEND_CONTAINER_REF_ENV,
})


class AsqendIdentityMismatch(ValueError):
    """Raised when request-supplied expected identity does not match boot identity."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _clean(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _snapshot_identity() -> Optional[dict[str, str]]:
    org_id = _clean(os.environ.get(ASQEND_ORG_ID_ENV))
    container_ref = _clean(os.environ.get(ASQEND_CONTAINER_REF_ENV))
    if not org_id or not container_ref:
        return None
    return {
        "org_id": org_id,
        "container_ref": container_ref,
        "source": ASQEND_IDENTITY_SOURCE,
        "version": ASQEND_IDENTITY_VERSION,
    }


_BOOT_IDENTITY = _snapshot_identity()


def get_asqend_identity(surface: str) -> Optional[dict[str, str]]:
    """Return process-start Asqend identity for a response surface."""
    if _BOOT_IDENTITY is None:
        return None
    return {
        **_BOOT_IDENTITY,
        "surface": surface,
    }


def with_asqend_identity(payload: dict[str, Any], surface: str) -> dict[str, Any]:
    """Return ``payload`` plus ``asqend_identity`` when configured."""
    identity = get_asqend_identity(surface)
    if identity is None:
        return payload
    return {
        **payload,
        "asqend_identity": identity,
    }


def _header(headers: Mapping[str, Any], name: str) -> Optional[str]:
    try:
        return _clean(headers.get(name))  # type: ignore[arg-type]
    except Exception:
        return None


def validate_expected_asqend_identity(headers: Mapping[str, Any]) -> None:
    """Reject mismatched expected identity headers.

    Headers are validation hints from Asqend, never a source of identity.  When
    neither expected header is present, the request is accepted.  When either is
    present, Hermes must already have process-start identity and the supplied
    values must match it exactly.
    """
    expected_org_id = _header(headers, ASQEND_EXPECTED_ORG_ID_HEADER)
    expected_container_ref = _header(headers, ASQEND_EXPECTED_CONTAINER_REF_HEADER)
    if not expected_org_id and not expected_container_ref:
        return
    if not expected_org_id or not expected_container_ref:
        raise AsqendIdentityMismatch(
            "asqend_identity_expected_incomplete",
            "Expected Asqend identity requires both org and container headers",
        )
    if _BOOT_IDENTITY is None:
        raise AsqendIdentityMismatch(
            "asqend_identity_missing",
            "Hermes process identity is not configured",
        )
    if (
        expected_org_id != _BOOT_IDENTITY["org_id"]
        or expected_container_ref != _BOOT_IDENTITY["container_ref"]
    ):
        raise AsqendIdentityMismatch(
            "asqend_identity_mismatch",
            "Expected Asqend identity does not match Hermes process identity",
        )
