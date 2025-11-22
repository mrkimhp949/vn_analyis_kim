# -*- coding: utf-8 -*-
"""
Authentication and Security Module
"""
import logging
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# CRITICAL FIX: Environment check for security
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
ALLOWED_ENVIRONMENTS = ["dev", "development", "staging", "production"]

if ENVIRONMENT not in ALLOWED_ENVIRONMENTS:
    logger.warning(
        f"⚠️ Unknown ENVIRONMENT: {ENVIRONMENT}. Defaulting to 'production' for safety."
    )
    ENVIRONMENT = "production"

# Load API keys from environment
VALID_API_KEYS = set()
api_keys_str = os.getenv("API_KEYS", "")
if api_keys_str:
    VALID_API_KEYS = set(key.strip() for key in api_keys_str.split(",") if key.strip())

# IP Whitelist
IP_WHITELIST = set()
ip_whitelist_str = os.getenv("IP_WHITELIST", "")
if ip_whitelist_str:
    IP_WHITELIST = set(ip.strip() for ip in ip_whitelist_str.split(",") if ip.strip())

# CRITICAL: Log security configuration
if ENVIRONMENT in ["dev", "development"]:
    if not VALID_API_KEYS:
        logger.warning(
            "🔓 DEVELOPMENT MODE: API key validation DISABLED. "
            "This is UNSAFE for production!"
        )
else:
    if not VALID_API_KEYS:
        logger.error(
            "🚨 CRITICAL SECURITY ERROR: No API keys configured in "
            f"{ENVIRONMENT.upper()} environment! "
            "Set API_KEYS environment variable or switch to ENVIRONMENT=dev"
        )


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header

    CRITICAL FIX: Only allow development mode if ENVIRONMENT=dev.
    In production/staging, API keys are REQUIRED.

    Args:
        api_key: API key from header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not VALID_API_KEYS:
        # CRITICAL FIX: Only allow development mode in dev/development environment
        if ENVIRONMENT in ["dev", "development"]:
            logger.warning(
                "🔓 Development mode: No API key validation. "
                f"ENVIRONMENT={ENVIRONMENT}"
            )
            return "dev_mode"
        else:
            # CRITICAL: In production/staging, API keys are MANDATORY
            logger.error(
                f"🚨 CRITICAL: No API keys configured in {ENVIRONMENT.upper()} environment! "
                "Server should not have started. Denying all requests."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Server misconfiguration: No API keys in {ENVIRONMENT} mode"
            )

    if not api_key:
        logger.warning("❌ Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API key"
        )

    if api_key not in VALID_API_KEYS:
        logger.warning(f"❌ Invalid API key: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return api_key


async def verify_ip_whitelist(request):
    """
    Verify client IP is in whitelist

    Args:
        request: FastAPI request object

    Raises:
        HTTPException: If IP not in whitelist
    """
    if not IP_WHITELIST:
        # If no whitelist configured, allow all
        return

    client_ip = request.client.host

    if client_ip not in IP_WHITELIST:
        logger.warning(f"❌ Blocked request from IP: {client_ip}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="IP not whitelisted")


# Rate limit decorators
def rate_limit_strict(func):
    """Strict rate limit: 5 requests per minute"""
    return limiter.limit("5/minute")(func)


def rate_limit_moderate(func):
    """Moderate rate limit: 20 requests per minute"""
    return limiter.limit("20/minute")(func)


def rate_limit_relaxed(func):
    """Relaxed rate limit: 60 requests per minute"""
    return limiter.limit("60/minute")(func)


# Security headers middleware
def add_security_headers(response):
    """Add security headers to response"""
    response.headers["X-Content-Type-Options"] = "nosnif"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


if __name__ == "__main__":
    # Generate sample API keys
    print("🔑 Sample API Keys:")
    for i in range(3):
        key = generate_api_key()
        print(f"  Key {i+1}: {key}")

    print("\n📝 Add to .env file:")
    print("API_KEYS=key1,key2,key3")
    print("IP_WHITELIST=127.0.0.1,192.168.1.100")
