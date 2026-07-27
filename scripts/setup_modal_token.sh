#!/usr/bin/env bash
# Sanitizes MODAL_TOKEN_ID / MODAL_TOKEN_SECRET (strips stray straight/curly
# quote characters that sometimes get pasted into secret values) and then
# registers + activates the Modal CLI token/profile.
set -euo pipefail

strip_quotes() {
  # Removes ASCII " and ' plus UTF-8 curly quotes U+201C/U+201D.
  printf '%s' "$1" | LC_ALL=C.UTF-8 sed -e "s/[\"']//g" -e 's/\xe2\x80\x9c//g' -e 's/\xe2\x80\x9d//g'
}

CLEAN_TOKEN_ID="$(strip_quotes "${MODAL_TOKEN_ID:-}")"
CLEAN_TOKEN_SECRET="$(strip_quotes "${MODAL_TOKEN_SECRET:-}")"

if [ -z "$CLEAN_TOKEN_ID" ] || [ -z "$CLEAN_TOKEN_SECRET" ]; then
  echo "ERROR: MODAL_TOKEN_ID or MODAL_TOKEN_SECRET is empty after sanitization" >&2
  exit 1
fi

uv run modal token set --token-id "$CLEAN_TOKEN_ID" --token-secret "$CLEAN_TOKEN_SECRET" --profile=kutyamajom
uv run modal profile activate kutyamajom
