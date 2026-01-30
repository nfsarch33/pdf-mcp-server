#!/bin/bash
# Block destructive shell commands.
input="$(cat)"
cmd="$(printf "%s" "$input" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"

case "$cmd" in
  *"git reset --hard"*|*"git checkout --"*|*"rm -rf "*)
    printf '%s\n' '{"permission":"deny","user_message":"Blocked destructive command. Use a safer alternative or request explicitly."}'
    exit 0
    ;;
  *)
    printf '%s\n' '{"permission":"allow"}'
    exit 0
    ;;
esac
