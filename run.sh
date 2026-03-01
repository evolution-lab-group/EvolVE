#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required to launch multiple runs. Please install tmux and retry." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to run experiments. Please install uv and retry." >&2
    echo "Installation: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    cat <<'EOF' >&2
Usage: ./run.sh <config1.yaml> [config2.yaml ...]

Each config will be launched in its own tmux session running:
    uv run main.py -c <config>
EOF
    exit 1
fi

sanitize_session_name() {
    local input="$1"
    # Replace non-alphanumeric characters with underscores and trim to 30 chars.
    local sanitized
    sanitized="$(echo "$input" | tr -c 'A-Za-z0-9_' '_' | cut -c1-30)"
    # Ensure name is not empty and does not start with a digit.
    if [[ -z "$sanitized" ]]; then
        sanitized="session"
    elif [[ "$sanitized" =~ ^[0-9] ]]; then
        sanitized="_$sanitized"
    fi
    echo "$sanitized"
}

unique_session_name() {
    local base="$1"
    local candidate="$base"
    local suffix=1
    while tmux has-session -t "$candidate" 2>/dev/null; do
        local suffix_str="_$suffix"
        local max_base_length=$((30 - ${#suffix_str}))
        if (( max_base_length < 1 )); then
            max_base_length=1
        fi
        local truncated_base="$base"
        if (( ${#truncated_base} > max_base_length )); then
            truncated_base="${truncated_base:0:max_base_length}"
        fi
        candidate="${truncated_base}${suffix_str}"
        ((suffix++))
    done
    echo "$candidate"
}

for config_path in "$@"; do
    if [ ! -f "$config_path" ]; then
        echo "Config file not found: $config_path" >&2
        continue
    fi

    abs_config_path="$(cd "$(dirname "$config_path")" && pwd)/$(basename "$config_path")"
    config_basename="$(basename "${config_path%.*}")"
    config_dirname="$(basename "$(dirname "$config_path")")"
    session_name="$(unique_session_name "$(sanitize_session_name "${config_dirname}_${config_basename}")")"

    tmux new-session -d -s "$session_name" bash -lc \
        "cd '$PROJECT_ROOT' && uv run main.py -c '$abs_config_path'"

    echo "Launched tmux session '$session_name' for config $abs_config_path"
done
