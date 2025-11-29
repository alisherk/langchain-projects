#!/usr/bin/env bash

# Add uv to PATH if it exists in the user's local bin
if [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if we're in a git repository before running git commands
if git rev-parse --git-dir > /dev/null 2>&1; then
    git config --local safe.directory $HOME/workspace
    git config --local pull.rebase false

    if [ -d ".githooks" ]; then
        git config --local core.hooksPath .githooks
        chmod +x .githooks/*
    fi
else
    echo "Not in a git repository, skipping git configuration"
fi

uv sync