#!/usr/bin/env bash
set -euo pipefail

git fetch upstream
git pull upstream main
git push origin main
