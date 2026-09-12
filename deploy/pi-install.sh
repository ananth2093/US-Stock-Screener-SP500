#!/bin/bash
# Install systemd services on the Raspberry Pi.
# Run this from inside the repo folder on the Pi.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

sudo cp "$SCRIPT_DIR/pi-screener-dashboard.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/pi-screener-updater.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/pi-screener-updater.timer" /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable pi-screener-dashboard
sudo systemctl enable pi-screener-updater.timer
sudo systemctl start pi-screener-dashboard
sudo systemctl start pi-screener-updater.timer

echo "Services installed. Check status with:"
echo "  sudo systemctl status pi-screener-dashboard"
echo "  sudo systemctl status pi-screener-updater.timer"
