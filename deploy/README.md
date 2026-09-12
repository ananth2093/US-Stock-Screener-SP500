# Raspberry Pi deployment

This folder contains systemd service/timer files to keep the screener running locally on a Pi.

## Setup

1. Copy this repo to the Pi:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /home/pi/screener
   cd /home/pi/screener
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create an environment file `/home/pi/screener/.env` with at least:
   ```bash
   FMP_API_KEY=your_fmp_api_key_here
   ```

4. Test the updater once:
   ```bash
   python update_data.py
   ```

5. Install the systemd services:
   ```bash
   chmod +x deploy/pi-install.sh
   ./deploy/pi-install.sh
   ```

## What the services do

- `pi-screener-dashboard.service` — runs `streamlit run screener_app.py` on port 8501, starts on boot, and restarts if it crashes.
- `pi-screener-updater.service` — runs `python update_data.py` when triggered.
- `pi-screener-updater.timer` — triggers the updater every hour.

`update_data.py` has a built-in throttle:
- It runs every hour while the US market is open (roughly 09:30–16:00 EST/EDT, weekdays).
- Outside market hours it runs at most every 4 hours.
- If no snapshot exists yet, it always runs.

## Access the dashboard

From any device on your home network:

```text
http://YOUR_PI_IP:8501
```

## Useful commands

```bash
# View dashboard status
sudo systemctl status pi-screener-dashboard

# View updater timer status
sudo systemctl status pi-screener-updater.timer

# View latest updater run
sudo journalctl -u pi-screener-updater -n 100 -f

# Run updater manually
sudo systemctl start pi-screener-updater

# Stop dashboard
sudo systemctl stop pi-screener-dashboard
```
