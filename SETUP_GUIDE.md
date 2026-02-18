# POE2 Price Overlay — Setup Guide

**What it does:** Hover over items in POE2, press Ctrl+C, and see the price instantly. Also auto-generates a loot filter with economy-based tiering.

---

## Step 1: Install Python

Go to https://www.python.org/downloads/ and download the latest version.

When installing, **check the box that says "Add Python to PATH"**. This is the one thing people mess up — if you miss it, nothing works.

## Step 2: Install Git

Go to https://git-scm.com/downloads/win and install with all the defaults (just keep clicking Next).

## Step 3: Download the tool

Open a folder where you want it (Desktop is fine). Click the address bar at the top of File Explorer, type `powershell`, hit Enter. Paste this and hit Enter:

```
git clone https://github.com/CarbonSMASH/POE2_OCR.git
```

## Step 4: Run it

Open the `POE2_OCR` folder and double-click **`POE2 Dashboard.bat`**.

First time it'll install a few packages automatically (takes ~30 seconds), then the dashboard window opens. Hit **Start** and you're good.

---

## Using it

- Hover over any item in POE2 and press **Ctrl+C** — price pops up near your cursor
- **Ctrl+Shift+C** — deep trade API search on the last item (for rares with specific mods)
- The **Loot Filter** tab lets you customize filter strictness, colors, and visibility
- Click **Update Filter Now** to regenerate your filter from live prices

## Running Tests

Double-click **`RUN_TESTS.bat`** — it spawns a PowerShell window for each test module so you can see results side by side.

Or run in a single terminal:

```
python -m pytest tests/ -v
```

To run just one module:

```
python run_tests.py --module mod_database
```

## Updating

Open PowerShell in the `POE2_OCR` folder and run:

```
git pull
```
