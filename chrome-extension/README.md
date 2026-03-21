# Token Savings Analyzer (Chrome Extension)

## What it does
- Connects directly to the local OptiPrompt backend for active compression.
- Compares token estimates for original and optimized prompts.
- Calculates estimated tokens saved and savings percentage.
- Saves analyses to `chrome.storage.local`.
- Shows historical metrics and a trend chart on the analytics page.

## Start backend
1. From project root, activate your environment.
2. Run: `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
3. Verify: `http://127.0.0.1:8000/health`

## Load in Chrome
1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select this folder: `chrome-extension`.

## Usage
1. Open the extension popup.
2. Set **Backend URL** (default: `http://127.0.0.1:8000`) and optimization mode.
3. Paste the original prompt.
4. Click **Compress** to call backend `/optimize` and fill optimized output.
5. Click **Save** to store the result.
6. Click **Open Full Analytics** for history, trend, export, and clear actions.

## Notes
- Token counts are estimates, not model-exact tokenizer values.
- Up to 200 recent entries are kept in local storage.
