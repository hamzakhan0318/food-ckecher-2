# 🥗 Daily Nutrition Tracker

A Streamlit app where you log your daily food (by typing or uploading a photo) and
Groq AI estimates calories, protein, carbohydrates, fat, fiber and sugar. You get a
daily summary against your goals, charts, a food log, and optional AI coaching tips.

> Nutrition values are AI **estimates**, not medical advice.

## Features
- Describe a meal in plain text, or upload a photo of it
- Per-item and daily totals for calories, protein, carbs, fat, fiber, sugar
- Progress bars against your own daily goals
- Charts for macros and calories by meal
- Remove wrong items, download your log as CSV, restore it later
- "AI coach" tips based on the day's totals

## Files
| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## 1. Get a Groq API key
Create a free key at <https://console.groq.com/keys>. Keep it private and never put it in `app.py`.

## 2. Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Paste your key into the sidebar, or provide it in one of these ways:
- Environment variable: `GROQ_API_KEY=your_key`
- File `.streamlit/secrets.toml` containing: `GROQ_API_KEY = "your_key"`
  (do **not** upload this file to GitHub)

## 3. Push to GitHub using the website (no Git needed)
1. Sign in at <https://github.com> and click **+ → New repository**.
2. Name it (for example `nutrition-tracker`), choose **Public** (or Private), and click **Create repository**.
3. On the new repo page click **uploading an existing file**.
4. Drag in `app.py`, `requirements.txt` and `README.md`, then click **Commit changes**.
5. Do not upload any file containing your API key.

## 4. Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and sign in with GitHub (allow access to your repo).
2. Click **Create app** (or **New app**) and choose to deploy from an existing repo.
3. Pick your repository, branch `main`, and set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Click **Deploy**. After a minute or two you get a public `*.streamlit.app` link.

To update the app later, edit the file on GitHub (pencil icon → **Commit changes**); Streamlit redeploys automatically.

## Notes and troubleshooting
- **Data is not stored permanently.** The log lives in your browser session. Use
  **Download full log (CSV)** in the sidebar and **Restore from CSV** next time.
- **"model not found / decommissioned" error:** Groq changes its model list. Open the sidebar
  **Advanced: models** box and enter a current model name from <https://console.groq.com/docs/models>
  (the photo model must support images).
- **Invalid API key error:** re-check the key in the sidebar or in Secrets.
- Photos are resized before upload; results from photos are rougher than from text with portions.
