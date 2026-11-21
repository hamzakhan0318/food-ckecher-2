"""Daily Nutrition Tracker: log what you eat and let Groq AI estimate the nutrients.

Run locally:  streamlit run app.py
"""

import base64
import io
import json
import os
import re
from datetime import date

import pandas as pd
import streamlit as st
from groq import Groq
from PIL import Image, ImageOps

# Models can be changed in the sidebar (Advanced) or via environment variables,
# because Groq retires and adds models over time. See https://console.groq.com/docs/models
DEFAULT_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
DEFAULT_VISION_MODEL = os.environ.get(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)

MEALS = ["Breakfast", "Lunch", "Dinner", "Snack"]
NUTRIENTS = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g"]
COLUMN_LABELS = {
    "meal": "Meal",
    "name": "Food",
    "quantity": "Quantity",
    "calories": "Calories (kcal)",
    "protein_g": "Protein (g)",
    "carbs_g": "Carbs (g)",
    "fat_g": "Fat (g)",
    "fiber_g": "Fiber (g)",
    "sugar_g": "Sugar (g)",
}
ENTRY_FIELDS = ["date", "meal", "name", "quantity"] + NUTRIENTS

SYSTEM_PROMPT = """You are a nutrition analysis assistant.
The user describes food they ate (and may attach a photo). Identify each food item,
estimate a realistic portion if none is given, and estimate its nutrients.

Respond with ONLY a JSON object in exactly this shape (no extra text, no markdown):
{"items": [{"name": "string", "quantity": "string, e.g. '2 slices' or '150 g'",
"calories": number, "protein_g": number, "carbs_g": number, "fat_g": number,
"fiber_g": number, "sugar_g": number}]}

Rules: one object per distinct food item; all nutrient values are numbers for the
whole quantity listed (not per 100 g); use 0 when a nutrient is negligible."""


# ----------------------------------------------------------------------------
# Helpers (pure functions: easy to test)
# ----------------------------------------------------------------------------
def _to_float(value) -> float:
    """Convert AI output such as 12, '12.5', or '12 g' to a non-negative float."""
    try:
        if isinstance(value, str):
            value = re.sub(r"[^0-9.]", "", value)
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number < 0:  # NaN or negative
        return 0.0
    return round(number, 1)


def parse_nutrition_json(text: str) -> list[dict]:
    """Turn the model's reply into a clean list of food items."""
    if not text or not text.strip():
        raise ValueError("The AI returned an empty response.")

    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("The AI reply was not valid JSON.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("The AI reply was not valid JSON.") from exc

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError("The AI did not return any food items.")

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {
            "name": str(item.get("name", "")).strip() or "Unknown food",
            "quantity": str(item.get("quantity", "")).strip(),
        }
        for nutrient in NUTRIENTS:
            row[nutrient] = _to_float(item.get(nutrient))
        result.append(row)

    if not result:
        raise ValueError("The AI did not return any usable food items.")
    return result


def image_to_data_url(raw: bytes, max_side: int = 1024) -> str:
    """Shrink an uploaded photo and encode it for the vision model."""
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_food(client, text_model, vision_model, description, image_bytes=None):
    """Ask Groq to estimate nutrients. Uses the vision model when a photo is given."""
    prompt = description.strip() or (
        "Identify the foods in this photo and estimate a typical portion."
    )
    if image_bytes:
        model = vision_model
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes)}},
        ]
    else:
        model = text_model
        content = prompt

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    try:
        response = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs
        )
    except Exception as exc:  # some models reject JSON mode: retry without it
        if "json" not in str(exc).lower():
            raise
        response = client.chat.completions.create(**kwargs)
    return parse_nutrition_json(response.choices[0].message.content)


def get_day_feedback(client, text_model, totals, goals):
    """Short coaching tips comparing the day's totals with the goals."""
    prompt = (
        "Eaten today: " + json.dumps(totals) + "\nDaily goals: " + json.dumps(goals) +
        "\nGive 3 short, friendly, practical tips (bullet points). "
        "This is general wellness guidance, not medical advice."
    )
    response = client.chat.completions.create(
        model=text_model,
        messages=[
            {"role": "system", "content": "You are a supportive nutrition coach."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=400,
    )
    return response.choices[0].message.content


def entries_from_dataframe(df: pd.DataFrame, start_id: int = 1) -> list[dict]:
    """Rebuild log entries from a previously downloaded CSV."""
    missing = [col for col in ENTRY_FIELDS if col not in df.columns]
    if missing:
        raise ValueError("The CSV is missing columns: " + ", ".join(missing))
    entries = []
    for offset, row in enumerate(df[ENTRY_FIELDS].to_dict("records")):
        entry = {"id": start_id + offset}
        entry["date"] = str(row["date"])[:10]
        entry["meal"] = str(row["meal"]) if str(row["meal"]) in MEALS else "Snack"
        entry["name"] = str(row["name"])
        entry["quantity"] = "" if pd.isna(row["quantity"]) else str(row["quantity"])
        for nutrient in NUTRIENTS:
            entry[nutrient] = _to_float(row[nutrient])
        entries.append(entry)
    return entries


def get_default_api_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file: fine when running locally
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("entries", [])
    st.session_state.setdefault("next_id", 1)


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Settings")

    api_key = get_default_api_key()
    if api_key:
        st.sidebar.success("Groq API key loaded from secrets/environment.")
    else:
        api_key = st.sidebar.text_input(
            "Groq API key", type="password", help="Get a free key at console.groq.com"
        )

    day = st.sidebar.date_input("Date to view / log", value=date.today())

    st.sidebar.subheader("Daily goals")
    goals = {
        "calories": st.sidebar.number_input("Calories (kcal)", 500, 6000, 2000, 50),
        "protein_g": st.sidebar.number_input("Protein (g)", 10, 400, 100, 5),
        "carbs_g": st.sidebar.number_input("Carbs (g)", 20, 800, 250, 5),
        "fat_g": st.sidebar.number_input("Fat (g)", 10, 300, 70, 5),
    }

    with st.sidebar.expander("Advanced: models"):
        text_model = st.text_input("Text model", value=DEFAULT_TEXT_MODEL)
        vision_model = st.text_input("Vision (photo) model", value=DEFAULT_VISION_MODEL)

    return {
        "api_key": api_key,
        "day": day,
        "goals": goals,
        "text_model": text_model.strip() or DEFAULT_TEXT_MODEL,
        "vision_model": vision_model.strip() or DEFAULT_VISION_MODEL,
    }


def render_backup_controls():
    st.sidebar.subheader("💾 Save / restore")
    st.sidebar.caption(
        "Your log lives in this browser session only. Download it to keep it, "
        "and restore it later."
    )
    if st.session_state.entries:
        csv = pd.DataFrame(st.session_state.entries)[ENTRY_FIELDS].to_csv(index=False)
        st.sidebar.download_button(
            "Download full log (CSV)", csv, "nutrition_log.csv", "text/csv"
        )
    uploaded = st.sidebar.file_uploader("Restore from CSV", type=["csv"], key="restore")
    if uploaded is not None and st.sidebar.button("Load this CSV"):
        try:
            entries = entries_from_dataframe(pd.read_csv(uploaded))
        except Exception as exc:
            st.sidebar.error(f"Could not load the CSV: {exc}")
        else:
            st.session_state.entries = entries
            st.session_state.next_id = len(entries) + 1
            st.sidebar.success(f"Loaded {len(entries)} items.")


def main():
    st.set_page_config(page_title="Daily Nutrition Tracker", page_icon="🥗", layout="wide")
    init_state()

    st.title("🥗 Daily Nutrition Tracker")
    st.caption(
        "Describe or photograph your meals and AI estimates calories, protein, "
        "carbs and more. Estimates only, not medical advice."
    )

    cfg = render_sidebar()
    render_backup_controls()

    # ---- 1. Log food ----------------------------------------------------
    st.subheader("1. Log what you ate")
    left, right = st.columns([2, 1])
    with left:
        meal = st.selectbox("Meal", MEALS)
        description = st.text_area(
            "What did you eat?",
            placeholder="e.g. 2 scrambled eggs, 2 slices of whole-wheat toast and a banana",
            height=120,
        )
    with right:
        photo = st.file_uploader("…or upload a photo of your meal", type=["png", "jpg", "jpeg", "webp"])
        if photo is not None:
            st.image(photo)

    if st.button("Analyze & add to log", type="primary"):
        if not cfg["api_key"]:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not description.strip() and photo is None:
            st.warning("Describe your meal or upload a photo first.")
        else:
            try:
                client = Groq(api_key=cfg["api_key"])
                with st.spinner("Analyzing your meal…"):
                    items = analyze_food(
                        client,
                        cfg["text_model"],
                        cfg["vision_model"],
                        description,
                        photo.getvalue() if photo is not None else None,
                    )
                for item in items:
                    st.session_state.entries.append(
                        {
                            "id": st.session_state.next_id,
                            "date": cfg["day"].isoformat(),
                            "meal": meal,
                            **item,
                        }
                    )
                    st.session_state.next_id += 1
                st.success(f"Added {len(items)} item(s) to {meal} on {cfg['day']:%d %b %Y}.")
            except Exception as exc:
                st.error(f"Could not analyze this meal: {exc}")

    # ---- 2. Daily summary -----------------------------------------------
    st.subheader(f"2. Summary for {cfg['day']:%A, %d %B %Y}")
    day_entries = [e for e in st.session_state.entries if e["date"] == cfg["day"].isoformat()]

    if not day_entries:
        st.info("Nothing logged for this date yet.")
        return

    df = pd.DataFrame(day_entries)
    totals = {n: round(float(df[n].sum()), 1) for n in NUTRIENTS}

    columns = st.columns(4)
    for col, (key, label, unit) in zip(
        columns,
        [
            ("calories", "Calories", "kcal"),
            ("protein_g", "Protein", "g"),
            ("carbs_g", "Carbs", "g"),
            ("fat_g", "Fat", "g"),
        ],
    ):
        goal = cfg["goals"][key]
        col.metric(label, f"{totals[key]:g} {unit}", f"goal {goal:g} {unit}", delta_color="off")
        col.progress(min(totals[key] / goal, 1.0))
    st.caption(f"Fiber: {totals['fiber_g']:g} g · Sugar: {totals['sugar_g']:g} g")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Macros (grams)**")
        st.bar_chart(
            pd.DataFrame(
                {"grams": [totals["protein_g"], totals["carbs_g"], totals["fat_g"]]},
                index=["Protein", "Carbs", "Fat"],
            )
        )
    with chart_right:
        st.markdown("**Calories by meal**")
        by_meal = df.groupby("meal")["calories"].sum().reindex(MEALS).fillna(0)
        st.bar_chart(by_meal)

    # ---- 3. Food log ----------------------------------------------------
    st.subheader("3. Food log")
    table = df[["meal", "name", "quantity"] + NUTRIENTS].rename(columns=COLUMN_LABELS)
    st.dataframe(table, hide_index=True)

    labels = {e["id"]: f"{e['meal']}: {e['name']} ({e['quantity']})" for e in day_entries}
    to_remove = st.multiselect("Remove items", options=list(labels), format_func=labels.get)
    if st.button("Remove selected") and to_remove:
        st.session_state.entries = [
            e for e in st.session_state.entries if e["id"] not in to_remove
        ]
        st.rerun()

    # ---- 4. AI coach ----------------------------------------------------
    st.subheader("4. AI coach")
    if st.button("Get tips for today"):
        if not cfg["api_key"]:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            try:
                with st.spinner("Thinking…"):
                    tips = get_day_feedback(
                        Groq(api_key=cfg["api_key"]), cfg["text_model"], totals, cfg["goals"]
                    )
                st.markdown(tips)
            except Exception as exc:
                st.error(f"Could not get tips: {exc}")


if __name__ == "__main__":
    main()
