import streamlit as st
import pandas as pd
import joblib
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
import random
import string
import time

# ===================== BASIC CONFIG =====================

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

DB_PATH = "auth_logs.db"
APP_TITLE = "Smart House Price Predictor"
PASSWORD_SALT = "some_static_salt_change_me"  # demo only


# ===================== DB HELPERS =======================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ts TEXT NOT NULL,
                price_lacs REAL NOT NULL,
                city TEXT,
                area TEXT,
                payload TEXT
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    data = (PASSWORD_SALT + password).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def create_user(username: str, email: str, password: str):
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, hash_password(password), datetime.utcnow().isoformat()),
            )
            conn.commit()
        return True, "Account created successfully. You can login now."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def authenticate_user(username: str, password: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
    if not row:
        return False
    return row["password_hash"] == hash_password(password)


def get_user_by_email(username: str, email: str):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE username = ? AND email = ?",
            (username, email),
        )
        return cur.fetchone()


def reset_user_password(username: str, new_password: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), username),
        )
        conn.commit()


def log_prediction(username: str, price_lacs: float, city: str, area: str, payload: dict):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_logs (username, ts, price_lacs, city, area, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (
                username,
                datetime.utcnow().isoformat(),
                price_lacs,
                city,
                area,
                json.dumps(payload),
            ),
        )
        conn.commit()


def get_user_logs(username: str):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT ts, price_lacs, city, area, payload FROM audit_logs "
            "WHERE username = ? ORDER BY ts DESC LIMIT 50",
            (username,),
        )
        return cur.fetchall()


# ===================== MODEL LOADING ====================

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


# ===================== UTIL HELPERS =====================

def generate_temp_password(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=length))


def format_inr(amount: float) -> str:
    """
    Format number like Indian rupee style.
    Example: 12345678 -> 1,23,45,678
    """
    s = f"{int(round(amount))}"
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    parts.reverse()
    return ",".join(parts) + "," + last3


# ===================== TOP INFO BAR =====================

def show_top_info_bar():
    """Top info bar with model summary + time (with seconds)."""
    now_str = datetime.now().strftime("%d %b %Y · %I:%M:%S %p")  # includes seconds

    st.markdown(
        f"""
        <style>
        .top-info-container {{
            max-width: 1100px;
            margin: 1.0rem auto 0.8rem auto;
            padding: 0.8rem 1rem;
            border-radius: 20px;
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.35);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            transition: all 0.25s ease-out;
        }}
        .top-info-container:hover {{
            box-shadow: 0 18px 45px rgba(15,23,42,0.9);
            transform: translateY(-2px);
        }}
        .top-pill {{
            padding: 0.45rem 0.9rem;
            border-radius: 12px;
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.6);
            font-size: 0.8rem;
            color: #e5e7eb;
        }}
        .clock-pill {{
            padding: 0.45rem 0.9rem;
            border-radius: 12px;
            background: rgba(15,23,42,0.85);
            border: 1px solid rgba(148,163,184,0.6);
            font-size: 0.8rem;
            color: #e5e7eb;
            white-space: nowrap;
        }}
        </style>

        <div class="top-info-container">
            <div class="top-pill">
                🤖 <b>About the model:</b>
                Random Forest Regressor trained on Indian housing data to estimate price in <b>₹ Lacs</b>
            </div>
            <div class="clock-pill">
                🕒 {now_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== SESSION INIT =====================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"  # login | register | forgot

init_db()


# ===================== GLOBAL STYLES ====================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1d4ed8 0, #020617 55%, #020617 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main-wrapper {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .card {
        background: rgba(15,23,42,0.96);
        border-radius: 22px;
        padding: 22px 24px;
        border: 1px solid rgba(148,163,184,0.35);
        box-shadow: 0 22px 55px rgba(15,23,42,0.9);
        transition: all 0.25s ease-out;
    }
    .card:hover {
        box-shadow: 0 24px 60px rgba(15,23,42,0.95);
        transform: translateY(-2px);
    }
    .header-card {
        background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(30,64,175,0.9));
        border-radius: 24px;
        padding: 22px 26px;
        border: 1px solid rgba(129,140,248,0.7);
        box-shadow: 0 25px 60px rgba(15,23,42,0.95);
        margin-bottom: 1.2rem;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #e5e7eb;
        opacity: 0.9;
    }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.7);
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #e5e7eb;
        margin-bottom: 0.35rem;
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #a5b4fc;
        margin-bottom: 0.25rem;
    }
    .section-sub {
        font-size: 0.86rem;
        color: #9ca3af;
        margin-bottom: 0.7rem;
    }
    .metric-box {
        padding: 1.6rem 1.5rem;
        border-radius: 22px;
        background: radial-gradient(circle at top left, rgba(34,197,94,0.2), rgba(15,23,42,0.98));
        border: 1px solid rgba(34,197,94,0.75);
        box-shadow: 0 24px 60px rgba(22,163,74,0.55);
        text-align: left;
    }
    .metric-main {
        font-size: 2.4rem;
        font-weight: 750;
        color: #bbf7d0;
        margin-bottom: 0.35rem;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    .metric-note {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.35rem;
    }
    .badge-soft {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(30,64,175,0.7);
        color: #e5e7eb;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .login-card {
        max-width: 450px;
        margin: 8vh auto 4vh auto;
        animation: floatUp 0.6s ease-out;
    }
    @keyframes floatUp {
        from { transform: translateY(12px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Glow effect for Streamlit buttons */
    .stButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(148,163,184,0.6) !important;
        background: linear-gradient(135deg, #1e293b, #020617) !important;
        color: #e5e7eb !important;
        padding: 0.35rem 1.1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease-out !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 0 1px #38bdf8, 0 0 22px rgba(59,130,246,0.7);
        transform: translateY(-1px);
    }

    /* Smooth transition for inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        transition: box-shadow 0.18s ease-out, border-color 0.18s ease-out;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div:hover > div {
        box-shadow: 0 0 0 1px #38bdf8;
        border-color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================== AUTH VIEWS =======================

def show_auth_switcher():
    cols = st.columns(3)
    with cols[0]:
        if st.button("🔐 Login", use_container_width=True):
            st.session_state.auth_view = "login"
    with cols[1]:
        if st.button("🆕 Register", use_container_width=True):
            st.session_state.auth_view = "register"
    with cols[2]:
        if st.button("❓ Forgot password", use_container_width=True):
            st.session_state.auth_view = "forgot"


def login_view():
    show_top_info_bar()

    st.markdown("<div class='login-card card'>", unsafe_allow_html=True)
    show_auth_switcher()
    st.markdown(
        f"<h2 style='margin-top:0.8rem; margin-bottom:0.2rem;'>🔐 Sign in to {APP_TITLE}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.88rem; color:#9ca3af;'>Use your account credentials. If you are new, register first.</p>",
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.8rem; color:#9ca3af; margin-bottom:0.3rem;'>Or continue with</p>",
        unsafe_allow_html=True,
    )
    if st.button("🔵 Continue with Google (demo)", use_container_width=True):
        st.info("Google login is a visual demo only in this project.")

    st.markdown(
        "<p style='font-size:0.8rem; color:#6b7280; margin-top:0.8rem;'>"
        "Forgot your password? Click <b>Forgot password</b> above.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def register_view():
    show_top_info_bar()

    st.markdown("<div class='login-card card'>", unsafe_allow_html=True)
    show_auth_switcher()
    st.markdown(
        "<h2 style='margin-top:0.8rem; margin-bottom:0.2rem;'>🆕 Create a new account</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.88rem; color:#9ca3af;'>Choose a username, enter email and a strong password.</p>",
        unsafe_allow_html=True,
    )

    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Pick a unique username")
        email = st.text_input("Email", placeholder="yourname@example.com")
        pw1 = st.text_input("Password", type="password", placeholder="Create a strong password")
        pw2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        if not username or not email or not pw1:
            st.error("Please fill all fields.")
        elif pw1 != pw2:
            st.error("Passwords do not match.")
        else:
            ok, msg = create_user(username, email, pw1)
            if ok:
                st.success(msg)
                st.info("You can now click Login and sign in with your new account.")
            else:
                st.error(msg)

    st.markdown("</div>", unsafe_allow_html=True)


def forgot_password_view():
    show_top_info_bar()

    st.markdown("<div class='login-card card'>", unsafe_allow_html=True)
    show_auth_switcher()
    st.markdown(
        "<h2 style='margin-top:0.8rem; margin-bottom:0.2rem;'>❓ Reset your password</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.88rem; color:#9ca3af;'>"
        "No OTP in this demo. Confirm your username & registered email, then choose a new password."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("reset_form"):
        username = st.text_input("Username")
        email = st.text_input("Registered email")
        new_pw1 = st.text_input("New password", type="password")
        new_pw2 = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password")

    if submitted:
        if not username or not email or not new_pw1:
            st.error("Please fill all fields.")
        elif new_pw1 != new_pw2:
            st.error("Passwords do not match.")
        else:
            user = get_user_by_email(username, email)
            if not user:
                st.error("No user found with that username & email.")
            else:
                reset_user_password(username, new_pw1)
                st.success("Password updated successfully. You can now login.")
                st.session_state.auth_view = "login"

    st.markdown("</div>", unsafe_allow_html=True)


# ===================== MAIN APP (AFTER LOGIN) ==========

def main_app():
    model = load_model()

    # Sidebar with compact toggle
    with st.sidebar:
        compact = st.checkbox("Compact sidebar", value=False)
        st.markdown("### 👋 Welcome")
        st.write(f"Logged in as **{st.session_state.username}**")
        st.markdown("---")
        if not compact:
            st.markdown("### ℹ️ About this app")
            st.write(
                """
                • ML stack: **Streamlit + scikit-learn**  
                • Model: Random Forest Regressor  
                • Target: Price in **₹ Lacs**  
                • Auth: SQLite users, custom reset, audit logs
                """
            )
            st.markdown("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        st.caption("Portfolio Project · ML · Real Estate")

    show_top_info_bar()

    st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="header-card">
            <div class="pill">Machine Learning · Real Estate</div>
            <div class="header-title">🏠 Smart House Price Predictor</div>
            <div class="header-sub">
                Estimate the market value of residential properties using your trained ML model.
                Enter details like city, locality, area and configuration to get an instant price in <b>₹ Lacs</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Property details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">City / country / area are shown in the summary. '
        'The model uses configuration, size and coordinates for prediction.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        loc1, loc2, loc3 = st.columns([1.4, 1.2, 1.4])
        with loc1:
            city = st.selectbox(
                "City",
                ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai", "Pune", "Other"],
                index=2,
            )
        with loc2:
            country = st.selectbox("Country", ["India", "Other"], index=0)
        with loc3:
            area_name = st.text_input("Area / Locality", value="Whitefield")

        col1, col2 = st.columns(2)
        with col1:
            posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
            bhk_or_rk = st.selectbox("Type", ["BHK", "RK"])
            bhk_no = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)
            square_ft = st.number_input("Area (Square Ft)", min_value=100, max_value=10000, value=1000, step=50)

        with col2:
            under_construction = st.selectbox("Under Construction?", ["No", "Yes"])
            rera = st.selectbox("RERA Approved?", ["No", "Yes"])
            ready_to_move = st.selectbox("Ready to Move?", ["No", "Yes"])
            resale = st.selectbox("Resale?", ["No", "Yes"])
            longitude = st.number_input("Longitude", value=77.59, format="%.5f")
            latitude = st.number_input("Latitude", value=12.97, format="%.5f")

        submitted = st.form_submit_button("🔮 Predict Price")

    def yn_to_int(x: str) -> int:
        return 1 if x == "Yes" else 0

    if submitted:
         posted_by_map = {"Owner": 0, "Dealer": 1, "Builder": 2}
         bhk_or_rk_map = {"BHK": 0, "RK": 1}
    feature_row = {
            "POSTED_BY": posted_by_map[posted_by],
            "UNDER_CONSTRUCTION": yn_to_int(under_construction),
            "RERA": yn_to_int(rera),
            "BHK_NO.": bhk_no,
            "BHK_OR_RK": bhk_or_rk_map[bhk_or_rk],
            "SQUARE_FT": square_ft,
            "READY_TO_MOVE": yn_to_int(ready_to_move),
            "RESALE": yn_to_int(resale),
            "LONGITUDE": longitude,
            "LATITUDE": latitude,
        }
    input_df = pd.DataFrame([feature_row])
    price_lacs = float(model.predict(input_df)[0])
    price_inr = price_lacs * 1_00_000
    input_df = pd.DataFrame([feature_row])
    price_lacs = float(model.predict(input_df)[0])
    price_inr = price_lacs * 1_00_000

        # log this prediction
    log_prediction(
            username=st.session_state.username,
            price_lacs=price_lacs,
            city=city,
            area=area_name,
            payload=feature_row,
        )

    c1, c2 = st.columns([1.7, 1.3])

        # Animated price reveal
    with c1:
            metric_placeholder = st.empty()
            steps = 25
            for i in range(steps + 1):
                val = price_lacs * i / steps
                metric_placeholder.markdown(
                    f"""
                    <div class="metric-box">
                        <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.09em; color:#bbf7d0; margin-bottom:0.35rem;">
                            Estimated Market Value
                        </div>
                        <div class="metric-main">₹ {val:,.2f} Lacs</div>
                        <div class="metric-sub">
                            ≈ ₹ {format_inr(price_inr)} <span style="opacity:0.8;">(Indian Rupees)</span>
                        </div>
                        <div class="metric-note">
                            This is an approximate valuation produced by a machine learning model.
                            Cross-check with recent deals in the same locality.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.02)

    with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Property snapshot</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="badge-soft">📍 {area_name}, {city}</span>
                <span class="badge-soft">🌎 {country}</span><br/>
                <span class="badge-soft">🛏 {bhk_no} {bhk_or_rk}</span>
                <span class="badge-soft">📐 {square_ft:.0f} sq ft</span>
                """,
                unsafe_allow_html=True,
            )
            tags = []
            if yn_to_int(ready_to_move):
                tags.append("Ready to move")
            else:
                tags.append("Under construction")
            if yn_to_int(resale):
                tags.append("Resale")
            if yn_to_int(rera):
                tags.append("RERA approved")
            if tags:
                st.markdown(
                    "<br/>" + " ".join(
                        [f'<span class="badge-soft">✅ {t}</span>' for t in tags]
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.7rem;">
                    Coordinates: <b>{latitude:.5f}</b>, <b>{longitude:.5f}</b><br/>
                    Posted by: <b>{posted_by}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.success("Prediction generated successfully ✅")

    # ============ History + Charts ============

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.expander("📊 Your recent price predictions (history & charts)"):
        rows = get_user_logs(st.session_state.username)
        if not rows:
            st.write("No predictions logged yet. Make a few predictions to see trends over time.")
        else:
            records = []
            for r in rows:
                payload = json.loads(r["payload"])
                ts_utc = datetime.fromisoformat(r["ts"])
                ts_ist = ts_utc + timedelta(hours=5, minutes=30)
                records.append(
                    {
                        "Time (IST)": ts_ist,
                        "Price (Lacs)": float(r["price_lacs"]),
                        "City": r["city"],
                        "Area": r["area"],
                        "BHK": payload.get("BHK_NO."),
                        "Sq Ft": payload.get("SQUARE_FT"),
                    }
                )
            df = pd.DataFrame(records).sort_values("Time (IST)")

            st.dataframe(df, use_container_width=True)

            # Download as CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download history as CSV",
                csv,
                file_name="house_price_history.csv",
                mime="text/csv",
            )

            if len(df) > 1:
                st.markdown("#### 📈 Price over time")
                chart_df = df[["Time (IST)", "Price (Lacs)"]].set_index("Time (IST)")
                st.line_chart(chart_df)
            else:
                st.info("Add more predictions to see the price trend over time 📈")

            st.markdown("#### 🏙️ Average price by city (in your history)")
            city_stats = df.groupby("City", dropna=True)["Price (Lacs)"].mean().reset_index()
            if not city_stats.empty and len(city_stats) > 0:
                city_stats = city_stats.rename(columns={"Price (Lacs)": "Avg Price (Lacs)"})
                st.bar_chart(city_stats.set_index("City"))
            else:
                st.info("Make predictions for different cities to compare average prices 🏙️")

    st.markdown(
        "<hr style='border-color:rgba(55,65,81,0.7); margin-top:1.8rem; margin-bottom:0.4rem;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.78rem; color:#6b7280;'>"
        "Built with ❤️ by Nakul Vashishtha · Smart House Price Predictor · Streamlit · scikit-learn · SQLite auth + audit logs."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ===================== ROUTER ===========================

if not st.session_state.logged_in:
    if st.session_state.auth_view == "login":
        login_view()
    elif st.session_state.auth_view == "register":
        register_view()
    else:
        forgot_password_view()
else:
    main_app()


