# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import re
import io

# ----------------- CONFIG -----------------
USERNAME = "demo"
PASSWORD = "Kapish@174"
MODEL_PATH = "trained_model.h5"   # ensure this file is in project folder
IMG_SIZE = (128, 128)             # same as original training / your working app
ACCENT = "#0f766e"                # accent color for styling
PAGE_BG = "#f6fbfb"
TEXT_COLOR = "#0b2545"
# ------------------------------------------

# ---- New imports for SQL-backed auth ----
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, select
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import bcrypt
from sqlalchemy.engine.url import make_url
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Database URL fallback to sqlite file in project root
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///users.db")

# ---- Lightweight page CSS (single-file) ----
page_css = f"""
<style>
:root {{
  --accent: {ACCENT};
  --page-bg: {PAGE_BG};
  --text-color: {TEXT_COLOR};
}}
html, body, .stApp {{
  background: var(--page-bg);
  color: var(--text-color);
}}
.header {{
  padding: 24px 18px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(15,118,110,0.95) 0%, rgba(3,105,161,0.9) 100%);
  color: white;
  margin-bottom: 12px;
}}
.header h1 {{ margin: 0; font-size: 26px; }}
.card {{ background: white; padding: 14px; border-radius: 10px; box-shadow: 0 6px 18px rgba(11,37,69,0.04); margin-bottom: 14px; }}
.stButton>button {{ background: var(--accent); color: white; border-radius: 8px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, white, #f7fbfb); border-radius: 10px; padding: 12px; }}
.footer {{ padding: 10px 0; color: #334155; font-size: 13px; opacity: 0.9; }}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# ---------------- Authentication state init ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    # keep the same type as original (string) to avoid changing other parts
    st.session_state.user = ""

# ---------------- Database helpers ----------------
def get_engine(db_url=DATABASE_URL):
    try:
        _ = make_url(db_url)  # validate
        engine = create_engine(db_url, future=True)
        return engine
    except Exception as e:
        st.error(f"Invalid DATABASE_URL: {e}")
        raise

def ensure_users_table(engine):
    meta = MetaData()
    users = Table(
        "users", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("email", String(255), unique=True, nullable=False),
        Column("password_hash", String(255), nullable=False),
        Column("display_name", String(255), nullable=True),
    )
    meta.create_all(engine)
    return users

def register_user(engine, users_table, email, password, display_name=None):
    """
    Registers a user. Truncates password bytes to 72 bytes before bcrypt hashing.
    Stores the resulting hash (bytes) decoded to utf-8 for storage in a TEXT column.
    """
    email_norm = (email or "").strip().lower()
    if not email_norm or not password:
        return False, "Email and password are required."
    try:
        # bcrypt supports up to 72 bytes — truncate to 72 bytes
        pw_bytes = password.encode("utf-8")[:72]
        hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt())  # bytes
        hashed_str = hashed.decode("utf-8")  # store as string
        ins = users_table.insert().values(email=email_norm, password_hash=hashed_str, display_name=display_name)
        with engine.begin() as conn:
            conn.execute(ins)
        return True, None
    except IntegrityError:
        return False, "An account with this email already exists."
    except SQLAlchemyError as e:
        return False, str(e.__dict__.get("orig", e))
    except Exception as e:
        return False, str(e)

def verify_user(engine, users_table, email, password):
    """
    Verifies a user by truncating the provided password to 72 bytes and passing
    to bcrypt.checkpw against the stored hash.
    Returns (True, email) on success or (False, error_message).
    """
    email_norm = (email or "").strip().lower()
    sel = select(users_table).where(users_table.c.email == email_norm)
    try:
        with engine.connect() as conn:
            row = conn.execute(sel).first()
            if not row:
                return False, "No account found with that email."
            stored_hash = row.password_hash  # stored as string
            if isinstance(stored_hash, str):
                stored_hash_bytes = stored_hash.encode("utf-8")
            else:
                stored_hash_bytes = stored_hash
            pw_bytes = (password or "").encode("utf-8")[:72]
            if bcrypt.checkpw(pw_bytes, stored_hash_bytes):
                # return user email as string (to match original session expectations)
                user_email = row.email
                return True, user_email
            else:
                return False, "Incorrect password."
    except SQLAlchemyError as e:
        return False, str(e.__dict__.get("orig", e))
    except Exception as e:
        return False, str(e)

# Initialize DB engine and users table (if DB available)
_engine = None
_users_table = None
try:
    _engine = get_engine()
    _users_table = ensure_users_table(_engine)
except Exception:
    _engine = None
    _users_table = None
    st.warning("User database not initialized. Auth will be disabled if DB issues occur.")

# ---------------- Validation helpers ----------------
EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
def is_valid_email(email: str):
    return bool(EMAIL_REGEX.match((email or "").strip()))

def password_strength(password: str):
    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    return score

# ---------------- Login widget (SQL-backed, includes Sign up) ----------------
def login_widget():
    """Show login widget — returns True once logged in.
    This keeps the original UI texts for heading/card but replaces the auth backend.
    """
    if st.session_state.logged_in:
        return True

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔐 Login to continue")

    # Provide a simple toggle between Login and Sign up
    mode = st.radio("", ("Login", "Sign up"), index=0, horizontal=True)

    if mode == "Login":
        # Keep layout similar to original: two columns
        col1, col2 = st.columns([2, 1])
        with col1:
            user = st.text_input("Email", key="login_user")
            pwd = st.text_input("Password", type="password", key="login_pwd")
        with col2:
            st.write(" ")
            if st.button("Login", key="login_btn"):
                # If DB not available, fall back to original demo creds
                if _engine is None or _users_table is None:
                    # fallback behavior: allow original demo
                    if (user or "").strip() == USERNAME and (pwd or "") == PASSWORD:
                        st.session_state.logged_in = True
                        st.session_state.user = user.strip()
                        st.success("Login successful — loading app...")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password, and database unavailable.")
                else:
                    ok, res = verify_user(_engine, _users_table, user, pwd)
                    if ok:
                        st.session_state.logged_in = True
                        # keep same type as original: string username/email
                        st.session_state.user = res
                        st.success("Login successful — loading app...")
                        st.rerun()
                    else:
                        st.error(res)
    else:
        # Sign up mode
        st.markdown("Create an account to use the Disease Recognition tool.")
        # Use a small form for sign up
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            display_name = st.text_input("Display name (optional)", key="signup_name")
            pwd = st.text_input("Password", type="password", key="signup_pwd")
            pwd2 = st.text_input("Confirm Password", type="password", key="signup_pwd2")
            submitted = st.form_submit_button("Create account")
            if submitted:
                if _engine is None or _users_table is None:
                    st.error("Database not available. Cannot create account.")
                elif not email or not pwd:
                    st.error("Please provide email and password.")
                elif not is_valid_email(email):
                    st.error("Please enter a valid email address.")
                elif pwd != pwd2:
                    st.error("Passwords do not match.")
                else:
                    score = password_strength(pwd)
                    if score < 2:
                        st.error("Weak password — use at least 8 chars, mixed case and numbers.")
                    else:
                        ok, err = register_user(_engine, _users_table, email, pwd, display_name)
                        if ok:
                            st.success("Account created. You can now switch to Login and sign in.")
                        else:
                            st.error(f"Could not create account: {err}")

    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.logged_in

# ---------------- Model loader (cached) ----------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

# ---------------- Prediction helper (matches original behavior) ----------------
def predict_from_image_bytes(image_bytes, model):
    """
    image_bytes: raw bytes of the uploaded file
    Uses same pipeline your original app used:
      - open with PIL
      - resize to IMG_SIZE
      - convert to array via keras img_to_array style (values 0..255)
      - batchify
      - model.predict
      - return raw preds
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(pil_img)  # dtype float32, values 0..255
    arr = np.expand_dims(arr, axis=0)  # shape (1,H,W,3)
    preds = model.predict(arr)
    return preds

# ---------------- Class names (unchanged) ----------------
CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight',
    'Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch',
    'Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# ---------------- Header ----------------
st.markdown(
    """
    <div class="header">
      <h1>🌿 Plant Disease Recognition System</h1>
      <p style="opacity:0.95">Upload leaf images and get model predictions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# If not logged in, show login and stop rendering the rest
if not login_widget():
    st.stop()

# Load model now (cached)
model, model_err = load_model()
if model_err:
    st.warning(model_err)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## Dashboard")
    st.markdown("Choose a page to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(" ")
    # Keep the same display logic as original
    st.markdown(f"**Logged in as:** {st.session_state.user or USERNAME}")
    page = st.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Logout"])
    st.markdown("---")
    st.markdown("<div style='font-size:12px;color:#64748b'>Tip: upload clear leaf photos (close-up, plain background)</div>", unsafe_allow_html=True)

if page == "Logout":
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.rerun()

# ---------------- Pages (kept your original content exactly) ----------------
if page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    image_path = "home_page.jpeg"
    if os.path.exists(image_path):
        try:
            st.image(image_path, use_container_width=True)
        except Exception:
            st.warning("Home image could not be displayed (file may be corrupted).")
    else:
        st.warning("Home image not found. Place 'home_page.jpeg' in the project folder.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
                
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    #### Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Disease Recognition":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Disease Recognition")

    uploaded_file = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    image_bytes = None
    if uploaded_file is not None:
        try:
            # Read bytes once and use for display and prediction to avoid pointer issues
            image_bytes = uploaded_file.read()
            pil_for_display = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(pil_for_display, use_container_width=True)
        except Exception:
            st.error("Uploaded file could not be displayed as an image.")

    # Place Predict button in right column similar to before
    col1, col2 = st.columns([2, 1])
    with col2:
        st.write(" ")
        if st.button("Predict"):
            if image_bytes is None:
                st.error("⚠ Please upload an image first!")
            else:
                if model is None:
                    st.error("Model not loaded. Place trained_model.h5 in project folder.")
                else:
                    with st.spinner("Running model..."):
                        try:
                            preds = predict_from_image_bytes(image_bytes, model)
                            # MATCH ORIGINAL: use raw prediction argmax (no softmax, no scaling)
                            if preds is None:
                                st.error("Model returned no output.")
                            else:
                                # raw argmax to match original behavior
                                idx = int(np.argmax(preds[0]))
                                label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                                st.success(f"Model predicts: **{label}**")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">Made with ❤️ • Demo app • Keep credentials safe for production</div>', unsafe_allow_html=True)
