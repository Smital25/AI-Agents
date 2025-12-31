from __future__ import annotations

# ---------------- STREAMLIT MUST COME FIRST ----------------
import streamlit as st

st.set_page_config(
    page_title="TeamCollab • AI-Ready",
    layout="centered",
    page_icon="🤖"
)

# ---------------- STANDARD IMPORTS ----------------
import datetime
import hashlib
import re
import uuid
import secrets
import string
import os
import logging
from typing import Any, Optional
import ollama


import requests
from pymongo import MongoClient, errors


# ---------------- basic config ----------------

logging.basicConfig(level=logging.INFO)

EMAIL_USER = os.getenv("EMAIL_USER", "shostelmanagement@gmail.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "ehbplmequzgyhzck")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
progress_col = db["progress"]
resources_col = db["resources"]
chat_col = db["chat"]
objectives_col = db["project_objectives"]

# try to create unique index for email
try:
    users_col.create_index("email", unique=True)
except Exception:
    pass

# ---------------- helpers ----------------
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def generate_password(length: int = 10) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def valid_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email))


def safe_get(obj: Any, key: str, default=None):
    return obj[key] if isinstance(obj, dict) and key in obj else default


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    if not to_email:
        return False
    # Minimal: try sending if credentials present
    if not EMAIL_USER or not EMAIL_PASS:
        return False
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, to_email, subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, to_email, msg.as_string())
        return True
    except Exception:
        return False


# ---------------- project suggestion via LLM ----------------


def suggest_projects(domain, count=5):

    ideas = []

    for i in range(count):

        prompt = f"""
Generate EXACTLY ONE academic project idea for the domain "{domain}".

STRICT OUTPUT FORMAT (FOLLOW EXACTLY):

---
TITLE: <Project Title>

INTRODUCTION:
<2 meaningful sentences>

WHAT IT DOES:
<1 clear sentence>

ACADEMIC BENEFIT:
<1 clear sentence>
---

RULES:
- Output ONLY one project.
- No numbering.
- No extra explanation.
"""
        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                }
            ).json()

            text = safe_get(res.get("message", {}), "content", "") or ""

            blk = text.strip()

            # Extract title
            match = re.search(r"TITLE:\s*(.*)", blk)
            title = match.group(1).strip() if match else f"{domain} Project {i+1}"

            ideas.append({
                "title": title,
                "description": blk
            })

        except Exception:
            ideas.append({
                "title": f"{domain} Project {i+1}",
                "description": (
                    "INTRODUCTION:\nBackup introduction.\nBackup intro line.\n"
                    "WHAT IT DOES:\nBackup function.\n"
                    "ACADEMIC BENEFIT:\nBackup benefit."
                )
            })

    return ideas


# ---------------- session defaults ----------------
if "active_user" not in st.session_state:
    st.session_state["active_user"] = None
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "active_project" not in st.session_state:
    st.session_state["active_project"] = None
if "suggested_projects" not in st.session_state:
    st.session_state["suggested_projects"] = []

# ---------------- UI header ----------------
st.markdown("<h1 style='text-align:center;'>🤖 TeamCollab</h1>"
            "<p style='text-align:center;color:gray;'>AI-powered Academic Project Manager</p><hr>",
            unsafe_allow_html=True)

DOMAINS = ["-- Select Domain --",
           "AI & Machine Learning", "Web Development", "Mobile App Development",
           "Cybersecurity", "Data Science", "IoT", "Cloud Computing", "Blockchain",
           "AR/VR", "Robotics", "NLP", "Computer Vision", "Game Development",
           "Big Data", "Embedded Systems", "Quantum Computing", "DevOps",
           "HealthTech", "FinTech", "EdTech", "Other"]

menu = st.sidebar.radio("Navigation", ["🏠 Register (Lead)", "🔐 Login", "📊 Dashboard"])

# ---------------- Register (Lead) ----------------
if menu == "🏠 Register (Lead)":
    st.subheader("👑 Register as Team Lead")
    with st.form("lead_info"):
        team_name = st.text_input("Team Name")
        lead_name = st.text_input("Full Name")
        lead_email = st.text_input("Email")
        domain = st.selectbox("Select Project Domain", DOMAINS)
        col1, col2 = st.columns(2)
        with col1:
            generate_btn = st.form_submit_button("✨ Generate Project Ideas")
        with col2:
            clear_btn = st.form_submit_button("🧹 Clear Suggestions")

    if generate_btn:
        if domain == "-- Select Domain --":
            st.error("Please select domain first.")
        else:
            st.session_state.suggested_projects = suggest_projects(domain)
            st.rerun()

    if clear_btn:
        st.session_state.suggested_projects = []
        st.rerun()

    if st.session_state.suggested_projects:
        st.markdown("### 💡 Suggested Project Ideas")
        for i, idea in enumerate(st.session_state.suggested_projects, 1):
            st.markdown(f"**{i}. {idea['title']}**")
            st.markdown(idea["description"])
            st.markdown("---")

    with st.form("create_project"):
        project_title_custom = ""
        use_suggested = False
        suggested_choice = None
        if st.session_state.suggested_projects:
            mode = st.radio("Title source:", ["Type my own", "Choose suggestion"])
            if mode == "Choose suggestion":
                idx = st.selectbox("Pick suggestion", range(len(st.session_state.suggested_projects)),
                                   format_func=lambda i: st.session_state.suggested_projects[i]["title"])
                suggested_choice = st.session_state.suggested_projects[idx]["title"]
                use_suggested = True
            else:
                project_title_custom = st.text_input("Project Title")
        else:
            project_title_custom = st.text_input("Project Title (your idea)")

        project_description = st.text_area("Project Description (optional)")
        submit_create = st.form_submit_button("🚀 Create Project")

    if submit_create:
        if not team_name.strip() or not lead_name.strip() or not valid_email(lead_email):
            st.error("Team name, full name and a valid email are required.")
        elif domain == "-- Select Domain --":
            st.error("Please select a domain.")
        else:
            project_title = suggested_choice if use_suggested else project_title_custom.strip()
            if not project_title:
                st.error("Project title required.")
            elif users_col.find_one({"email": lead_email.lower()}):
                st.error("Email already registered. Login instead.")
            else:
                pid = str(uuid.uuid4())
                tid = str(uuid.uuid4())
                pwd = generate_password()
                user_doc = {
                    "user_id": str(uuid.uuid4()),
                    "name": lead_name,
                    "email": lead_email.lower(),
                    "password": hash_password(pwd),
                    "status": "accepted",
                    "projects": [{
                        "project_id": pid,
                        "team_id": tid,
                        "role": "lead",
                        "domain": domain,
                        "project_name": project_title,
                    }],
                    "created_at": datetime.datetime.utcnow()
                }
                team_doc = {
                    "team_id": tid,
                    "project_id": pid,
                    "project_name": project_title,
                    "team_name": team_name,
                    "domain": domain,
                    "lead_email": lead_email.lower(),
                    "created_at": datetime.datetime.utcnow()
                }
                try:
                    users_col.insert_one(user_doc)
                    teams_col.insert_one(team_doc)
                except errors.DuplicateKeyError:
                    st.error("Email already exists.")
                    st.stop()

                send_email(
        lead_email,
        "Your TeamCollab Login",
        f"""
        <div style='font-family: Arial, sans-serif; max-width: 650px; margin:auto; line-height:1.6;'>

            <h2 style='color:#2B579A;'>Welcome to TeamCollab</h2>

            <p>Dear Project Lead,</p>

            <p>
                Your project {project_title} has been successfully created on <strong>TeamCollab</strong> with team name as {team_name} 
                and your selected domain is {domain} .  
                Below are your login credentials to access your Lead Dashboard.
            </p>

            <div style='background:#f3f6fb; padding:15px; border-left:4px solid #2B579A; border-radius: 6px; margin-bottom:15px;'>
                <p style='margin:0;'><strong>Email:</strong> {lead_email}</p>
                <p style='margin:0;'><strong>Password:</strong> <b>{pwd}</b></p>
            </div>

            <p>
                You can log in using the link below:
                <br><br>
                🔗 <a href='http://localhost:8501' style='color:#2B579A; font-size:16px; font-weight:bold;'>
                    Open TeamCollab Dashboard
                </a>
            </p>

            <p>
                For security reasons, please update your password after logging in.
            </p>

            <p>If you did not request this project setup, you may safely ignore this email.</p>

            <p>
                Regards,<br>
                <strong>TeamCollab System</strong>
            </p>

        </div>
        """
    )

    st.success("Project created. Check email if SMTP configured.")
    st.session_state.suggested_projects = []
    st.rerun()


# ---------------- Login ----------------
elif menu == "🔐 Login":
    st.subheader("🔑 Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        email_norm = (email or "").strip().lower()
        if not email_norm or not pwd:
            st.error("Email and password required.")
        else:
            db_user = users_col.find_one({"email": email_norm})
            if not db_user:
                st.error("User not found.")
            elif hash_password(pwd) != db_user.get("password", ""):
                st.error("Incorrect password.")
            else:
                # Load user's projects
                projects = db_user.get("projects", []) or []

                # Default active project
                active_proj = projects[0] if projects else None

                # ------------- FIXED SESSION VARIABLES -------------
                st.session_state["logged_in"] = True
                st.session_state["user"] = db_user            # <--- IMPORTANT
                st.session_state["active_user"] = db_user     # for lead dashboard
                st.session_state["project"] = active_proj     # <--- IMPORTANT
                st.session_state["active_project"] = active_proj
                # --------------------------------------------------

                st.success("Login successful.")
                st.rerun()

# ---------------- Dashboard router ----------------
elif menu == "📊 Dashboard":
    if not st.session_state.get("logged_in"):
        st.warning("Please log in first.")
        st.stop()

    active_user = st.session_state.get("active_user", None)
    if not isinstance(active_user, dict):
        st.warning("Session invalid. Please login again.")
        st.stop()

    projects = active_user.get("projects", []) or []
    if not projects:
        st.info("You are not part of any project.")
        st.stop()

    options = [f'{p.get("project_name","(untitled)")} | {p.get("domain","(no domain)")} | {p.get("role","member")}'
               for p in projects]
    selected_index = st.selectbox("Select Project", range(len(options)), format_func=lambda i: options[i])
    proj = projects[selected_index] if 0 <= selected_index < len(projects) else projects[0]

    if not isinstance(proj, dict):
        st.error("Project data corrupted.")
        st.stop()

    role = (proj.get("role") or "member").lower()

    # Save last_active_project in DB (best-effort)
    try:
        users_col.update_one({"email": active_user.get("email")}, {"$set": {"last_active_project": proj.get("project_id")}})
    except Exception:
        pass

    # Save session keys BEFORE switching
    st.session_state["active_user"] = active_user
    st.session_state["active_project"] = proj
    st.session_state["active_role"] = role
    st.session_state["logged_in"] = True

    # Route to the correct page inside the pages folder
    if role == "lead":
        st.success("Opening Lead Dashboard…")
        st.switch_page("pages/lead_dashboard.py")
    elif role == "mentor":
        st.success("Opening Mentor Dashboard…")
        st.switch_page("pages/mentor_dashboard.py")
    else:
        st.success("Opening Member Dashboard…")
        st.switch_page("pages/member_dashboard.py")
