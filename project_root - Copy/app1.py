# app.py
from __future__ import annotations

import datetime
import hashlib
import re
import uuid
import secrets
import smtplib
import string
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import requests
from pymongo import MongoClient
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

st.set_page_config(page_title="TeamCollab • AI-Ready", layout="centered", page_icon="🤖")

EMAIL_USER = os.getenv("EMAIL_USER", "shostelmanagement@gmail.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "ehbplmequzgyhzck")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
tasks_col = db["tasks"]
progress_col = db["progress"]
resources_col = db["resources"]
chat_col = db["chat"]
objectives_col = db["objectives"]

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def safe_get(obj, key, default=None):
    return obj[key] if isinstance(obj, dict) and key in obj else default

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def generate_password(length=8):
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))

def valid_email(email: str) -> bool:
    return bool(email and re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email))

def send_email(to_email: str, subject: str, html_body: str):
    if not to_email:
        return False

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, to_email, subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, to_email, msg.as_string())
        return True
    except Exception:
        return False

def ensure_progress_record(pid, email):
    rec = progress_col.find_one({"project_id": pid, "user_email": email})
    if rec:
        return rec
    new = {
        "progress_id": str(uuid.uuid4()),
        "project_id": pid,
        "user_email": email,
        "skills": [],
        "percentage": 0,
        "notes": "",
        "comments": [],
        "created_at": datetime.datetime.utcnow(),
    }
    progress_col.insert_one(new)
    return new

# -------------------------------------------------------
# SESSION INIT
# -------------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# active_project will hold the full project dict (not just id)
if "active_project" not in st.session_state:
    st.session_state.active_project = None

if "suggested_projects" not in st.session_state:
    st.session_state.suggested_projects = []

# -------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------

st.markdown(
    "<h1 style='text-align:center;'>🤖 TeamCollab</h1>"
    "<p style='text-align:center;color:gray;'>AI-powered Academic Project Manager</p><hr>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# DOMAINS
# -------------------------------------------------------

DOMAINS = [
    "AI & Machine Learning", "Web Development", "Mobile App Development",
    "Cybersecurity", "Data Science", "IoT", "Cloud Computing", "Blockchain",
    "AR/VR", "Robotics", "NLP", "Computer Vision", "Game Development",
    "Big Data", "Embedded Systems", "Quantum Computing", "DevOps",
    "HealthTech", "FinTech", "EdTech"
]

# -------------------------------------------------------
# PROJECT SUGGESTION ENGINE (FULL INTRO + DESCRIPTION + BENEFITS)
# -------------------------------------------------------

def suggest_projects(domain, count=5):

    try:
        prompt = (
    f"You are an expert academic project mentor. Generate {count} UNIQUE academic project ideas "
    f"for the domain '{domain}'.\n\n"

    "STRICT OUTPUT FORMAT (must follow EXACTLY):\n"
    "1. <Project Title>\n"
    "   <1st introduction sentence>\n"
    "   <2nd introduction sentence>\n"
    "   <Sentence describing what the project DOES>\n"
    "   <Sentence describing the academic BENEFIT>\n\n"

    "HARD RULES:\n"
    "- DO NOT write 'Line 1', 'Line 2', 'Line 3', or 'Line 4'.\n"
    "- DO NOT add extra explanations, paragraphs, or numbering like 'Project Title:'.\n"
    "- Each project must appear exactly as: Number + Title + 4 lines.\n"
    "- All 5 projects MUST be generated.\n"
    "- No repetition, no generic filler, no missing items.\n\n"

    f"Generate exactly {count} projects using ONLY the format above."
)


        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        ).json()

        text = safe_get(safe_get(res, "message", {}), "content", "")

        raw_projects = re.split(r"\n\d+\.\s*", "\n" + (text or ""))
        raw_projects = [p.strip() for p in raw_projects if p.strip()]

        ideas = []
        for block in raw_projects[:count]:
            lines = block.split("\n")
            title = lines[0].strip() if lines else f"{domain} Project"
            desc = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            ideas.append({
                "title": title,
                "description": desc
            })

        while len(ideas) < count:
            ideas.append({
                "title": f"{domain} Project {len(ideas)+1}",
                "description": (
                    "This is an academic project related to the selected domain.\n"
                    "It teaches practical implementation and domain-specific skills.\n"
                    "The system performs a real-world function.\n"
                    "Useful for academic research and hands-on learning."
                )
            })

        return ideas

    except Exception:
        return [
            {
                "title": f"{domain} Project {i+1}",
                "description": (
                    "A domain-relevant academic project designed for practical learning.\n"
                    "It demonstrates core technical concepts.\n"
                    "Provides real-world application understanding.\n"
                    "Highly valuable for students and research."
                )
            }
            for i in range(count)
        ]

# -------------------------------------------------------
# MAIN MENU
# -------------------------------------------------------

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Register (Lead)", "🔐 Login", "📊 Dashboard"],
)

# -------------------------------------------------------
# REGISTER (LEAD)
# -------------------------------------------------------

if menu == "🏠 Register (Lead)":
    st.subheader("👑 Register as Team Lead")

    with st.form("lead_step1"):
        team_name = st.text_input("Team Name")
        lead_name = st.text_input("Full Name")
        lead_email = st.text_input("Email")
        domain = st.selectbox("Select Project Domain", DOMAINS)
        generate_btn = st.form_submit_button("✨ Generate Project Ideas")

    if generate_btn:
        st.session_state.suggested_projects = suggest_projects(domain)
        st.rerun()

    if st.session_state.suggested_projects:
        st.markdown("### 💡 Suggested Project Ideas")
        for i, idea in enumerate(st.session_state.suggested_projects, 1):
            st.markdown(f"## {i}. {idea['title']}")
            st.markdown(idea["description"])
            st.markdown("---")

    with st.form("lead_step2"):
        project_title = st.text_input("Project Title (choose from above OR type your own)")
        submit_btn = st.form_submit_button("🚀 Create Project")

    if submit_btn:

        if not all([team_name, lead_name, lead_email, domain, project_title]):
            st.error("All fields required.")
            st.stop()

        if not valid_email(lead_email):
            st.error("Invalid email format.")
            st.stop()

        pid = str(uuid.uuid4())
        tid = str(uuid.uuid4())
        pwd = generate_password()

        users_col.insert_one({
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
        })

        teams_col.insert_one({
            "team_id": tid,
            "project_id": pid,
            "project_name": project_title,
            "team_name": team_name,
            "domain": domain,
            "lead_email": lead_email.lower(),
            "created_at": datetime.datetime.utcnow()
        })

        send_email(
            lead_email,
            "Your TeamCollab Account",
            f"<h3>Welcome to TeamCollab!</h3>Email: {lead_email}<br>Password: <code>{pwd}</code>"
        )

        st.success("Project created successfully! Check your email for login details.")
        st.session_state.suggested_projects = []
        st.rerun()

# -------------------------------------------------------
# LOGIN
# -------------------------------------------------------

elif menu == "🔐 Login":
    st.subheader("🔑 Login")

    with st.form("login"):
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        user = users_col.find_one({"email": (email or "").lower()})
        if not user:
            st.error("User not found.")
        elif hash_password(pwd or "") != user.get("password", ""):
            st.error("Incorrect password.")
        else:
            st.session_state.logged_in = True
            st.session_state.user = user

            # Try to set active_project as a full project dict (not just id)
            projects = user.get("projects", []) or []
            desired_pid = user.get("last_active_project") or (projects[0]["project_id"] if projects else None)
            active_proj = None
            if desired_pid:
                # find matching project object in user.projects
                for p in projects:
                    if str(p.get("project_id")) == str(desired_pid):
                        active_proj = p
                        break
            # fallback to first project if none found
            if not active_proj and projects:
                active_proj = projects[0]

            # store the full project dict in session
            st.session_state.active_project = active_proj

            st.success("Login successful!")
            st.rerun()

# -------------------------------------------------------
# DASHBOARD ROUTER
# -------------------------------------------------------

elif menu == "📊 Dashboard":

    if not st.session_state.logged_in:
        st.warning("Please log in first.")
        st.stop()

    if st.session_state.user is None:
        st.warning("Session expired. Please login again.")
        st.stop()

    user = st.session_state.user
    projects = user.get("projects", []) or []

    if not projects:
        st.info("You are not part of any project.")
        st.stop()

    options = [
        f"{p.get('project_name','(untitled)')} | {p.get('domain','(no domain)')} | {p.get('role','member').capitalize()}"
        for p in projects
    ]
    selected_idx = st.selectbox("Select Project", range(len(options)), format_func=lambda i: options[i])

    proj = projects[selected_idx]
    # store full project dict into session (this is the key fix)
    st.session_state.active_project = proj

    # persist last_active_project as id in DB (keeps user doc small)
    try:
        users_col.update_one(
            {"email": user.get("email")},
            {"$set": {"last_active_project": proj.get("project_id")}}
        )
    except Exception:
        pass

    role = proj.get("role", "member")
    st.success(f"Opening {role.capitalize()} Dashboard…")

    # NOTE: streamlit requires the page name (file name without .py) inside pages/ folder.
    # Adjust these names if your pages have different filenames.
    # Use page *names* that match your folder filenames (without .py)
    # -------------------------------------------------------
# DASHBOARD ROUTER
# -------------------------------------------------------

elif menu == "📊 Dashboard":

    if not st.session_state.logged_in:
        st.warning("Please log in first.")
        st.stop()

    if st.session_state.user is None:
        st.warning("Session expired. Please login again.")
        st.stop()

    user = st.session_state.user
    projects = user.get("projects", []) or []

    if not projects:
        st.info("You are not part of any project.")
        st.stop()

    options = [
        f"{p.get('project_name','(untitled)')} | {p.get('domain','(no domain)')} | {p.get('role','member').capitalize()}"
        for p in projects
    ]
    selected_idx = st.selectbox("Select Project", range(len(options)), format_func=lambda i: options[i])

    proj = projects[selected_idx]

    # store full project dict into session (not id only!)
    st.session_state.active_project = proj

    # persist last_active_project to DB
    try:
        users_col.update_one(
            {"email": user.get("email")},
            {"$set": {"last_active_project": proj.get("project_id")}}
        )
    except Exception:
        pass

    role = proj.get("role", "member")
    st.success(f"Opening {role.capitalize()} Dashboard…")


    # ===== Normal Dashboards (existing) =====
    if role == "lead":
        try:
            st.switch_page("lead_Dashboard")
        except Exception:
            st.info("Please open the Lead Dashboard page manually.")
    elif role == "mentor":
        try:
            st.switch_page("mentor_Dashboard")
        except Exception:
            st.info("Please open the Mentor Dashboard page manually.")
    else:
        try:
            st.switch_page("member_Dashboard")
        except Exception:
            st.info("Please open the Member Dashboard page manually.")

    # ===== ADD THIS (NOT ROLE BASED) =====
    try:
        st.switch_page("report_generator")
    except Exception:
        st.info("Please open the Report Generator page from the left menu (if available).")
