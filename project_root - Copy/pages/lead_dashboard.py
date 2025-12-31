# pages/lead_Dashboard.py
from __future__ import annotations

import datetime
import uuid
import hashlib
import re
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
import requests
from typing import Any

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
EMAIL_USER = os.getenv("EMAIL_USER", "shostelmanagement@gmail.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "ehbp lmeq uzgy hzck")


client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
progress_col = db["progress"]
chat_col = db["chat"]
resources_col = db["resources"]

# ---------------- HELPERS ----------------
def safe_get(d: Any, k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def hash_password(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", email))


def generate_password(length: int = 8) -> str:
    import secrets, string
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def user_role_for_project(user: dict, pid: str):
    """
    Safely get user's role for a project (returns None if not found).
    """
    for p in safe_get(user, "projects", []) or []:
        try:
            if str(p.get("project_id")) == str(pid):
                return p.get("role")
        except Exception:
            continue
    return None


def send_email(to: str, subject: str, body: str) -> bool:
    """Minimal safe email sender (best-effort)."""
    if not EMAIL_USER or not EMAIL_PASS:
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "html")
        msg["From"] = EMAIL_USER
        msg["To"] = to
        msg["Subject"] = subject
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, to, msg.as_string())
        return True
    except Exception:
        return False


#<the entire file you pasted remains same until the login email section>

def create_invite(email: str, name: str, role: str, team: dict) -> bool:
    email = (email or "").lower().strip()
    if not email:
        return False

    otp = generate_password()
    otp_hash = hash_password(otp)

    proj_entry = {
        "project_id": safe_get(team, "project_id", ""),
        "team_id": safe_get(team, "team_id", ""),
        "role": role,
        "domain": safe_get(team, "domain", ""),
        "project_name": safe_get(team, "project_name", ""),
    }

    user_existing = users_col.find_one({"email": email})

    if user_existing:
        # ALWAYS update password + reset status on reinvite
        users_col.update_one(
            {"email": email},
            {
                "$push": {"projects": proj_entry},
                "$set": {
                    "password": otp_hash,
                    "status": "pending",
                },
            }
        )
    else:
        # create new user
        users_col.insert_one({
            "user_id": str(uuid.uuid4()),
            "email": email,
            "name": name or email.split("@")[0],
            "password": otp_hash,
            "status": "pending",
            "projects": [proj_entry],
            "created_at": datetime.datetime.utcnow(),
        })

    # Email content
    login_link = "http://localhost:8501"

    body = f"""
        <h2>👋 Welcome to TeamCollab</h2>

        <p>You have been added to:</p>

        <table style="border:1px solid #ccc;padding:12px;border-collapse:collapse;">
        <tr><td><b>Project</b></td><td>{safe_get(team,'project_name')}</td></tr>
        <tr><td><b>Team</b></td><td>{safe_get(team,'team_name')}</td></tr>
        <tr><td><b>Domain</b></td><td>{safe_get(team,'domain')}</td></tr>
        <tr><td><b>Your Role</b></td><td>{role}</td></tr>
        </table>

        <br><h3>🔐 Login</h3>
        <p>
        <b>Email:</b> {email}<br>
        <b>One Time Password:</b> <code>{otp}</code><br>
        </p>

        <p>
        <a href="{login_link}" style="padding:10px 18px;background:#2E86C1;color:white;text-decoration:none;border-radius:6px;">
        Login Now
        </a>
        </p>

        <hr>
        <p style="font-size:13px;color:#999;">
        If you didn’t request this, you can ignore this message.
        </p>
    """

    send_email(email, "TeamCollab Invitation", body)
    return True





def ensure_progress_record(pid: str, email: str | None) -> dict:
    """
    Guarantee a progress record exists for (pid, email).
    Returns record dict or empty dict if pid/email missing.
    """
    if not pid or not email:
        return {}
    pid_s = str(pid)
    email_s = str(email)
    rec = progress_col.find_one({"project_id": pid_s, "user_email": email_s})
    if rec:
        return rec
    new = {
        "progress_id": str(uuid.uuid4()),
        "project_id": pid_s,
        "user_email": email_s,
        "skills": [],
        "percentage": 0,
        "comments": [],
        "notes": "",
        "created_at": datetime.datetime.utcnow(),
    }
    progress_col.insert_one(new)
    return new


# ---------------- PAGE START ----------------
st.set_page_config(page_title="Lead Dashboard", page_icon="👑", layout="centered")
st.markdown("<h1 style='text-align:center;'>👑 Lead Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- AUTH CHECK ----------------
user = st.session_state.get("user")
if not user or not isinstance(user, dict):
    st.error("Please login first.")
    st.stop()

# active_project in session should be the full project dict (not only id).
active_project = st.session_state.get("active_project")
# Ensure active_project is always a full project dict
if isinstance(active_project, str):
    pid_from_session = active_project
    for p in user.get("projects", []):
        if str(p.get("project_id")) == str(pid_from_session):
            active_project = p
            st.session_state.active_project = p
            break

if not active_project:
    st.error("No active project selected.")
    st.stop()

# Determine project id correctly (active_project may be dict or a plain id)
if isinstance(active_project, dict):
    pid = safe_get(active_project, "project_id", "") or ""
else:
    pid = str(active_project or "")

if not pid:
    st.error("No active project selected.")
    st.stop()

role = user_role_for_project(user, pid)

if role != "lead":
    st.error("Only Leads can access this page.")
    st.stop()

team = teams_col.find_one({"project_id": pid}) or {}
lead_email = safe_get(team, "lead_email", safe_get(user, "email"))

# ---------------- HEADER ----------------
st.subheader(f"Project: {safe_get(team,'project_name','(unknown)')} • Domain: {safe_get(team,'domain','(unknown)')}")
st.caption(f"Lead: {lead_email or '(unknown)'}")

ensure_progress_record(pid, lead_email)

# =====================================================================
# 1) INVITE TEAM MEMBERS
# =====================================================================
st.header("✉ Invite Member / Mentor")

with st.form("invite_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    role_choice = st.selectbox("Role", ["member", "mentor"])
    submit_invite = st.form_submit_button("Send Invite")

if submit_invite:
    if not valid_email(email or ""):
        st.error("Invalid email.")
    else:
        ok = create_invite(email, name, role_choice, team)
        if ok:
            st.success("Invite processed.")
            st.rerun()
        else:
            st.error("Invite could not be processed.")

st.markdown("---")

# =====================================================================
# 2) TEAM MEMBERS OVERVIEW
# =====================================================================
st.header("🧑‍🤝‍🧑 Team Members")

members = list(users_col.find({"projects.project_id": pid})) or []

if members:
    rows = []
    for m in members:
        proj_roles = [safe_get(p, "role", "") for p in (m.get("projects") or []) if str(safe_get(p, "project_id", "")) == pid]
        role_val = proj_roles[0] if proj_roles else ""
        rows.append({
            "Name": safe_get(m, "name", "(no name)"),
            "Email": safe_get(m, "email", ""),
            "Role": role_val
        })
    df = pd.DataFrame(rows)
    st.dataframe(df)
else:
    st.info("No members yet.")

st.markdown("---")

# =====================================================================
# 3) PROJECT RESOURCES
# =====================================================================
st.header("📂 Project Resources")

resources = list(resources_col.find({"project_id": pid})) or []
if resources:
    rows = []
    for r in resources:
        rows.append({
            "Title": safe_get(r, "title", "(Untitled)"),
            "By": safe_get(r, "user_email", ""),
            "Notes": safe_get(r, "notes", ""),
            "Link": safe_get(r, "link", "")
        })
    df_res = pd.DataFrame(rows)
    st.dataframe(df_res)
else:
    st.info("No submissions yet.")

st.markdown("---")

# =====================================================================
# 4) PROJECT CHAT
# =====================================================================
st.header("💬 Project Chat")

with st.form("chat_form"):
    msg = st.text_area("Message")
    send_msg = st.form_submit_button("Send")

if send_msg and (msg or "").strip():
    chat_col.insert_one({
        "project_id": pid,
        "user_email": lead_email,
        "text": (msg or "").strip(),
        "created_at": datetime.datetime.utcnow()
    })
    st.success("Message sent.")
    st.rerun()

# Display messages (most recent first)
msgs = list(chat_col.find({"project_id": pid}).sort("created_at", -1)) or []
for m in msgs[:50]:
    t = safe_get(m, "created_at")
    if isinstance(t, datetime.datetime):
        t_str = t.strftime("%Y-%m-%d %H:%M")
    else:
        t_str = ""
    st.markdown(f"**{safe_get(m, 'user_email', '')}** ({t_str}): {safe_get(m, 'text', '')}")

st.markdown("---")

# =====================================================================
# 5) LEAD PERSONAL PROGRESS
# =====================================================================
st.header("🧑‍💼 Your Progress")

pr = ensure_progress_record(pid, lead_email)
skills = ", ".join(pr.get("skills", []) or [])
notes = pr.get("notes", "") or ""
percent = int(pr.get("percentage", 0) or 0)

with st.form("lead_progress"):
    in_skills = st.text_input("Skills (comma separated)", skills)
    in_notes = st.text_area("Notes", notes)
    in_percent = st.slider("Progress %", 0, 100, percent)
    save_progress = st.form_submit_button("Save")

if save_progress:
    progress_col.update_one(
        {"progress_id": pr["progress_id"]},
        {"$set": {
            "skills": [s.strip() for s in (in_skills or "").split(",") if s.strip()],
            "notes": in_notes,
            "percentage": in_percent,
            "updated_at": datetime.datetime.utcnow()
        }}
    )
    st.success("Progress updated.")
    st.rerun()

st.markdown("---")

# =====================================================================
# 6) CREATE NEW PROJECT (WITH DOMAIN → AI SUGGESTIONS → FINALIZE)
# =====================================================================
st.header("➕ Create New Project")

# -----------------------------
# AI Project Suggestion Engine
# -----------------------------
def suggest_projects(domain, count=5):
    """AI or fallback project suggestion engine."""
    try:
        prompt = (
            f"You are an expert academic project mentor. Generate {count} UNIQUE academic project ideas "
            f"for the domain '{domain}'.\n\n"
            "STRICT FORMAT:\n"
            "1. <Project Title>\n"
            "   <Intro sentence 1>\n"
            "   <Intro sentence 2>\n"
            "   <What the system DOES>\n"
            "   <Academic Benefit>\n\n"
            "NO explanations. ONLY the numbered list."
        )

        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        ).json()

        text = (res.get("message", {}) or {}).get("content", "")

        # split into numbered blocks
        raw = re.split(r"\n\d+\.\s*", "\n" + text)
        raw = [p.strip() for p in raw if p.strip()]

        ideas = []
        for block in raw[:count]:
            lines = block.split("\n")
            title = lines[0].strip()
            desc = "\n".join(lines[1:]).strip()
            ideas.append({"title": title, "description": desc})

        return ideas if ideas else []
    
    except Exception:
        # fallback
        return [
            {"title": f"{domain} Project {i+1}", 
             "description": "A domain-relevant academic project for learning & research."}
            for i in range(count)
        ]


# -----------------------------
# Step 1 — Select Domain
# -----------------------------
st.subheader("✨ Step 1: Select Domain to Generate Ideas")

domain_list = [
    "AI & Machine Learning", "Web Development", "Mobile App Development",
    "Cybersecurity", "Data Science", "IoT", "Cloud Computing", "Blockchain",
    "AR/VR", "Robotics", "NLP", "Computer Vision", "Game Development",
    "Big Data", "Embedded Systems", "Quantum Computing", "DevOps",
    "HealthTech", "FinTech", "EdTech"
]

sel_domain = st.selectbox("Select Domain", domain_list)

if st.button("✨ Generate Suggestions"):
    st.session_state.new_proj_suggestions = suggest_projects(sel_domain)
    st.success("Suggestions generated!")
    st.rerun()

suggestions = st.session_state.get("new_proj_suggestions", [])


# -----------------------------
# Step 2 — Choose one suggested project
# -----------------------------
selected_title = None
selected_desc = None

if suggestions:
    st.subheader("📋 Step 2: Choose a Project Idea")

    titles = [f"{i+1}. {s['title']}" for i, s in enumerate(suggestions)]
    choice = st.radio("Select one project", titles)

    idx = titles.index(choice)
    selected_title = suggestions[idx]["title"]
    selected_desc = suggestions[idx]["description"]

    st.markdown(f"### 📘 Selected: **{selected_title}**")
    st.markdown(selected_desc)
    st.markdown("---")


# -----------------------------
# Step 3 — Finalize & Create Project
# -----------------------------
st.subheader("🚀 Step 3: Finalize & Create")

with st.form("final_create"):
    new_team = st.text_input("Team Name")
    final_title = st.text_input("Project Title", selected_title or "")
    final_domain = st.text_input("Domain", sel_domain or "")
    submit_new = st.form_submit_button("Create Project")

if submit_new:
    if not (new_team and final_title and final_domain):
        st.error("All fields required.")
    else:
        tid = str(uuid.uuid4())
        pid_new = str(uuid.uuid4())

        # Insert into teams table
        teams_col.insert_one({
            "team_id": tid,
            "project_id": pid_new,
            "team_name": new_team,
            "project_name": final_title,
            "domain": final_domain,
            "lead_email": lead_email,
            "created_at": datetime.datetime.utcnow(),
        })

        # Add project to lead's account
        users_col.update_one(
            {"email": lead_email},
            {"$push": {
                "projects": {
                    "project_id": pid_new,
                    "team_id": tid,
                    "role": "lead",
                    "domain": final_domain,
                    "project_name": final_title
                }
            }},
            upsert=True
        )

        # Update active project in session
        st.session_state.active_project = {
            "project_id": pid_new,
            "team_id": tid,
            "role": "lead",
            "domain": final_domain,
            "project_name": final_title
        }

        st.success("🎉 New project created successfully!")
        st.session_state.new_proj_suggestions = []
        st.rerun()
st.success("Lead Dashboard Loaded Successfully.")