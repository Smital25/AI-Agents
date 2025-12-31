# pages/lead_dashboard.py

from __future__ import annotations
import datetime
import uuid
import hashlib
import re
import smtplib
import os
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import gridfs
from bson import ObjectId

# ---------------------------------------------------
# DB + EMAIL CONFIG
# ---------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
EMAIL_USER = os.getenv("EMAIL_USER", "shostelmanagement@gmail.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "ehbplmequzgyhzck")

client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
progress_col = db["progress"]
chat_col = db["chat"]
resources_col = db["resources"]
fs = gridfs.GridFS(db)
tasks_col = db["tasks"]
# ---------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------
st.set_page_config(page_title="Lead Dashboard", page_icon="👑", layout="centered")
st.title("👑 Lead Dashboard")
st.markdown("---")

# ---------------------------------------------------
# SESSION VALIDATION
# ---------------------------------------------------
if "active_user" not in st.session_state or not st.session_state.active_user:
    st.error("Session expired. Login again.")
    st.stop()

if "active_project" not in st.session_state or not st.session_state.active_project:
    st.error("No project selected.")
    st.stop()

active_user = st.session_state.active_user
active_project = st.session_state.active_project

user_projects = active_user.get("projects", [])
pid = active_project.get("project_id")

# -----------------------------------------------------------
# 🔄 UNIVERSAL PROJECT SWITCHER (Safe for Multi-Role Users)
# -----------------------------------------------------------

user_projects = active_user.get("projects", [])

if len(user_projects) > 1:
    st.sidebar.markdown("### 🔄 Switch Project")

    # Build human-readable labels
    label_map = {}
    for p in user_projects:
        label = f"{p.get('project_name','Untitled')} ({p.get('role','member')})"
        label_map[label] = p

    # Current
    current_label = f"{active_project.get('project_name')} ({active_project.get('role')})"

    selected_label = st.sidebar.selectbox(
        "Select Project",
        list(label_map.keys()),
        index=list(label_map.keys()).index(current_label)
    )

    # If user picked a different project
    if selected_label != current_label:
        chosen_proj = label_map[selected_label]

        # Update session
        st.session_state.active_project = chosen_proj
        st.session_state.active_role = chosen_proj.get("role", "").lower()

        # Route to correct dashboard
        role = st.session_state.active_role

        if role == "lead":
            st.switch_page("pages/lead_dashboard.py")
        elif role == "mentor":
            st.switch_page("pages/mentor_dashboard.py")
        else:
            st.switch_page("pages/member_dashboard.py")

        st.rerun()

# ---------------------------------------------------
# ROLE CHECK
# ---------------------------------------------------
active_role = active_project.get("role", "").lower()

if active_role != "lead":
    st.error("❌ Access Denied: Only PROJECT LEADS can open Lead Dashboard.")
    st.stop()



# ---------------------------------------------------
# LOAD PROJECT DETAILS
# ---------------------------------------------------
team = teams_col.find_one({"project_id": pid})
if not team:
    st.error("Team not found.")
    st.stop()

lead_email = team.get("lead_email")


def refresh_user(email):
    return users_col.find_one({"email": email})


fresh = refresh_user(lead_email)
if fresh:
    st.session_state.active_user = fresh
    active_user = fresh

st.subheader(f"📘 {team.get('project_name')}")
st.caption(f"Domain: {team.get('domain')} • Lead: {lead_email}")


# ---------------------------------------------------
# EMAIL HELPERS
# ---------------------------------------------------
def valid_email(email: str) -> bool:
    return bool(email and re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", email))


def hash_password(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def generate_password(length: int = 8) -> str:
    import secrets, string
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


def send_email(to: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, to, subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, to, msg.as_string())

        return True
    except:
        return False

#--------------------------------MENTOR RESTRICTION--------------------------------
def mentor_exists(project_id: str) -> bool:
    return users_col.count_documents({
        "projects": {
            "$elemMatch": {
                "project_id": project_id,
                "role": "mentor"
            }
        }
    }) > 0
st.divider()


# ---------------------------------------------------
# INVITE MEMBER / MENTOR
# ---------------------------------------------------
st.header("✉ Invite Member / Mentor")

def create_invite(email: str, name: str, role: str, team: dict):
    email = email.strip().lower()
    otp = generate_password()
    otp_hash = hash_password(otp)

     # 🚫 Enforce ONE mentor per project
    if role == "mentor" and mentor_exists(team["project_id"]):
        st.error("🚫 This project already has a mentor. Only ONE mentor is allowed.")
        return

    proj_entry = {
        "project_id": team["project_id"],
        "team_id": team["team_id"],
        "role": role,
        "domain": team["domain"],
        "project_name": team["project_name"]
    }

    existing = users_col.find_one({"email": email})
    if existing:
        users_col.update_one(
            {"email": email},
            {"$push": {"projects": proj_entry},
             "$set": {"password": otp_hash}}
        )
    else:
        users_col.insert_one({
            "user_id": str(uuid.uuid4()),
            "email": email,
            "name": name,
            "password": otp_hash,
            "status": "pending",
            "projects": [proj_entry],
            "created_at": datetime.datetime.utcnow()
        })

    # ----------------------------
    # PROFESSIONAL EMAIL TEMPLATE
    # ----------------------------
    body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; line-height: 1.6;">

        <h2 style="color: #2B579A;">TeamCollab – Project Invitation</h2>

        <p>Dear {name},</p>

        <p>
            You have been invited to join the project 
            <strong>{team.get('project_name')}</strong> as a 
            <strong>{role.capitalize()}</strong>.
        </p>

        <p>Please use the following One-Time Password (OTP) to activate your account:</p>

        <div style="background: #f3f6fb; padding: 15px; border-left: 4px solid #2B579A; border-radius: 6px;">
            <h3 style="margin: 0; color: #2B579A; font-size: 22px;">{otp}</h3>
        </div>

        <p>
            Enter this OTP on the login page to complete your registration.  
            If you did not expect this invitation, kindly ignore this email.
        </p>

        <p>
                You can log in using the link below:
                <br><br>
                🔗 <a href='http://localhost:8501' style='color:#2B579A; font-size:16px; font-weight:bold;'>
                    Open TeamCollab Dashboard
                </a>
        </p>

        <p>
            Best regards,<br>
            <strong>TeamCollab System</strong>
        </p>

    </div>
    """

    # SAME FUNCTION NAME — NO CHANGE
    send_email(
        email,
        "TeamCollab Invitation – OTP Verification",
        body
    )


# ---- FORM UI ----
with st.form("invite_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    mentor_taken = mentor_exists(pid)
    role_choice = st.selectbox(
        "Role",
        ["member"] if mentor_taken else ["member", "mentor"]
    )

    btn = st.form_submit_button("Send Invite")

if btn:
    if not valid_email(email):
        st.error("Invalid email")
    else:
        create_invite(email, name, role_choice, team)
        st.success("Invitation sent!")
        st.rerun()

st.divider()


# ---------------------------------------------------
# TEAM MEMBERS
# ---------------------------------------------------
st.header("🧑‍🤝‍🧑 Team Members")

members = list(users_col.find({"projects.project_id": pid}))

rows = []
for m in members:
    proj = next((p for p in m["projects"] if p["project_id"] == pid), {})
    rows.append({
        "Name": m.get("name"),
        "Email": m.get("email"),
        "Role": proj.get("role", "member")
    })

st.dataframe(pd.DataFrame(rows))
st.divider()

# ---------------------------------------------------
# PROGRESS
# ---------------------------------------------------
st.header("🛠 Lead Skills")

def ensure_progress(project_id: str, email: str) -> dict:
    rec = progress_col.find_one({
        "project_id": project_id,
        "user_email": email
    })
    if rec:
        return rec

    new = {
        "progress_id": str(uuid.uuid4()),
        "project_id": project_id,
        "user_email": email,
        "skills": [],          # ONLY skills now
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow(),
    }
    progress_col.insert_one(new)
    return new

progress = ensure_progress(pid, lead_email)
existing_skills = progress.get("skills", [])

skill_name = st.text_input("Skill name")
skill_level = st.selectbox(
    "Skill level",
    ["Beginner", "Intermediate", "Expert"],
    key="lead_skill_level"
)

if st.button("➕ Add / Update Skill", key="lead_skill_add"):
    filtered = [s for s in existing_skills if s["name"].lower() != skill_name.lower()]
    filtered.append({
        "name": skill_name.strip(),
        "level": skill_level.lower()
    })

    progress_col.update_one(
        {"progress_id": progress["progress_id"]},
        {"$set": {
            "skills": filtered,
            "updated_at": datetime.datetime.utcnow()
        }}
    )

    st.success("Skill updated!")
    st.rerun()

if existing_skills:
    st.dataframe(pd.DataFrame(existing_skills))
else:
    st.info("No skills added yet.")


# ---------------------------------------------------
# LEAD ASSIGNED TASKS
# ---------------------------------------------------
st.header("📝 My Assigned Tasks (Lead)")

lead_tasks = list(tasks_col.find({
    "project_id": pid,
    "assigned_to": lead_email
}))

if not lead_tasks:
    st.info("No tasks assigned to you yet.")
else:
    for t in lead_tasks:
        tid = t.get("task_id")
        with st.expander(f"📌 {t.get('title')}"):
            st.write("**Description:**", t.get("description"))
            st.write("**Status:**", t.get("status"))

            status_opts = ["Not Started", "In Progress", "Blocked", "Completed"]
            curr = t.get("status", "Not Started")

            new_status = st.selectbox(
                "Update Status",
                status_opts,
                index=status_opts.index(curr),
                key=f"lead_task_status_{tid}"
            )

            if st.button("💾 Save Status", key=f"lead_save_{tid}"):
                tasks_col.update_one(
                    {"task_id": tid},
                    {"$set": {"status": new_status}}
                )
                st.success("Task status updated")
                st.rerun()
st.divider()

# ---------------------------------------------------
# SUBMIT WORK (LEAD)
# ---------------------------------------------------
st.header("📂 Submit Your Work (Lead)")

with st.form("lead_submit_work"):
    title = st.text_input("Title")
    notes = st.text_area("Notes / Explanation")
    link = st.text_input("Reference / GitHub / Drive Link")

    uploaded_files = st.file_uploader(
        "Upload Files",
        accept_multiple_files=True,
        type=[
            "pdf","doc","docx","xls","xlsx",
            "ppt","pptx","png","jpg","jpeg",
            "mp4","mov","avi","zip","rar",
            "txt","py","js","ipynb"
        ]
    )

    submit_btn = st.form_submit_button("🚀 Submit Work")

if submit_btn:
    base_doc = {
        "resource_id": str(uuid.uuid4()),
        "project_id": pid,
        "user_email": lead_email,
        "title": title or "(Untitled)",
        "notes": notes,
        "link": link,
        "created_at": datetime.datetime.utcnow(),
        "submitted_by_role": "lead"
    }

    if uploaded_files:
        for f in uploaded_files:
            file_id = fs.put(
                f.read(),
                filename=f.name,
                content_type=f.type
            )

            doc = base_doc.copy()
            doc["file_id"] = str(file_id)
            doc["filename"] = f.name
            doc["file_type"] = f.type

            resources_col.insert_one(doc)
    else:
        resources_col.insert_one(base_doc)

    st.success("✅ Work submitted successfully!")
    st.rerun()
st.divider()

# ---------------------------------------------------
# 6️⃣ CREATE NEW PROJECT (LEAD CAN CREATE MORE PROJECTS)
# ---------------------------------------------------
st.header("🚀 Create Another Project")

DOMAIN_LIST = [
    "AI & Machine Learning", "Web Development", "Mobile App Development",
    "Cybersecurity", "Data Science", "IoT", "Cloud Computing", "Blockchain",
    "AR/VR", "Robotics", "NLP", "Computer Vision", "Game Development",
    "Big Data", "Embedded Systems", "Quantum Computing", "DevOps",
    "HealthTech", "FinTech", "EdTech"
]

def safe_get(obj, key, default=None):
    return obj[key] if isinstance(obj, dict) and key in obj else default

# ---- AI PROJECT SUGGESTION ENGINE ----
def suggest_projects(domain, count=5):
    ideas = []

    for i in range(count):
        prompt = f"""
Generate EXACTLY ONE academic project idea for the domain "{domain}".

STRICT OUTPUT FORMAT:

---
TITLE: <Project Title>

INTRODUCTION:
<2 meaningful sentences>

WHAT IT DOES:
<1 clear sentence>

ACADEMIC BENEFIT:
<1 clear sentence>
---

Do NOT number. Do NOT add extra explanation.
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

            match = re.search(r"TITLE:\s*(.*)", blk)
            title = match.group(1).strip() if match else f"{domain} Project {i+1}"

            ideas.append({"title": title, "description": blk})

        except Exception:
            ideas.append({
                "title": f"{domain} Project {i+1}",
                "description": "INTRODUCTION:\nBackup intro.\nWHAT IT DOES:\nBackup.\nACADEMIC BENEFIT:\nBackup."
            })

    return ideas


# ---- UI FOR PROJECT IDEAS ----
sel_domain = st.selectbox("Select Domain", DOMAIN_LIST, key="lead_new_domain")

if st.button("✨ Generate Project Ideas (Lead)", key="lead_generate"):
    st.session_state["lead_suggestions"] = suggest_projects(sel_domain)
    st.rerun()

lead_suggestions = st.session_state.get("lead_suggestions", [])

if lead_suggestions:
    titles = [s["title"] for s in lead_suggestions]
    selected_title = st.radio("Choose from Suggestions", titles)
    chosen = next(s for s in lead_suggestions if s["title"] == selected_title)

    st.markdown(f"### {chosen['title']}")
    st.markdown(chosen["description"])
else:
    chosen = {"title": ""}


# ---- PROJECT CREATION FORM ----
with st.form("lead_create_new_project"):
    team_name = st.text_input("Team Name")
    proj_title = st.text_input("Project Title", chosen.get("title", ""))
    proj_desc = st.text_area("Project Description")
    domain_input = st.text_input("Domain", sel_domain)

    create_btn = st.form_submit_button("🚀 Create Project")

if create_btn:
    if not (team_name and proj_title and domain_input):
        st.error("All fields are required.")
    else:
        new_pid = str(uuid.uuid4())
        new_tid = str(uuid.uuid4())

        # Insert into teams collection
        teams_col.insert_one({
            "team_id": new_tid,
            "project_id": new_pid,
            "team_name": team_name,
            "project_name": proj_title,
            "domain": domain_input,
            "lead_email": active_user["email"],
            "description": proj_desc,
            "created_at": datetime.datetime.utcnow(),
        })

        # Add project to user's profile
        users_col.update_one(
            {"email": active_user["email"]},
            {"$push": {
                "projects": {
                    "project_id": new_pid,
                    "team_id": new_tid,
                    "role": "lead",
                    "domain": domain_input,
                    "project_name": proj_title,
                }
            }}
        )

        # Switch session to new project
        st.session_state.active_project = {
            "project_id": new_pid,
            "team_id": new_tid,
            "role": "lead",
            "domain": domain_input,
            "project_name": proj_title,
        }

        st.success("🎉 New Project Created Successfully!")
        st.rerun()
