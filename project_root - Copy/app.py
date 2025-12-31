"""
TeamCollab v12 — AI-Ready, Role-Based, Charts + Gantt + Chat
Streamlit + MongoDB + Gmail SMTP
✅ Mentor feedback visible to members
✅ All roles can create new teams/projects
✅ All status fields + timestamps updated
✅ Role-wise dashboards + Gantt + Project Chat + Resources
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
import uuid
import secrets
import smtplib
import string
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px  # for Gantt / charts
import requests
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pymongo import MongoClient
from pages.scrum_task_planner import task_planner_page
from pages.objectives_generator import generate_objectives  # noqa: F401  # (import used externally)

st.set_page_config(page_title="TeamCollab • AI-Ready", layout="centered", page_icon="🤖")

EMAIL_USER = "shostelmanagement@gmail.com"
EMAIL_PASS = "ehbp lmeq uzgy hzck"
MONGO_URI = "mongodb://localhost:27017/"
BASE_URL = "http://localhost:8501"

client_db = MongoClient(MONGO_URI)
db = client_db["team_collab_db"]
users_col = db["users"]
teams_col = db["teams"]
progress_col = db["progress"]
tasks_col = db["tasks"]
chat_col = db["chat"]
resources_col = db["resources"]

# ---------------- HELPERS ----------------
def safe_get(obj: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    return obj[key] if isinstance(obj, dict) and key in obj else default


def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def generate_password(length: int = 8) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def get_full_project_dict(
    user_doc: Optional[Dict[str, Any]], project_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full project dict from user's projects list by project_id.
    This is needed because pages expect active_project to be a full dict, not just an ID.
    """
    if not user_doc or not project_id:
        return None
    for p in (safe_get(user_doc, "projects", []) or []):
        if str(safe_get(p, "project_id", "")) == str(project_id):
            return p
    return None


def valid_email(email: Optional[str]) -> bool:
    return bool(email and re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email.strip()))


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = EMAIL_USER, to_email, subject
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"📧 Email failed: {e}")
        return False


# ---------------- DOMAINS ----------------
DOMAINS: List[str] = [
    "AI & ML",
    "Web Development",
    "Data Science",
    "Cybersecurity",
    "Internet of Things (IoT)",
    "Blockchain",
    "Cloud Computing",
    "Augmented & Virtual Reality (AR/VR)",
    "FinTech",
    "EdTech",
    "HealthTech",
    "Game Development",
]

TASK_STATUSES: List[str] = ["Not Started", "In Progress", "Blocked", "Completed"]


# ---------------- PROJECT SUGGESTIONS ----------------
def clean_idea_text(text: str) -> List[str]:
    ideas = re.split(r"\d+\.\s*", text or "")
    return [i.strip("•- \n") for i in ideas if i.strip()]


def suggest_projects(domain: str, count: int = 5) -> List[str]:
    try:
        prompt = f"Suggest {count} creative and realistic project ideas related to {domain}."
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": [{"role": "user", "content": prompt}]},
            timeout=25,
        )
        text = ""
        for line in (response.iter_lines() or []):
            if line:
                data = json.loads(line.decode("utf-8"))
                if isinstance(data, dict) and "message" in data:
                    text += str(safe_get(data.get("message", {}), "content", ""))
        ideas = clean_idea_text(text)
        if ideas:
            return ideas[:count]
    except Exception:  # noqa: BLE001
        pass

    defaults: Dict[str, List[str]] = {
        "AI & ML": ["AI Resume Screener", "Fake News Detector", "Smart Chatbot Assistant", "Traffic Prediction System"],
        "Web Development": ["Online Hostel Portal", "Crowdfunding Website", "Portfolio Builder", "E-Library Portal"],
        "Blockchain": ["Decentralized Voting System", "NFT Marketplace", "Digital Identity Chain"],
        "Data Science": ["Sales Forecasting Tool", "Sports Analytics Dashboard", "AI Research Summarizer"],
        "HealthTech": ["Heart Disease Predictor", "AI Health Chatbot", "Remote Patient Monitoring"],
        "Game Development": ["2D RPG Game", "Multiplayer Racing Simulator", "AI Opponent Chess"],
    }
    base = defaults.get(domain, ["Generic Project Idea"])
    return random.sample(base, min(count, len(base)))


# ---------------- DB HELPERS ----------------
def ensure_progress_record(project_id: Optional[str], user_email: Optional[str]) -> Dict[str, Any]:
    if not project_id or not user_email:
        return {}
    pid, email = str(project_id), str(user_email)
    rec = progress_col.find_one({"project_id": pid, "user_email": email})
    if rec:
        return rec
    rec = {
        "progress_id": str(uuid.uuid4()),
        "project_id": pid,
        "user_email": email,
        "skills": [],
        "marks": None,
        "comments": [],
        "percentage": 0.0,
        "notes": "",
        "updated_at": datetime.datetime.utcnow(),
    }
    progress_col.insert_one(rec)
    return rec


def user_role_for_project(user_doc: Optional[Dict[str, Any]], project_id: Optional[str]) -> Optional[str]:
    if not user_doc or not project_id:
        return None
    for p in (safe_get(user_doc, "projects", []) or []):
        if safe_get(p, "project_id") == str(project_id):
            return safe_get(p, "role")
    return None


def create_invite_user(email: str, name: str, role: str, team_doc: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(team_doc, dict):
        return False
    email_l = email.lower().strip()
    existing = users_col.find_one({"email": email_l})
    proj_entry = {
        "project_id": safe_get(team_doc, "project_id", ""),
        "team_id": safe_get(team_doc, "team_id", ""),
        "role": role,
        "domain": safe_get(team_doc, "domain", "—"),
        "project_name": safe_get(team_doc, "project_name", "—"),
    }
    otp = generate_password(8)
    if existing:
        if any(
            safe_get(p, "project_id") == proj_entry["project_id"]
            for p in (safe_get(existing, "projects", []) or [])
        ):
            return True
        users_col.update_one({"email": email_l}, {"$push": {"projects": proj_entry}})
    else:
        users_col.insert_one(
            {
                "user_id": str(uuid.uuid4()),
                "name": name,
                "email": email_l,
                "password": hash_password(otp),
                "status": "pending",
                "projects": [proj_entry],
                "created_at": datetime.datetime.utcnow(),
            }
        )
    html = f"""
    <h3>You're invited to TeamCollab 🚀</h3>
    <p>Hi <b>{name}</b>, you’ve been added as a <b>{role}</b> to <b>{proj_entry['project_name']}</b>.</p>
    <p><b>Email:</b> {email_l}<br><b>One-time password:</b> <code>{otp}</code></p>
    <p>Login 👉 <a href="{BASE_URL}">{BASE_URL}</a></p>
    """
    send_email(email_l, "TeamCollab — Project Invite", html)
    return True


def set_user_last_active(email: Optional[str], project_id: Optional[str]) -> None:
    if not email or not project_id:
        return
    users_col.update_one({"email": str(email)}, {"$set": {"last_active_project": str(project_id)}})
    refreshed = users_col.find_one({"email": str(email)})
    if refreshed:
        st.session_state.user = refreshed


# ---------------- TASK HELPERS ----------------
def get_project_tasks(project_id: str) -> List[Dict[str, Any]]:
    return list(tasks_col.find({"project_id": str(project_id)}))


def create_project_task(
    project_id: str,
    title: str,
    assigned_to: str,
    start_date: datetime.date,
    end_date: datetime.date,
    status: str,
    is_milestone: bool,
    created_by: str,
) -> None:
    now = datetime.datetime.utcnow()
    tasks_col.insert_one(
        {
            "task_id": str(uuid.uuid4()),
            "project_id": str(project_id),
            "title": title.strip(),
            "assigned_to": assigned_to,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": status,
            "is_milestone": is_milestone,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
    )


def update_task_status(task_id: str, new_status: str) -> None:
    tasks_col.update_one(
        {"task_id": task_id},
        {"$set": {"status": new_status, "updated_at": datetime.datetime.utcnow()}},
    )


# ---------------- CHAT & RESOURCES HELPERS ----------------
def add_chat_message(project_id: str, user_email: Optional[str], text: str) -> None:
    if not user_email:
        return
    if not text.strip():
        return
    chat_col.insert_one(
        {
            "project_id": str(project_id),
            "user_email": str(user_email),
            "text": text.strip(),
            "created_at": datetime.datetime.utcnow(),
        }
    )


def get_chat_messages(project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    return list(chat_col.find({"project_id": str(project_id)}).sort("created_at", -1).limit(limit))


def add_resource(
    project_id: str,
    user_email: Optional[str],
    title: str,
    link: str,
    notes: str,
) -> None:
    if not user_email:
        return
    if not title.strip() and not link.strip() and not notes.strip():
        return
    resources_col.insert_one(
        {
            "resource_id": str(uuid.uuid4()),
            "project_id": str(project_id),
            "user_email": str(user_email),
            "title": title.strip() or "(Untitled)",
            "link": link.strip(),
            "notes": notes.strip(),
            "created_at": datetime.datetime.utcnow(),
        }
    )


def get_resources(project_id: str) -> List[Dict[str, Any]]:
    return list(resources_col.find({"project_id": str(project_id)}).sort("created_at", -1))


# ---------------- SESSION INIT ----------------
if "user" not in st.session_state:
    st.session_state.user = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "active_project" not in st.session_state:
    st.session_state.active_project = None
if "suggested_projects" not in st.session_state:
    st.session_state.suggested_projects = []

# ---------------- UI HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🤖 TeamCollab</h1>"
    "<p style='text-align:center;color:gray;'>Smart Multi-Project Collaboration Hub</p><hr>",
    unsafe_allow_html=True,
)
menu = st.sidebar.radio(
    "Menu",
    ["🏠 Register (Lead)", "🔐 Login", "📊 Dashboard", "🧠 Task Planner", "📘 Literature Review"],
)

# ---------------- REGISTER ----------------
if menu == "🏠 Register (Lead)":
    st.subheader("👑 Register as Team Lead")
    with st.form("lead_register"):
        team_name = st.text_input("Team Name")
        lead_name = st.text_input("Full Name")
        lead_email = st.text_input("Email Address")
        domain = st.selectbox("Select Domain", DOMAINS)
        generate = st.form_submit_button("✨ Generate Project Ideas")
    if generate and domain:
        st.session_state.suggested_projects = suggest_projects(domain)
        st.rerun()

    if st.session_state.suggested_projects:
        st.markdown("### 💡 Suggested Project Ideas:")
        for i, idea in enumerate(st.session_state.suggested_projects or [], 1):
            st.write(f"{i}. {idea}")

    with st.form("final_create"):
        project_title = st.text_input("📌 Choose or enter your own project title")
        submit = st.form_submit_button("🚀 Create Project")
    if submit:
        if not all([team_name, lead_name, lead_email, domain, project_title]):
            st.error("All fields are required.")
        elif not valid_email(lead_email):
            st.error("Invalid email format.")
        elif users_col.find_one({"email": lead_email.lower()}):
            st.warning("Email already registered.")
        else:
            tid, pid = str(uuid.uuid4()), str(uuid.uuid4())
            pwd = generate_password()
            hashed = hash_password(lead_email)
            teams_col.insert_one(
                {
                    "team_id": tid,
                    "project_id": pid,
                    "team_name": team_name.strip(),
                    "project_name": project_title.strip(),
                    "domain": domain,
                    "lead_email": lead_email.lower(),
                    "created_at": datetime.datetime.utcnow(),
                }
            )
            users_col.insert_one(
                {
                    "user_id": str(uuid.uuid4()),
                    "name": lead_name.strip(),
                    "email": lead_email.lower(),
                    "password": hashed,
                    "status": "accepted",
                    "projects": [
                        {
                            "project_id": pid,
                            "team_id": tid,
                            "role": "lead",
                            "domain": domain,
                            "project_name": project_title.strip(),
                        }
                    ],
                    "last_active_project": pid,
                    "created_at": datetime.datetime.utcnow(),
                }
            )
            send_email(
                lead_email,
                "🎉 TeamCollab — Account Created",
                f"<h3>Welcome!</h3><p>Your project <b>{project_title}</b> under <b>{domain}</b> is ready."
                f"<br>Email: {lead_email}<br>Password: <code>{pwd}</code></p>",
            )
            st.success("✅ Project created. Credentials sent.")
            st.session_state.suggested_projects = []
            time.sleep(1)
            st.rerun()

# ---------------- LOGIN ----------------
elif menu == "🔐 Login":
    st.subheader("🔑 Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        sub = st.form_submit_button("Login")
    if sub:
        user_doc = users_col.find_one({"email": email.lower().strip()}) if email else None
        if not user_doc:
            st.error("User not found.")
        elif safe_get(user_doc, "password") != hash_password(pwd):
            st.error("Wrong password.")
        else:
            st.session_state.logged_in = True
            st.session_state.user = user_doc
            # Try to set active_project as a full project dict (not just id)
            projects = safe_get(user_doc, "projects", []) or []
            desired_pid = safe_get(user_doc, "last_active_project") or (projects[0].get("project_id") if projects else None)
            active_proj = None
            if desired_pid:
                # find matching project object in user.projects
                active_proj = get_full_project_dict(user_doc, desired_pid)
            # fallback to first project if none found
            if not active_proj and projects:
                active_proj = projects[0]
            # store the full project dict in session
            st.session_state.active_project = active_proj
            st.success(f"Welcome {safe_get(user_doc,'name','User')} 👋")
            time.sleep(1)
            st.rerun()

# ---------------- DASHBOARD ----------------
elif menu == "📊 Dashboard":
    if not st.session_state.logged_in or not isinstance(st.session_state.user, dict):
        st.warning("Please login first.")
        st.stop()

    # refresh user
    user_doc = users_col.find_one({"email": safe_get(st.session_state.user, "email")}) or st.session_state.user
    st.session_state.user = user_doc
    st.write(f"👋 Hey, **{safe_get(user_doc,'name','User')}**")

    # PROJECT SELECTION
    projects = safe_get(user_doc, "projects", []) or []
    if not projects:
        st.info("No projects found.")
    else:
        opts = [
            f"{safe_get(p,'project_name','')} | {safe_get(p,'domain','')} | "
            f"{(safe_get(p,'role','') or '').capitalize()}"
            for p in projects
        ]
        sel = st.selectbox("🔀 Select Project", opts)
        sel_proj = projects[opts.index(sel)]
        pid = str(safe_get(sel_proj, "project_id", ""))
        st.session_state.active_project = sel_proj  # Store full project dict
        set_user_last_active(safe_get(user_doc, "email"), pid)
        st.markdown(f"### 🧠 {safe_get(sel_proj,'project_name')}")
        st.markdown(f"- **Domain:** {safe_get(sel_proj,'domain')}")
        st.markdown(f"- **Role:** {(safe_get(sel_proj,'role','') or '').capitalize()}")

    st.divider()
    # Extract project_id from the full project dict
    active_proj = st.session_state.active_project or {}
    pid_str = str(safe_get(active_proj, "project_id", "")) if isinstance(active_proj, dict) else str(active_proj or "")
    if not pid_str:
        st.info("Select or create a project to view dashboard.")
        st.stop()

    role = user_role_for_project(user_doc, pid_str)
    current_email: Optional[str] = safe_get(user_doc, "email")

    # ---------- TEAM MEMBERS ----------
    st.subheader("👥 Team Members")
    mems: List[Dict[str, Any]] = []
    for m in (users_col.find({"projects.project_id": pid_str}) or []):
        pr = next(
            (x for x in (safe_get(m, "projects", []) or []) if safe_get(x, "project_id") == pid_str),
            {},
        )
        mems.append(
            {
                "Name": safe_get(m, "name"),
                "Email": safe_get(m, "email"),
                "Role": (safe_get(pr, "role") or "").capitalize(),
            }
        )
    if mems:
        st.dataframe(pd.DataFrame(mems))
    else:
        st.info("No members yet.")

    st.divider()

    # ---------- PERSONAL PROGRESS (ALL ROLES) ----------
    st.subheader("👤 My Skills & Notes")
    my_rec = ensure_progress_record(pid_str, current_email)
    my_cur_skills: List[str] = safe_get(my_rec, "skills", []) or []
    col1, col2 = st.columns(2)
    with col1:
        my_skills_text = st.text_input(
            "Skills (comma separated)",
            value=", ".join(my_cur_skills),
            key="my_skills_input",
        )
    with col2:
        my_notes = st.text_area("My Notes", value=safe_get(my_rec, "notes", ""), key="my_notes_input")

    if st.button("💾 Update My Skills & Notes"):
        sk_list = [s.strip() for s in my_skills_text.split(",") if s.strip()]
        progress_col.update_one(
            {"progress_id": safe_get(my_rec, "progress_id")},
            {
                "$set": {
                    "skills": sk_list,
                    "notes": my_notes,
                    "updated_at": datetime.datetime.utcnow(),
                }
            },
        )
        st.success("Your skills & notes updated!")
        st.rerun()

    st.markdown("---")

    # ---------- ROLE-SPECIFIC DASHBOARDS ----------
    if role == "member":
        st.subheader("🛠 Member Dashboard")
        st.markdown("### 🗒️ Feedback from Mentor / Lead")
        comments = safe_get(my_rec, "comments", []) or []
        if comments:
            for c in comments:
                st.markdown(
                    f"- **{safe_get(c,'by')}**: {safe_get(c,'text')} "
                    f"*(at {safe_get(c,'at')})*"
                )
        else:
            st.info("No feedback yet.")

    elif role == "mentor":
        st.subheader("🧑‍🏫 Mentor Dashboard")
        members = list(users_col.find({"projects.project_id": pid_str})) or []
        rows: List[Dict[str, Any]] = []
        for m in members:
            rec = progress_col.find_one({"project_id": pid_str, "user_email": safe_get(m, "email")}) or {}
            rows.append(
                {
                    "Name": safe_get(m, "name"),
                    "Email": safe_get(m, "email"),
                    "Skills": ", ".join(safe_get(rec, "skills", []) or []),
                    "Marks": safe_get(rec, "marks"),
                    "Percentage": safe_get(rec, "percentage", 0),
                }
            )
        if rows:
            df = pd.DataFrame(rows)
            st.markdown("### 📊 Members Progress Overview")
            st.dataframe(df)

            selected_email = st.selectbox("Select member to comment", [r["Email"] for r in rows])
            rec = ensure_progress_record(pid_str, selected_email)
            st.markdown(f"### 💬 Comments for **{selected_email}**")
            comments = safe_get(rec, "comments", []) or []
            if comments:
                for c in comments:
                    st.markdown(
                        f"- **{safe_get(c,'by')}**: {safe_get(c,'text')} "
                        f"*(at {safe_get(c,'at')})*"
                    )
            else:
                st.info("No comments yet.")

            comment_text = st.text_area("✏️ Add new comment")
            if st.button("💾 Post Comment"):
                if comment_text.strip():
                    progress_col.update_one(
                        {"progress_id": safe_get(rec, "progress_id")},
                        {
                            "$push": {
                                "comments": {
                                    "by": current_email,
                                    "text": comment_text.strip(),
                                    "at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                                }
                            }
                        },
                    )
                    st.success("Comment added successfully!")
                    st.rerun()
        else:
            st.info("No members yet.")

    elif role == "lead":
        st.subheader("👑 Lead Dashboard")
        team_doc = teams_col.find_one({"project_id": pid_str}) or {}

        st.markdown("### ✉️ Invite Member / Mentor")
        with st.form("invite"):
            nm = st.text_input("Invitee Name")
            em = st.text_input("Invitee Email")
            rl = st.selectbox("Role", ["member", "mentor"])
            sub = st.form_submit_button("Send Invite")
        if sub:
            if not valid_email(em):
                st.error("Invalid email.")
            else:
                create_invite_user(em, nm, rl, team_doc)
                st.success("Invite sent.")
                st.rerun()

        st.markdown("### 📊 Progress Overview")
        recs = list(progress_col.find({"project_id": pid_str})) or []
        if recs:
            df = pd.DataFrame(
                [
                    {
                        "Email": safe_get(r, "user_email"),
                        "Skills": ", ".join(safe_get(r, "skills", []) or []),
                        "Marks": safe_get(r, "marks"),
                        "Percent": safe_get(r, "percentage", 0),
                    }
                    for r in recs
                ]
            )
            st.dataframe(df)
        else:
            st.info("No progress data yet.")
    else:
        st.info("Select a project to view role-specific dashboard.")

    # ---------- PROJECT PROGRESS CHART (Lead & Mentor) ----------
    st.divider()
    if role in ("lead", "mentor"):
        st.subheader("📈 Project Progress Chart")
        recs = list(progress_col.find({"project_id": pid_str})) or []
        if recs:
            df = pd.DataFrame(
                [
                    {
                        "Email": safe_get(r, "user_email"),
                        "Percentage": safe_get(r, "percentage", 0.0),
                    }
                    for r in recs
                ]
            )
            st.bar_chart(df.set_index("Email"))
        else:
            st.info("No progress data yet to chart.")

    # ---------- PROJECT TASKS + GANTT ----------
    st.divider()
    st.subheader("🗓 Project Timeline (Gantt) & Tasks")

    project_tasks = get_project_tasks(pid_str)
    if project_tasks:
        try:
            gantt_rows: List[Dict[str, Any]] = []
            today = datetime.date.today()
            for t in project_tasks:
                start_str = str(safe_get(t, "start_date") or today.isoformat())
                end_str = str(safe_get(t, "end_date") or today.isoformat())
                try:
                    start_date_parsed = datetime.date.fromisoformat(start_str)
                    end_date_parsed = datetime.date.fromisoformat(end_str)
                except ValueError:
                    start_date_parsed = end_date_parsed = today
                gantt_rows.append(
                    {
                        "Task": safe_get(t, "title", ""),
                        "Assignee": safe_get(t, "assigned_to", ""),
                        "Start": start_date_parsed,
                        "End": end_date_parsed,
                        "Status": safe_get(t, "status", "Not Started"),
                    }
                )
            df_gantt = pd.DataFrame(gantt_rows)
            if not df_gantt.empty:
                fig = px.timeline(
                    df_gantt,
                    x_start="Start",
                    x_end="End",
                    y="Task",
                    color="Status",
                    hover_data=["Assignee"],
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Tasks Table")
            st.dataframe(df_gantt)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not render Gantt chart: {e}")
    else:
        st.info("No tasks yet. Create one below.")

    # Lead & Mentor: can create tasks
    if role in ("lead", "mentor"):
        st.markdown("### ➕ Create / Assign Task")
        with st.form("create_task_form"):
            task_title = st.text_input("Task Title")
            assignee_options = [m["Email"] for m in mems] if mems else [current_email or ""]
            assigned_to_raw = st.selectbox("Assign To", assignee_options)
            today = datetime.date.today()
            start_date_val = st.date_input("Start Date", today)
            end_date_val = st.date_input("End Date", today)
            status = st.selectbox("Status", TASK_STATUSES)
            is_milestone = st.checkbox("Is Milestone?")
            task_submit = st.form_submit_button("💾 Create Task")

        # normalize dates
        start_date = start_date_val if isinstance(start_date_val, datetime.date) else today
        end_date = end_date_val if isinstance(end_date_val, datetime.date) else today

        if task_submit:
            if not task_title.strip():
                st.error("Task title is required.")
            elif isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date) and end_date < start_date:
                st.error("End date cannot be before start date.")
            else:
                create_project_task(
                    project_id=pid_str,
                    title=task_title,
                    assigned_to=str(assigned_to_raw or ""),
                    start_date=start_date,
                    end_date=end_date,
                    status=status,
                    is_milestone=is_milestone,
                    created_by=str(current_email or ""),
                )
                st.success("Task created.")
                st.rerun()

    # Members: can update status of their tasks
    if role == "member":
        st.markdown("### 🔄 Update My Task Status")
        my_tasks = [t for t in project_tasks if safe_get(t, "assigned_to") == current_email]
        if my_tasks:
            for t in my_tasks:
                st.markdown(f"**{safe_get(t,'title','(no title)')}**")
                cur_status = safe_get(t, "status", "Not Started")
                idx = TASK_STATUSES.index(cur_status) if cur_status in TASK_STATUSES else 0
                new_status = st.selectbox(
                    "Status",
                    TASK_STATUSES,
                    index=idx,
                    key=f"task_status_{safe_get(t,'task_id')}",
                )
                if st.button("Update", key=f"update_task_{safe_get(t,'task_id')}"):
                    task_id = str(safe_get(t, "task_id") or "")
                    update_task_status(task_id, new_status)
                    st.success("Task status updated.")
                    st.rerun()
        else:
            st.info("No tasks assigned to you yet.")

    # ---------- PROJECT RESOURCES (ALL ROLES) ----------
    st.divider()
    st.subheader("📁 Project Resources & Submissions (All Roles)")
    with st.form("resource_form"):
        res_title = st.text_input("Title (e.g., Dataset link, PDF summary, Prompt doc)")
        res_link = st.text_input("Link (GitHub, Drive, PDF URL, etc.)")
        res_notes = st.text_area("Notes / Prompts / Description")
        res_submit = st.form_submit_button("💾 Add Resource")
    if res_submit:
        add_resource(pid_str, current_email, res_title, res_link, res_notes)
        st.success("Resource added.")
        st.rerun()

    resources = get_resources(pid_str)
    if resources:
        res_rows: List[Dict[str, Any]] = []
        for r in resources:
            ts = safe_get(r, "created_at")
            if isinstance(ts, datetime.datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = ""
            res_rows.append(
                {
                    "Title": safe_get(r, "title", ""),
                    "Link": safe_get(r, "link", ""),
                    "Notes": safe_get(r, "notes", ""),
                    "By": safe_get(r, "user_email", ""),
                    "Added At": ts_str,
                }
            )
        st.dataframe(pd.DataFrame(res_rows))
    else:
        st.info("No resources added yet.")

    # ---------- PROJECT CHAT (ALL ROLES) ----------
    st.divider()
    st.subheader("💬 Project Chat (All Roles)")
    with st.form("chat_form"):
        chat_text = st.text_area("Message")
        chat_submit = st.form_submit_button("Send")
    if chat_submit:
        add_chat_message(pid_str, current_email, chat_text)
        st.success("Message sent.")
        st.rerun()

    msgs = get_chat_messages(pid_str)
    if msgs:
        for m in msgs:
            ts = safe_get(m, "created_at")
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime.datetime) else ""
            st.markdown(
                f"**{safe_get(m,'user_email','')}** "
                f"_({ts_str})_: {safe_get(m,'text','')}"
            )
    else:
        st.info("No messages yet. Start the conversation!")

    # ---------- CREATE NEW TEAM (Smart Flow) ----------
    st.divider()
    st.subheader("➕ Create a New Team / Project")

    if "new_team_suggestions" not in st.session_state:
        st.session_state.new_team_suggestions = []

    with st.form("new_team_form"):
        new_team_name = st.text_input("Team Name")
        new_domain = st.selectbox("Select Domain", DOMAINS, key="new_domain")
        gen = st.form_submit_button("✨ Generate Ideas")

    if gen and new_domain:
        st.session_state.new_team_suggestions = suggest_projects(new_domain)
        st.rerun()

    if st.session_state.new_team_suggestions:
        st.markdown("### 💡 Suggested Project Ideas:")
        for i, idea in enumerate(st.session_state.new_team_suggestions, 1):
            st.write(f"{i}. {idea}")

    with st.form("create_new_team_final"):
        selected_project_title = st.text_input(
            "📌 Choose from above or enter your own project title"
        )
        final_submit = st.form_submit_button("🚀 Create Team")

    user_for_new = st.session_state.get("user") or {}
    user_email_new: Optional[str] = safe_get(user_for_new, "email")
    if final_submit:
        if not all(
            [
                new_team_name.strip(),
                new_domain.strip(),
                selected_project_title.strip(),
            ]
        ):
            st.error("All fields required.")
        else:
            new_team_id = str(uuid.uuid4())
            new_proj_id = str(uuid.uuid4())
            teams_col.insert_one(
                {
                    "team_id": new_team_id,
                    "project_id": new_proj_id,
                    "team_name": new_team_name.strip(),
                    "project_name": selected_project_title.strip(),
                    "domain": new_domain,
                    "lead_email": user_email_new,
                    "created_at": datetime.datetime.utcnow(),
                }
            )
            if user_email_new:
                users_col.update_one(
                    {"email": user_email_new},
                    {
                        "$push": {
                            "projects": {
                                "project_id": new_proj_id,
                                "team_id": new_team_id,
                                "role": "lead",
                                "domain": new_domain,
                                "project_name": selected_project_title.strip(),
                            }
                        }
                    },
                )
                send_email(
                    str(user_email_new),
                    "🎉 TeamCollab — New Project Created",
                    f"<h3>New Project Created!</h3>"
                    f"<p>You created <b>{selected_project_title}</b> under <b>{new_domain}</b> domain.</p>",
                )
            st.success(f"✅ New team '{new_team_name}' created successfully!")
            st.session_state.new_team_suggestions = []
            st.rerun()

#---------------Task Planner---------------------
elif menu == "🧠 Task Planner":
    if not st.session_state.get("logged_in") or not st.session_state.get("user"):
        st.warning("Please login first to access the Task Planner.")
        st.stop()

    user_tp = st.session_state.get("user", {}) or {}
    active_proj = st.session_state.get("active_project") or {}

    if not isinstance(active_proj, dict) or not active_proj.get("project_id"):
        st.warning("Please select a project in your dashboard before accessing the Task Planner.")
        st.stop()

    task_planner_page(user_tp, active_proj)

# ---------------- LITERATURE REVIEW ----------------
elif menu == "📘 Literature Review":
    from pages.literature_review import literature_review_page

    if not st.session_state.get("logged_in") or not st.session_state.get("user"):
        st.warning("Please login first to access the Literature Review.")
        st.stop()

    user_lr = st.session_state.get("user", {}) or {}
    active_proj = st.session_state.get("active_project") or {}

    if not isinstance(active_proj, dict) or not active_proj.get("project_id"):
        st.warning("Please select a project from your dashboard before accessing the Literature Review.")
        st.stop()

    literature_review_page(user_lr, active_proj)

# Default welcome screen
else:
    st.markdown("## 👋 Welcome to TeamCollab!")
    st.markdown("""
    **TeamCollab** is an AI-powered academic project collaboration platform designed to help teams:
    - 📋 Plan projects with AI-assisted scrum backlog generation
    - 🎯 Set and track objectives with intelligent recommendations
    - 📚 Conduct literature reviews efficiently
    - 👥 Manage team members and track progress
    - 💬 Collaborate in real-time
    
    ### Getting Started:
    1. **Register** as a Team Lead to create your first project
    2. **Login** with your credentials
    3. Access the **Dashboard** to manage your projects
    4. Use **Task Planner** to generate your scrum backlog
    5. Run **Literature Review** for research planning
    """)
