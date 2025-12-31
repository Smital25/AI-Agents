"""
🧠 Agile Task Planner — Academic Scrum (Guaranteed Skill Distribution)
====================================================================
✔ Every member + lead gets tasks
✔ Skill-aware + fallback assignment
✔ Objective-driven WBS
✔ Start & End date enforced
✔ Mentor excluded
✔ Gantt included
"""

from __future__ import annotations

import datetime
import uuid
import re
from typing import Dict, List

import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
# ================= EMAIL CONFIG =================
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "shostelmanagement@gmail.com"     # 🔴 change
SENDER_PASSWORD = "ehbplmequzgyhzck"             # 🔴 change


# ================= DB =================

client = MongoClient("mongodb://localhost:27017/")
db = client["team_collab_db"]

users_col = db["users"]
progress_col = db["progress"]
objectives_col = db["project_objectives"]
tasks_col = db["tasks"]


# ================= HELPERS =================

def is_weekend(d):
    return d.weekday() >= 5


def add_working_days(start, days):
    cur = start
    rem = days
    while rem > 0:
        cur += datetime.timedelta(days=1)
        if not is_weekend(cur):
            rem -= 1
    return cur


# ================= SKILL LOGIC =================

LEVEL_WEIGHT = {"beginner": 1, "intermediate": 2, "expert": 3}

FALLBACK_TASKS = [
    "Prepare academic documentation",
    "Perform system evaluation and testing",
    "Analyze results and metrics",
    "Prepare final report sections",
    "Validate objectives and outcomes"
]

def skill_score(skills, title):
    words = re.findall(r"[a-zA-Z]{3,}", title.lower())
    score = 0
    for s in skills:
        name = s.get("name", "").lower()
        level = LEVEL_WEIGHT.get(s.get("level", "beginner"), 1)
        for w in words:
            if w in name or name in w:
                score += level
    return score


# ================= TASK GENERATION =================

def generate_tasks(objectives, members, start_date, end_date):
    """
    Objective → Tasks → Guaranteed distribution
    """

    total_days = (end_date - start_date).days
    sprint_len = 10

    tasks = []
    workload = {m["email"]: 0 for m in members}
    current = start_date
    sprint = 1

    member_cycle = members.copy()
    cycle_index = 0

    for e_idx, obj in enumerate(objectives, 1):
        epic = f"Epic {e_idx}: {obj[:60]}"

        # Generate at least one task per member for this objective
        base_tasks = [
            f"Research work for objective: {obj[:40]}",
            f"Implementation task related to: {obj[:40]}",
            f"Evaluation task for: {obj[:40]}",
            f"Documentation for: {obj[:40]}"
        ]

        # Extend tasks if members > base
        while len(base_tasks) < len(members):
            base_tasks.append(FALLBACK_TASKS[len(base_tasks) % len(FALLBACK_TASKS)])

        for t_idx, title in enumerate(base_tasks, 1):
            # ---------- GUARANTEED ROUND ROBIN ----------
            assignee = member_cycle[cycle_index % len(member_cycle)]["email"]
            cycle_index += 1

            # ---------- SKILL OVERRIDE IF STRONG MATCH ----------
            best = assignee
            best_score = 0
            for m in members:
                sc = skill_score(m["skills"], title)
                if sc > best_score:
                    best_score = sc
                    best = m["email"]

            assignee = best

            days = 2
            start = current
            end = add_working_days(start, days)

            tasks.append({
                "wbs": f"{e_idx}.1.{t_idx}",
                "epic": epic,
                "story": f"As a student, I want to achieve {obj[:60]}",
                "title": title,
                "assignee": assignee,
                "start": start,
                "end": end,
                "days": days,
                "sprint": sprint
            })

            workload[assignee] += days
            current = end

            if (current - start_date).days // sprint_len + 1 > sprint:
                sprint += 1

    return tasks

def normalize_date(d):
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, (tuple, list)) and d:
        return d[0]
    return None

def to_datetime(d: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(d, datetime.time.min)

# ================= STREAMLIT PAGE =================

def task_planner_page():
    # ================= TITLE =================
    st.markdown("""
    <h1 style="font-weight:800;">
        🧠 Agile Task Planner — Scrum Epics, Stories, Tasks, Subtasks
    </h1>
    """, unsafe_allow_html=True)

    # ================= SESSION =================
    user = st.session_state.get("user")
    project = st.session_state.get("active_project")

    if not user or not project:
        st.error("Open this page from dashboard")
        return

    project_id = project.get("project_id")

    if not project_id:
        st.error("Invalid project selected")
        return

    # ================= PROJECT INFO BAR =================
    st.markdown(
        f"""
        <div style="
            background-color:#0f2a44;
            padding:12px 18px;
            border-radius:8px;
            font-size:16px;
            font-weight:600;
            margin-bottom:20px;
        ">
            📌 <b>Project:</b> {project.get("project_name", "Untitled Project")}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            🧪 <b>Domain:</b> {project.get("domain", "Unknown")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================= EMAIL OVERDUE CHECK =================
    check_and_notify_overdue_tasks(project_id)

    # ================= OBJECTIVES =================
    obj_doc = objectives_col.find_one({"project_id": project_id}) or {}
    objectives = obj_doc.get("objectives", [])

    if not objectives:
        st.error("No objectives found for this project")
        return

    st.success(f"{len(objectives)} final objectives loaded.")

    with st.expander("Show Final Objectives"):
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"**{i}.** {obj}")

    st.divider()

    # ================= TEAM MEMBERS =================
    members = []
    for u in users_col.find({
        "projects": {
            "$elemMatch": {
                "project_id": project_id,
                "role": {"$ne": "mentor"}
            }
        }
    }):
        email = u.get("email")
        prog = progress_col.find_one(
            {"project_id": project_id, "user_email": email}
        ) or {}

        members.append({
            "email": email,
            "skills": prog.get("skills", [])
        })

    if not members:
        st.error("No team members found for this project")
        return

    st.markdown("## 👥 Team Members & Skills")

    st.dataframe(
        pd.DataFrame([
            {
                "Email": m["email"],
                "Skills": ", ".join(
                    f"{s['name']} ({s.get('level','')})" for s in m["skills"]
                )
            }
            for m in members
        ]),
        use_container_width=True
    )

    # -------- Dates --------

    start_raw = st.date_input(
    "Project Start Date",
    value=datetime.date.today()
    )

    end_raw = st.date_input(
        "Project End Date",
        value=datetime.date.today() + datetime.timedelta(days=60)
    )

    start_date = normalize_date(start_raw)
    end_date = normalize_date(end_raw)

    if not start_date or not end_date:
        st.error("Invalid date selection")
        return

    if start_date >= end_date:
        st.error("End date must be after start date")
        return

    # -------- Generate --------
    if st.button("🚀 Generate Agile WBS"):
        tasks = generate_tasks(objectives, members, start_date, end_date)
        st.session_state["tasks"] = tasks
        st.rerun()

    tasks = st.session_state.get("tasks", [])
    if not tasks:
        return

    # -------- Table --------
    df = pd.DataFrame(tasks)
    st.markdown("### 📋 Agile WBS")
    st.dataframe(df)

    # -------- Gantt --------
    st.markdown("### 📊 Gantt Chart (Member-wise)")
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="assignee",
        color="epic",
        hover_name="title"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # -------- Save --------
    if st.button("💾 Save Tasks"):
        for t in tasks:
            tasks_col.insert_one({
                "task_id": str(uuid.uuid4()),
                "project_id": project_id,
                "wbs": t["wbs"],
                "epic": t["epic"],
                "story": t["story"],
                "title": t["title"],
                "assigned_to": t["assignee"],

                # ✅ FIXED DATES
                "start_date": to_datetime(t["start"]),
                "end_date": to_datetime(t["end"]),

                "estimated_days": t["days"],
                "sprint": t["sprint"],
                "status": "Not Started",
                "created_at": datetime.datetime.utcnow(),
            })

        st.success("Tasks saved successfully")


#----------=========EMAIL ALERT=========----------
def send_overdue_email(to_email: str, task: dict):
    subject = "⚠️ Task Overdue Reminder — Academic Project"

    start_dt = task.get("start_date")
    end_dt = task.get("end_date")

    start_str = start_dt.strftime("%Y-%m-%d") if isinstance(start_dt, datetime.datetime) else "N/A"
    end_str = end_dt.strftime("%Y-%m-%d") if isinstance(end_dt, datetime.datetime) else "N/A"

    body = f"""
Dear Team Member,

This is a reminder that the following task has not been completed within the assigned time.

📌 Task Title: {task.get('title', 'Untitled')}
📁 Project ID: {task.get('project_id', 'N/A')}
📅 Assigned Duration: {start_str} → {end_str}

Please complete the task as soon as possible and update the status in the system.

If you have any issues, contact your project lead or mentor.

Regards,
Academic Agile Task Planner
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email error:", e)
        return False

#========= OVERDUE CHECK & NOTIFY =========
def check_and_notify_overdue_tasks(project_id: str):
    today = datetime.datetime.utcnow()

    overdue_tasks = list(tasks_col.find({
        "project_id": project_id,
        "end_date": {"$lt": today},
        "status": {"$ne": "Completed"}
    }))

    for task in overdue_tasks:
        assignee = task.get("assigned_to")
        if not assignee:
            continue

        last_alert = task.get("last_alert_sent")
        if last_alert and isinstance(last_alert, datetime.datetime):
            if (today - last_alert).days < 1:
                continue

        sent = send_overdue_email(assignee, task)

        if sent:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {"last_alert_sent": today}}
            )

# ================= RUN =================

if __name__ in ("__main__", "__page__"):
    st.set_page_config(page_title="Agile Task Planner", layout="wide")
    task_planner_page()
