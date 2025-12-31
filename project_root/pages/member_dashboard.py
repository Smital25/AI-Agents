# pages/member_dashboard.py

from __future__ import annotations
import datetime
import uuid
from typing import Any
import re

import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import gridfs
from bson import ObjectId
import ollama


# ---------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
tasks_col = db["tasks"]
progress_col = db["progress"]
resources_col = db["resources"]

fs = gridfs.GridFS(db)


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj[key] if isinstance(obj, dict) and key in obj else default


def ensure_progress_record(pid: str, email: str) -> dict:
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


def classify_file_type(name: str, mime: str = "") -> str:
    name = name.lower()
    if name.endswith(".pdf"): return "pdf"
    if name.endswith(".doc") or name.endswith(".docx"): return "doc"
    if name.endswith(".ppt") or name.endswith(".pptx"): return "ppt"
    if name.endswith(".txt") or "text" in mime: return "text"
    if name.endswith(".py") or name.endswith(".js") or name.endswith(".ipynb"): return "code"
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"): return "image"
    if name.endswith(".zip") or name.endswith(".rar"): return "archive"
    return "other"


# ---------------------------------------------------
# PAGE START
# ---------------------------------------------------
st.set_page_config(page_title="Member Dashboard", page_icon="🧑‍💻")

# SESSION CHECK
if "user" not in st.session_state or not st.session_state.user:
    st.error("Session expired. Login again.")
    st.stop()

if "active_project" not in st.session_state or not st.session_state.active_project:
    st.error("No active project selected.")
    st.stop()

user = st.session_state.user
active_project = st.session_state.active_project


# -----------------------------------------------------------
# 🔄 PROJECT SWITCHER — appears in Sidebar for all users
# -----------------------------------------------------------
user_projects = user.get("projects", [])

if len(user_projects) > 1:
    st.sidebar.markdown("### 🔄 Switch Project")

    options = []
    for p in user_projects:
        label = f"{p.get('project_name','Untitled')}  ({p.get('role','member')})"
        options.append(label)

    current_label = f"{active_project.get('project_name')}  ({active_project.get('role')})"

    selected_label = st.sidebar.selectbox("Select Project", options, index=options.index(current_label))

    if selected_label != current_label:
        chosen = None
        for p in user_projects:
            label = f"{p.get('project_name')}  ({p.get('role')})"
            if label == selected_label:
                chosen = p
                break

        if chosen:
            st.session_state.active_project = chosen

            if chosen["role"] == "lead":
                st.switch_page("pages/lead_dashboard.py")
            elif chosen["role"] == "mentor":
                st.switch_page("pages/mentor_dashboard.py")
            else:
                st.switch_page("pages/member_dashboard.py")

            st.rerun()


# ---------------------------------------------------
# NEW LEAD REDIRECT ONCE
# ---------------------------------------------------
if st.session_state.get("new_lead_redirect_once") is True:
    st.session_state.new_lead_redirect_once = False
    st.switch_page("pages/lead_dashboard.py")


# ---------------------------------------------------
# ROLE VALIDATION
# ---------------------------------------------------
pid = safe_get(active_project, "project_id", "")

role = None
for p in user.get("projects", []):
    if p.get("project_id") == pid:
        role = p.get("role")
        break

if role != "member":
    st.error("❌ You are not authorized to access Member Dashboard.")
    st.stop()

# ---------------------------------------------------
# FIXED ROLE VALIDATION
# ---------------------------------------------------
pid = safe_get(active_project, "project_id", "")
active_role = safe_get(active_project, "role", "").lower()

if active_role == "":
    st.error("❌ Role missing for this project.")
    st.stop()

elif active_role == "member":
    pass  # Allowed → Stay in member dashboard

elif active_role == "lead":
    st.warning("Redirecting to Lead Dashboard…")
    st.switch_page("pages/lead_dashboard.py")
    st.stop()

elif active_role == "mentor":
    st.warning("Redirecting to Mentor Dashboard…")
    st.switch_page("pages/mentor_dashboard.py")
    st.stop()

else:
    st.error("❌ Unknown role detected.")
    st.stop()


# ---------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------
team_doc = teams_col.find_one({"project_id": pid}) or {}
project_name = safe_get(team_doc, "project_name", "Untitled")

st.title("🧑‍💻 Member Dashboard")
st.subheader(f"Project: {project_name}")
st.divider()


# ---------------------------------------------------
# 1️⃣ MY TASKS
# ---------------------------------------------------
my_email = user.get("email", "")
tasks = list(tasks_col.find({"project_id": pid, "assigned_to": my_email}))

st.header("📝 My Tasks")

if not tasks:
    st.info("No tasks assigned yet.")
else:
    for t in tasks:
        tid = t.get("task_id")
        with st.expander(f"📌 {t.get('title')}"):
            st.write("**Status:**", t.get("status"))
            st.write("**Description:**", t.get("description"))

            options = ["Not Started", "In Progress", "Blocked", "Completed"]
            current = t.get("status", "Not Started")

            new_status = st.selectbox(
                f"Change status for {tid}",
                options,
                index=options.index(current),
                key=f"s_{tid}",
            )

            if st.button(f"Save Status ({tid})", key=f"btn_{tid}"):
                tasks_col.update_one({"task_id": tid}, {"$set": {"status": new_status}})
                st.success("Updated!")
                st.rerun()

st.divider()


# ---------------------------------------------------
# 2️⃣ SUBMIT WORK
# ---------------------------------------------------
st.header("📂 Submit Work")

with st.form("submit_form"):
    title = st.text_input("Title")
    notes = st.text_area("Notes")
    link = st.text_input("Link")

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["pdf","doc","docx","ppt","pptx","png","jpg","jpeg","txt","zip","py","js","ipynb"],
    )

    submit_btn = st.form_submit_button("Submit")

if submit_btn:
    base_doc = {
        "resource_id": str(uuid.uuid4()),
        "project_id": pid,
        "user_email": my_email,
        "title": title or "(Untitled)",
        "notes": notes,
        "link": link,
        "created_at": datetime.datetime.utcnow(),
    }

    if uploaded_files:
        for f in uploaded_files:
            file_id = fs.put(f.read(), filename=f.name, content_type=f.type)
            doc = base_doc.copy()
            doc["file_id"] = str(file_id)
            doc["filename"] = f.name
            doc["file_type"] = classify_file_type(f.name, f.type)
            resources_col.insert_one(doc)
    else:
        resources_col.insert_one(base_doc)

    st.success("Submission uploaded!")
    st.rerun()

st.divider()


# ---------------------------------------------------
# 3️⃣ VIEW MY SUBMISSIONS
# ---------------------------------------------------
st.header("🧠 My Submissions")

subs = list(resources_col.find({"project_id": pid, "user_email": my_email}).sort("created_at", -1))

if not subs:
    st.info("No submissions yet.")
else:
    for r in subs:
        rid = r["resource_id"]

        with st.expander(f"📄 {r.get('title')}"):
            st.write("**Notes:**", r.get("notes"))
            if r.get("link"):
                st.write(f"[🔗 Open Link]({r.get('link')})")

            # download if exists
            if r.get("file_id"):
                try:
                    fobj = fs.get(ObjectId(r["file_id"]))
                    st.download_button(
                        "⬇ Download File",
                        data=fobj.read(),
                        file_name=fobj.filename,
                        key=f"dl_{rid}",
                    )
                except:
                    st.warning("Missing file in storage.")

st.divider()


# ---------------------------------------------------
# 4️⃣ MEMBER PROGRESS
# ---------------------------------------------------
st.header("🛠 My Skills")

progress = ensure_progress_record(pid, my_email)

existing_skills = progress.get("skills", [])

skill_name = st.text_input("Skill name (e.g., Python, React, ML)")
skill_level = st.selectbox(
    "Skill level",
    ["Beginner", "Intermediate", "Expert"]
)

if st.button("➕ Add / Update Skill"):
    # remove old entry if exists
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

# Display skills
if existing_skills:
    st.markdown("### ✅ Your Skills")
    st.dataframe(pd.DataFrame(existing_skills))
else:
    st.info("No skills added yet.")


# ---------------------------------------------------
# 5️⃣ CREATE NEW PROJECT (MEMBER BECOMES LEAD)
# ---------------------------------------------------
st.header("🚀 Create Your Own Project (Become Lead)")

DOMAIN_LIST = [
    "AI & Machine Learning", "Web Development", "Mobile App Development",
    "Cybersecurity", "Data Science", "IoT", "Cloud Computing", "Blockchain",
    "AR/VR", "Robotics", "NLP", "Computer Vision", "Game Development",
    "Big Data", "Embedded Systems", "Quantum Computing", "DevOps",
    "HealthTech", "FinTech", "EdTech"
]


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

sel_domain = st.selectbox("Domain", DOMAIN_LIST, key="member_domain")

if st.button("✨ Generate Ideas", key="gen_member"):
    st.session_state["member_suggestions"] = suggest_projects(sel_domain)
    st.rerun()

suggestions = st.session_state.get("member_suggestions", [])

if suggestions:
    titles = [s["title"] for s in suggestions]
    selected_title = st.radio("Choose Idea", titles)
    selected = next(s for s in suggestions if s["title"] == selected_title)
    st.markdown(f"### {selected['title']}")
    st.write(selected["description"])
else:
    selected = {"title": ""}


# -------- NEW PROJECT FORM -------
with st.form("member_create_project"):
    team_name = st.text_input("Team Name")
    proj_title = st.text_input("Project Title", selected.get("title", ""))
    proj_desc = st.text_area("Project Description")
    domain_input = st.text_input("Domain", sel_domain)

    create_btn = st.form_submit_button("🚀 Create Project")

if create_btn:
    if not (team_name and proj_title and domain_input):
        st.error("All fields required.")
    else:
        new_pid = str(uuid.uuid4())
        new_tid = str(uuid.uuid4())

        teams_col.insert_one({
            "team_id": new_tid,
            "project_id": new_pid,
            "team_name": team_name,
            "project_name": proj_title,
            "domain": domain_input,
            "lead_email": my_email,
            "description": proj_desc,
            "created_at": datetime.datetime.utcnow(),
        })

        users_col.update_one(
            {"email": my_email},
            {"$push": {
                "projects": {
                    "project_id": new_pid,
                    "team_id": new_tid,
                    "role": "lead",
                    "domain": domain_input,
                    "project_name": proj_title,
                }
            }},
        )

        # redirect into lead dashboard
        st.session_state.active_project = {
            "project_id": new_pid,
            "team_id": new_tid,
            "role": "lead",
            "domain": domain_input,
            "project_name": proj_title,
        }
        st.session_state.new_lead_redirect_once = True

        st.success("🎉 Project created!")
        st.rerun()

st.info("✔ Member Dashboard Loaded Successfully.")
