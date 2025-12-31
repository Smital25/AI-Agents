from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, Optional, List

import streamlit as st
import pandas as pd
import requests
from pymongo import MongoClient
import gridfs
from bson import ObjectId

# ------------------ DB CONNECTION ------------------
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
tasks_col = db["tasks"]
progress_col = db["progress"]
resources_col = db["resources"]

fs = gridfs.GridFS(db)

# ------------------ HELPERS ------------------
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj[key] if isinstance(obj, dict) and key in obj else default


def ai_chat(prompt: str, model: str = "llama3.2") -> str:
    """
    Generic AI chat wrapper using local Ollama /api/chat.
    Returns plain text, max ~4000 chars.
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=45,
        )
        resp.raise_for_status()
        j = resp.json()
        content = safe_get(safe_get(j, "message", {}), "content", "")
        return str(content)[:4000]
    except Exception as e:
        return f"(AI Error: {e})"


def ai_review_submission(title: str, notes: str, link: str) -> str:
    """
    Detailed but student-friendly self review.
    """
    prompt = f"""
Act as a helpful but strict academic mentor.

A student has submitted the following work:

Title: {title}
Notes: {notes}
Link: {link}

Give feedback with the sections:

1. Short Summary (3–4 lines)
2. Strengths (3–5 bullet points)
3. Weaknesses / Gaps (3–5 bullet points)
4. 5 Concrete Actionable Improvements
5. Suggested additional topics / tools to learn

Write clearly and practically.
"""
    return ai_chat(prompt)


def ai_judge_submission(
    title: str,
    notes: str,
    link: str,
    related_task: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Strict judge: correctness, accuracy %, badge, risk level.
    """
    task_info = ""
    if related_task:
        task_info = (
            f"\nExpected Task Title: {safe_get(related_task, 'title', '')}"
            f"\nExpected Task Description: {safe_get(related_task, 'description', '')}"
        )

    prompt = f"""
You are an expert evaluator. Judge this submission.

Submission:
- Title: {title}
- Notes: {notes}
- Link: {link}
{task_info}

You MUST respond in this exact format (plain text, no JSON):

Verdict: <CORRECT | PARTIALLY_CORRECT | INCORRECT>
Accuracy: <integer 0-100>
Badge: <GOLD | SILVER | BRONZE | RED_FLAG>
RiskLevel: <GREEN | YELLOW | RED>
PlagiarismSuspicion: <LOW | MEDIUM | HIGH>

Reasoning:
- <point 1>
- <point 2>
- <point 3>
- <point 4>
- <point 5>

ImprovementPlan:
- <step 1>
- <step 2>
- <step 3>
"""
    return ai_chat(prompt, model="llama3.2")


def ai_next_steps(title: str, notes: str, link: str) -> str:
    """
    Suggest next tasks / learning steps.
    """
    prompt = f"""
You are an academic project coach.

Based on this submission:

Title: {title}
Notes: {notes}
Link: {link}

Suggest:

1. 3–5 immediate next tasks the student should do on this project.
2. For each task, 1–2 lines describing what to deliver.
3. 3 learning resources/topics to explore next (short list).
"""
    return ai_chat(prompt)


def ensure_progress_record(project_id: str, email: str) -> Dict[str, Any]:
    rec = progress_col.find_one({"project_id": project_id, "user_email": email})
    if rec:
        return rec
    new = {
        "progress_id": str(uuid.uuid4()),
        "project_id": project_id,
        "user_email": email,
        "skills": [],
        "percentage": 0,
        "comments": [],
        "notes": "",
        "created_at": datetime.datetime.utcnow(),
    }
    progress_col.insert_one(new)
    return new


def classify_file_type(filename: str, mime: str | None = None) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".doc") or name.endswith(".docx"):
        return "doc"
    if name.endswith(".ppt") or name.endswith(".pptx"):
        return "ppt"
    if name.endswith(".txt") or "text" in (mime or ""):
        return "text"
    if name.endswith(".py") or name.endswith(".ipynb") or name.endswith(".js"):
        return "code"
    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
        return "image"
    if name.endswith(".zip") or name.endswith(".rar"):
        return "archive"
    return "other"


# ------------------ PAGE START ------------------
st.set_page_config(page_title="Member Dashboard", page_icon="🧑‍💻")

if "user" not in st.session_state or not st.session_state.user:
    st.error("Session expired. Please log in again.")
    st.stop()

if "active_project" not in st.session_state or not st.session_state.active_project:
    st.error("No active project selected.")
    st.stop()

user = st.session_state.user
active_project = st.session_state.active_project

# active_project can be dict (from app.py) or just id (older pattern)
if isinstance(active_project, dict):
    pid = str(safe_get(active_project, "project_id", ""))
else:
    pid = str(active_project or "")

if not pid:
    st.error("Active project is missing project_id.")
    st.stop()

# Validate role
role = None
for p in safe_get(user, "projects", []) or []:
    try:
        if str(safe_get(p, "project_id")) == pid:
            role = safe_get(p, "role")
            break
    except Exception:
        continue

if role != "member":
    st.error("You are not authorized to view this page. (Role != member)")
    st.stop()

team_doc = teams_col.find_one({"project_id": pid}) or {}
project_name = safe_get(team_doc, "project_name", "Untitled Project")

st.title("🧑‍💻 Member Dashboard")
st.subheader(f"Project: {project_name}")
st.divider()

# ===============================================================
# 1️⃣ MY TASKS — VIEW + UPDATE STATUS
# ===============================================================
st.header("📝 My Tasks")

my_email = safe_get(user, "email", "")
tasks = list(tasks_col.find({"project_id": pid, "assigned_to": my_email}))

if not tasks:
    st.info("No tasks assigned to you yet.")
else:
    for t in tasks:
        task_id = safe_get(t, "task_id", "")
        with st.expander(f"📌 {safe_get(t, 'title', '(Untitled Task)')}"):
            st.write(f"**Start:** {safe_get(t, 'start_date', '-')}")
            st.write(f"**End:** {safe_get(t, 'end_date', '-')}")
            st.write(f"**WBS:** {safe_get(t, 'wbs', '-')}")
            st.write(f"**Current Status:** `{safe_get(t, 'status', 'Not Started')}`")
            st.write(f"**Description:** {safe_get(t, 'description', '')}")

            status_options = ["Not Started", "In Progress", "Blocked", "Completed"]
            current_status = safe_get(t, "status") or "Not Started"
            if current_status not in status_options:
                current_status = "Not Started"

            new_status = st.selectbox(
                f"Update Status — {task_id}",
                status_options,
                index=status_options.index(current_status),
                key=f"select_{task_id}",
            )

            if st.button(f"Save Status — {task_id}", key=f"btn_{task_id}"):
                tasks_col.update_one(
                    {"task_id": task_id},
                    {"$set": {"status": new_status}},
                )
                st.success("Status updated!")
                st.rerun()

st.divider()

# ===============================================================
# 2️⃣ SUBMIT WORK / RESOURCES (TEXT + LINK + FILES)
# ===============================================================
st.header("📂 Submit Work / Upload Resources")

with st.form("submit_work"):
    title = st.text_input("Title (e.g., Module Report, Dataset Link, Demo Video)")
    link = st.text_input("Resource Link (GitHub / Drive / Website)")
    notes = st.text_area("Notes / Summary / Description")

    # Map tasks to selection
    task_label = "-- No specific task --"
    task_map: Dict[str, str] = {task_label: ""}
    for t in tasks:
        tid = safe_get(t, "task_id", "")
        ttitle = safe_get(t, "title", "(Untitled Task)")
        task_map[f"{ttitle} [{tid}]"] = tid

    selected_task_label = st.selectbox(
        "Link to which task? (optional)",
        list(task_map.keys()),
    )
    linked_task_id = task_map.get(selected_task_label, "")

    uploaded_files = st.file_uploader(
        "Upload Files (pdf, docx, ppt, images, txt, zip, code)",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "ppt", "pptx", "png", "jpg", "jpeg", "txt", "zip", "py", "ipynb", "js"],
    )

    submit_btn = st.form_submit_button("Submit Work")

if submit_btn:
    if not (title.strip() or link.strip() or notes.strip() or uploaded_files):
        st.error("Please provide at least title/link/notes or upload a file.")
    else:
        ensure_progress_record(pid, my_email)

        base_doc: Dict[str, Any] = {
            "resource_id": str(uuid.uuid4()),
            "project_id": pid,
            "user_email": my_email,
            "title": title or "(Untitled)",
            "link": link,
            "notes": notes,
            "task_id": linked_task_id or None,
            "created_at": datetime.datetime.utcnow(),
        }

        # If files uploaded -> one document per file
        if uploaded_files:
            for up in uploaded_files:
                try:
                    file_bytes = up.read()
                    file_id = fs.put(file_bytes, filename=up.name, content_type=up.type)
                    doc = base_doc.copy()
                    doc["resource_id"] = str(uuid.uuid4())
                    doc["file_id"] = str(file_id)
                    doc["filename"] = up.name
                    doc["file_type"] = classify_file_type(up.name, up.type)
                    resources_col.insert_one(doc)
                except Exception:
                    # fallback: at least store metadata
                    doc = base_doc.copy()
                    doc["resource_id"] = str(uuid.uuid4())
                    doc["filename"] = up.name
                    doc["file_error"] = True
                    resources_col.insert_one(doc)
        else:
            resources_col.insert_one(base_doc)

        st.success("Submission uploaded!")
        st.rerun()

st.divider()

# ===============================================================
# 3️⃣ MY SUBMISSIONS + AI AGENT + FEEDBACK
# ===============================================================
st.header("🧠 My Submissions, AI Feedback & Mentor Comments")

subs = list(
    resources_col.find({"project_id": pid, "user_email": my_email}).sort("created_at", -1)
)

if not subs:
    st.info("You haven’t submitted anything yet.")
else:
    for r in subs:
        rid = safe_get(r, "resource_id", str(uuid.uuid4()))
        title = safe_get(r, "title", "(Untitled)")
        notes = safe_get(r, "notes", "")
        link = safe_get(r, "link", "")

        created_at = safe_get(r, "created_at")
        if isinstance(created_at, datetime.datetime):
            created_str = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            created_str = str(created_at or "")

        with st.expander(f"📄 {title}  —  {created_str}"):
            st.write(f"**Notes:** {notes or '(none)'}")
            if link:
                st.write(f"🔗 [Open Link]({link})")

            # Attachments if any
            file_id_str = safe_get(r, "file_id")
            if file_id_str:
                st.write(f"📎 File: {safe_get(r, 'filename', '(no name)')}")
                try:
                    file_obj = fs.get(ObjectId(file_id_str))
                    st.download_button(
                        label="⬇ Download Attachment",
                        data=file_obj.read(),
                        file_name=file_obj.filename,
                        key=f"dl_{rid}",
                    )
                except Exception:
                    st.warning("File not available (GridFS error).")

            # Mentor feedbacks
            feedbacks = safe_get(r, "feedbacks", []) or []
            if feedbacks:
                st.markdown("### 💬 Mentor Feedback")
                for fb in feedbacks:
                    at = safe_get(fb, "at")
                    by = safe_get(fb, "by", "mentor")
                    txt = safe_get(fb, "text", "")
                    st.write(f"- **{by}** ({at}): {txt}")

            # Mentor AI review (from mentor dashboard)
            mentor_ai = safe_get(r, "mentor_review")
            if mentor_ai:
                st.markdown("### 🤖 Mentor AI Review")
                st.info(mentor_ai)

            # Self AI review
            self_ai = safe_get(r, "self_ai_review")
            if self_ai:
                with st.expander("📘 Your Previous AI Self Review"):
                    st.write(self_ai)

            # AI judgment (accuracy, badge, risk)
            judge_ai = safe_get(r, "ai_judge_report")
            if judge_ai:
                with st.expander("🎯 Previous AI Judge / Score"):
                    st.write(judge_ai)

            # AI next steps
            next_ai = safe_get(r, "ai_next_steps")
            if next_ai:
                with st.expander("🧭 Previous AI Next Steps"):
                    st.write(next_ai)

            # Buttons row
            col1, col2, col3 = st.columns(3)

            # ---- Self Review ----
            if col1.button("🤖 AI Self Review", key=f"ai_self_{rid}"):
                prompt = (
                    "You are reviewing your own work. Give honest, detailed feedback.\n\n"
                    f"Title: {title}\nNotes: {notes}\nLink: {link}"
                )
                review = ai_review_submission(title, notes, link)
                resources_col.update_one(
                    {"resource_id": rid},
                    {"$set": {"self_ai_review": review}},
                    upsert=True,
                )
                st.success("AI self review generated.")
                st.write(review)

            # ---- Judge (Accuracy / Badge / Risk) ----
            # try to load related task if present
            related_task = None
            task_id = safe_get(r, "task_id")
            if task_id:
                related_task = tasks_col.find_one({"task_id": task_id})

            if col2.button("🎯 AI Judge My Work", key=f"ai_judge_{rid}"):
                report = ai_judge_submission(title, notes, link, related_task)
                resources_col.update_one(
                    {"resource_id": rid},
                    {"$set": {"ai_judge_report": report}},
                    upsert=True,
                )
                st.success("AI judgment generated (accuracy, badge, risk).")
                st.write(report)

            # ---- Next Steps ----
            if col3.button("🧭 AI Next Steps", key=f"ai_next_{rid}"):
                next_plan = ai_next_steps(title, notes, link)
                resources_col.update_one(
                    {"resource_id": rid},
                    {"$set": {"ai_next_steps": next_plan}},
                    upsert=True,
                )
                st.success("AI next-step guidance generated.")
                st.write(next_plan)

st.divider()

# ===============================================================
# 4️⃣ PROGRESS & PERSONAL NOTES
# ===============================================================
st.header("📊 My Progress & Notes")

progress = ensure_progress_record(pid, my_email)

# Basic task-based completion
my_task_docs = tasks
total_my_tasks = len(my_task_docs)
completed_my_tasks = sum(
    1 for t in my_task_docs if safe_get(t, "status") == "Completed"
)
task_progress_pct = (
    int(round((completed_my_tasks / total_my_tasks) * 100))
    if total_my_tasks > 0
    else 0
)

st.write(f"**Task Completion:** {completed_my_tasks}/{total_my_tasks} tasks → {task_progress_pct}%")
st.progress(task_progress_pct / 100 if task_progress_pct else 0.0)

skills_input = st.text_input(
    "Your Skills (comma separated)", ", ".join(progress.get("skills", []))
)
notes_input = st.text_area("Personal Notes", progress.get("notes", ""))

if st.button("💾 Save Progress & Notes"):
    skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]

    progress_col.update_one(
        {"progress_id": progress["progress_id"]},
        {
            "$set": {
                "skills": skills_list,
                "notes": notes_input,
                "percentage": task_progress_pct,
                "updated_at": datetime.datetime.utcnow(),
            }
        },
    )
    st.success("Progress & notes updated!")
    st.rerun()

st.info("✔ Member dashboard loaded successfully.")
