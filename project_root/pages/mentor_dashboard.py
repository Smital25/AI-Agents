import streamlit as st
import datetime
import uuid
import pandas as pd
import requests
import altair as alt
from pymongo import MongoClient
from typing import Any, Dict, List, Optional
import re

# ===============================================================
# DB CONNECTION
# ===============================================================
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
tasks_col = db["tasks"]
progress_col = db["progress"]
resources_col = db["resources"]

# ===============================================================
# HELPERS
# ===============================================================
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj[key] if isinstance(obj, dict) and key in obj else default


def ai_chat(prompt: str, model: str = "llama3.2") -> str:
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
        return safe_get(safe_get(j, "message", {}), "content", "")[:4000]
    except Exception as e:
        return f"(AI Error: {e})"


# --- AI UTILITIES (unchanged logic, only formatting) ---
def ai_review_text(title, notes, link):
    prompt = f"""
Act as a strict academic mentor and review:

Title: {title}
Notes: {notes}
Link: {link}

Sections:
1. Summary (3–4 lines)
2. Score /10
3. Strengths (3–5 bullets)
4. Weaknesses (3–5 bullets)
5. Improvements (at least 5 steps)
"""
    return ai_chat(prompt)


def ai_check_accuracy(title, notes, link, expected_task=None):
    task_context = ""
    if expected_task:
        task_context = (
            f"\nExpected Title: {safe_get(expected_task,'title')}"
            f"\nExpected Desc: {safe_get(expected_task,'description')}"
            f"\nExpected Status: {safe_get(expected_task,'status')}"
        )

    prompt = f"""
Judge correctness of submission:

Title: {title}
Notes: {notes}
Link: {link}
{task_context}

Return:
Verdict: CORRECT/PARTIALLY_CORRECT/INCORRECT
Accuracy: 0–100
Reasoning: 5–8 bullets
"""
    return ai_chat(prompt)


def ai_plagiarism_check(title, notes, link):
    prompt = f"""
Check plagiarism style:

Title: {title}
Notes: {notes}
Link: {link}

Return:
1. Risk: Low/Medium/High
2. Originality Score (0–100)
3. Reasons (5 bullets)
4. Suggestions
"""
    return ai_chat(prompt)


def ai_soft_coaching_feedback(title, notes):
    prompt = f"""
Give warm supportive feedback:

Title: {title}
Notes: {notes}

Return:
- Encouragement (2–3 lines)
- What was good (3 bullets)
- What to improve (3 bullets)
- Motivational closing line
"""
    return ai_chat(prompt)


def ai_judge_project(project_name, tasks, members):
    prompt = f"""
Evaluate project:

Project: {project_name}
Tasks: {tasks}
Members: {[{'name': m.get('name'), 'email': m.get('email')} for m in members]}

Return:
1. Status
2. Depth
3. Progress quality
4. Risks (5 bullets)
5. Coordination
6. Lead effectiveness
7. Improvement plan (10 actions)
8. Final score /100
"""
    return ai_chat(prompt)


def ai_judge_member(email, project_name, tasks, submissions, progress):
    member_tasks = [t for t in tasks if t.get("assigned_to") == email]
    member_subs = [s for s in submissions if s.get("user_email") == email]

    prompt = f"""
Evaluate member:

Project: {project_name}
Member: {email}

Tasks: {member_tasks}
Submissions: {member_subs}
Progress: {progress}

Return:
1. Contribution
2. Reliability
3. Technical quality
4. Problems
5. Role suggestion
6. Improvement plan (5–7 actions)
7. Rating /10
"""
    return ai_chat(prompt)


def ai_judge_lead(lead_email, project_name, tasks, members):
    prompt = f"""
Judge Lead:

Lead: {lead_email}
Project: {project_name}
Tasks: {tasks}
Members: {members}

Return:
1. Distribution fairness
2. Planning quality
3. Leadership risks
4. Positives
5. Leadership improvement (6 steps)
6. Rating /10
"""
    return ai_chat(prompt)


def ai_risk_and_timeline_report(project_name, tasks):
    prompt = f"""
Risk & timeline report:

Project: {project_name}
Tasks: {tasks}

Return:
1. Estimated remaining days
2. Delay risk %
3. Critical path tasks
4. Bottlenecks
5. Risk heat
6. Improvement plan (8 steps)
"""
    return ai_chat(prompt)


def ai_rank_members_from_stats(member_stats):
    prompt = f"""
Rank team members:

Stats:
{member_stats}

Return:
1. Ranking
2. Top 3 reasons
3. Bottom 2 issues
4. Improvement for each member
5. Overall team health
"""
    return ai_chat(prompt)

def compute_member_stats(
    project_id: str,
    members: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    task_stats = {}
    for t in tasks:
        email = safe_get(t, "assigned_to", "unassigned") or "unassigned"
        if email not in task_stats:
            task_stats[email] = {"total": 0, "completed": 0}
        task_stats[email]["total"] += 1
        if safe_get(t, "status") == "Completed":
            task_stats[email]["completed"] += 1

    rows = []
    for m in members:
        email = safe_get(m, "email", "")
        name = safe_get(m, "name", "")
        prog = ensure_progress_record(project_id, email)
        progress_pct = float(safe_get(prog, "percentage", 0) or 0)

        stat = task_stats.get(email, {"total": 0, "completed": 0})
        tot = float(stat["total"])
        comp = float(stat["completed"])
        completion_pct = round((comp / tot) * 100, 1) if tot > 0 else 0.0

        combined_score = round(0.6 * completion_pct + 0.4 * progress_pct, 1)

        if combined_score >= 90:
            badge = "🥇 Gold"
        elif combined_score >= 75:
            badge = "🥈 Silver"
        elif combined_score >= 50:
            badge = "🥉 Bronze"
        else:
            badge = "🚩 Red Flag"

        rows.append(
            {
                "Name": name,
                "Email": email,
                "Tasks": int(tot),
                "Completed": int(comp),
                "CompletionPercent": completion_pct,
                "ProgressPercent": progress_pct,
                "Badge": badge,
                "RedFlag": combined_score < 40,
                "CombinedScore": combined_score,
            }
        )

    return rows

# ===============================================================
# PROGRESS RECORD
# ===============================================================
def ensure_progress_record(project_id: str, email: str):
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


# ===============================================================
# PAGE START — SESSION CHECK
# ===============================================================
st.set_page_config(page_title="Mentor Dashboard", page_icon="🧑‍🏫")

user = st.session_state.get("user")
active_project = st.session_state.get("active_project")

if not user:
    st.error("Session expired. Login again.")
    st.stop()

if not active_project:
    st.error("No active project selected.")
    st.stop()

# ===============================================================
# UNIVERSAL PROJECT SWITCHER (correct location)
# ===============================================================
user_projects = user.get("projects", [])

if len(user_projects) > 1:
    st.sidebar.markdown("### 🔄 Switch Project")

    label_map = {
        f"{p.get('project_name')} ({p.get('role')})": p
        for p in user_projects
    }

    current_label = f"{active_project.get('project_name')} ({active_project.get('role')})"

    selected_label = st.sidebar.selectbox(
        "Select Project",
        list(label_map.keys()),
        index=list(label_map.keys()).index(current_label)
    )

    if selected_label != current_label:
        chosen = label_map[selected_label]
        st.session_state.active_project = chosen
        role = chosen.get("role")

        if role == "lead":
            st.switch_page("pages/lead_dashboard.py")
        elif role == "mentor":
            st.switch_page("pages/mentor_dashboard.py")
        else:
            st.switch_page("pages/member_dashboard.py")

        st.rerun()

# ===============================================================
# ROLE VALIDATION (AFTER SWITCHER)
# ===============================================================
active_role = active_project.get("role", "").lower()

if active_role != "mentor":
    st.error("❌ You are not authorized to access Mentor Dashboard.")
    st.stop()

# ===============================================================
# LOAD PROJECT DATA
# ===============================================================
project_id = active_project.get("project_id")

team_doc = teams_col.find_one({"project_id": project_id}) or {}
project_name = team_doc.get("project_name", "Untitled Project")
lead_email = team_doc.get("lead_email", "--")

members = list(users_col.find({"projects.project_id": project_id}))
task_docs = list(tasks_col.find({"project_id": project_id}))
subs = list(resources_col.find({"project_id": project_id}).sort("created_at", -1))

st.title("🧑‍🏫 Mentor Dashboard")
st.subheader(f"Project: {project_name}")
st.caption(f"Lead: {lead_email}")
st.divider()

# ===============================================================
# 0️⃣ GLOBAL AI TOOLS
# ===============================================================
st.header("🤖 AI Mentor — Global Tools")

col_g1, col_g2, col_g3, col_g4 = st.columns(4)

if col_g1.button("👨‍⚖️ Judge Project"):
    st.write(ai_judge_project(project_name, task_docs, members))

if col_g2.button("🚦 Risk & Timeline Report"):
    st.write(ai_risk_and_timeline_report(project_name, task_docs))

if col_g3.button("👑 Judge Lead"):
    st.write(ai_judge_lead(lead_email, project_name, task_docs, members))

if col_g4.button("🏅 AI Rank Members"):
    stats = compute_member_stats(project_id, members, task_docs)
    st.write(ai_rank_members_from_stats(stats))

st.divider()

# ===============================================================
# Continue… (rest of your code below — unchanged)
# ===============================================================

# ===============================================================
# 1️⃣ TEAM MEMBERS OVERVIEW + BADGES / FLAGS
# ===============================================================
st.header("👥 Team Members Overview (with Badges & Flags)")

if members:
    member_stats = compute_member_stats(project_id, members, task_docs)

    df_members = pd.DataFrame(member_stats)
    st.dataframe(df_members, use_container_width=True)

    # RED FLAG LIST
    red_flags = [row for row in member_stats if row["RedFlag"]]
    if red_flags:
        st.error(f"🚩 Red Flag Members (need attention): {[r['Email'] for r in red_flags]}")
    else:
        st.success("✅ No red flag members detected based on current stats.")

    # AI explanation button
    if st.button("🧠 Explain Team Health (AI)"):
        st.write(ai_rank_members_from_stats(member_stats))
else:
    st.info("No members found for this project.")

st.divider()

# ===============================================================
# 2️⃣ PROJECT & INDIVIDUAL PROGRESS (CHARTS + GANTT + HEATMAP)
# ===============================================================
st.header("📈 Project & Member Progress")

if task_docs:
    # Overall project progress
    status_counts: Dict[str, int] = {}
    for t in task_docs:
        stt = safe_get(t, "status", "Not Started")
        status_counts[stt] = status_counts.get(stt, 0) + 1

    total_tasks = len(task_docs)
    completed = status_counts.get("Completed", 0)
    overall_pct = round((completed / total_tasks) * 100, 1) if total_tasks > 0 else 0.0

    st.subheader(f"Overall Project Progress: {overall_pct}% Completed")
    st.progress(min(overall_pct / 100, 1.0))

    status_df = pd.DataFrame(
        {
            "Status": list(status_counts.keys()),
            "Count": list(status_counts.values()),
        }
    )
    st.markdown("**Task Status Breakdown**")
    st.bar_chart(status_df.set_index("Status"))

    # ---- Per-member completion ----
    member_stats = compute_member_stats(project_id, members, task_docs)
    if member_stats:
        st.markdown("**Member-wise Task Completion & Scores**")
        st.dataframe(pd.DataFrame(member_stats), use_container_width=True)

        # Heatmap: Email vs Status Count
        heat_rows = []
        for t in task_docs:
            email = safe_get(t, "assigned_to", "unassigned")
            status = safe_get(t, "status", "Not Started")
            heat_rows.append({"Email": email, "Status": status, "Count": 1})
        if heat_rows:
            df_heat = pd.DataFrame(heat_rows)
            df_heat = (
        df_heat.groupby(["Email", "Status"], as_index=False)
        .agg({"Count": "sum"})
    )

            st.subheader("🔥 Workload & Status Heatmap")
            heat_chart = (
                alt.Chart(df_heat)
                .mark_rect()
                .encode(
                    x=alt.X("Status:N", title="Task Status"),
                    y=alt.Y("Email:N", title="Member"),
                    color=alt.Color("Count:Q", title="Number of Tasks"),
                    tooltip=["Email", "Status", "Count"],
                )
                .properties(height=300)
            )
            st.altair_chart(heat_chart, use_container_width=True)

   #-Gant Chart---
   # ---- Gantt Chart ----
st.subheader("📊 Gantt Chart (Tasks Timeline)")

gantt_rows = []

for t in task_docs:
    start_dt = safe_get(t, "start_date")
    end_dt = safe_get(t, "end_date")

    # ✅ MongoDB stores datetime.datetime
    if not isinstance(start_dt, datetime.datetime) or not isinstance(end_dt, datetime.datetime):
        continue

    gantt_rows.append(
        {
            "Task": safe_get(t, "title", ""),
            "Assignee": safe_get(t, "assigned_to", "unassigned"),
            "Status": safe_get(t, "status", "Not Started"),
            "Start": start_dt,
            "End": end_dt,
        }
    )

if gantt_rows:
    gantt_df = pd.DataFrame(gantt_rows)

    # Ensure pandas datetime
    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["End"] = pd.to_datetime(gantt_df["End"])

    gantt_df["DurationDays"] = (
        gantt_df["End"] - gantt_df["Start"]
    ).apply(lambda td: int(td / pd.Timedelta(days=1)))

    chart = (
        alt.Chart(gantt_df)
        .mark_bar()
        .encode(
            x="Start:T",
            x2="End:T",
            y=alt.Y("Task:N", sort="-x"),
            color="Assignee:N",
            tooltip=[
                "Task",
                "Assignee",
                "Status",
                "Start",
                "End",
                "DurationDays",
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)

else:
    st.info("No valid start/end dates set on tasks, so Gantt chart cannot be drawn.")

# ===============================================================
# 3️⃣ INDIVIDUAL AI JUDGE (Member / Task)
# ===============================================================
st.header("🧪 AI Judge — Member / Task")

# ---- Judge Member ----
if members:
    member_emails = [safe_get(m, "email", "") for m in members]
    sel_member = st.selectbox("Select Member to Judge", member_emails, key="judge_member_sel")

    if st.button("👤 Judge Selected Member"):
        prog = ensure_progress_record(project_id, sel_member)
        report = ai_judge_member(sel_member, project_name, task_docs, subs, prog)
        st.subheader(f"Member Evaluation — {sel_member}")
        st.write(report)

# ---- Judge Task ----
if task_docs:
    task_titles = [f"{safe_get(t, 'title','(no title)')} | {safe_get(t, 'assigned_to','unassigned')}" for t in task_docs]
    sel_idx = st.selectbox("Select Task to Judge", list(range(len(task_titles))), format_func=lambda i: task_titles[i], key="judge_task_sel")
    sel_task = task_docs[sel_idx]

    if st.button("📌 Judge Selected Task"):
        prompt = f"""
You are an academic mentor evaluating ONE task.

Task:
{sel_task}

Evaluate:

1. Is this task well-defined and clear?
2. Is the estimated duration reasonable?
3. Is the assignment (member) appropriate?
4. Risks or issues for this task.
5. Suggestions to improve the task definition or execution.

Be neutral and concise.
"""
        st.subheader("Task Evaluation")
        st.write(ai_chat(prompt))

st.divider()
# ===============================================================
# 4️⃣ REVIEW STUDENT / LEAD SUBMISSIONS (AI + MANUAL + EXTRA MODES)
# ===============================================================
st.header("📂 Student & Lead Submissions")


def is_valid_submission(r: dict) -> bool:
    """
    A valid submission MUST:
    - be linked to a task
    - have a title
    - contain notes or a link
    """
    if not safe_get(r, "task_id"):
        return False

    title = safe_get(r, "title", "").strip()
    notes = safe_get(r, "notes", "").strip()
    link = safe_get(r, "link", "").strip()

    if not title:
        return False

    if not notes and not link:
        return False

    return True


# Filter junk / invalid submissions
valid_subs = [r for r in subs if is_valid_submission(r)]

if not valid_subs:
    st.info("No valid student or lead submissions available for review yet.")
else:
    for r in valid_subs:
        rid = safe_get(r, "resource_id", str(uuid.uuid4()))
        title = safe_get(r, "title", "")
        submitter = safe_get(r, "user_email", "unknown")
        notes = safe_get(r, "notes", "")
        link = safe_get(r, "link", "")

        st.subheader(f"{title}  — by {submitter}")

        if notes:
            st.markdown("**Notes:**")
            st.write(notes)

        if link:
            st.write(f"🔗 [Open Submitted Link]({link})")

        # ---------------- Previous AI results ----------------
        prior_review = safe_get(r, "mentor_review")
        prior_acc = safe_get(r, "ai_accuracy_report")
        prior_plag = safe_get(r, "plagiarism_report")

        if prior_review:
            with st.expander("📄 Previous AI Mentor Review"):
                st.write(prior_review)

        if prior_acc:
            with st.expander("🎯 Previous AI Accuracy Check"):
                st.write(prior_acc)

        if prior_plag:
            with st.expander("🧬 Previous Plagiarism / Originality Check"):
                st.write(prior_plag)

        # ---------------- Expected task ----------------
        expected_task = None
        if safe_get(r, "task_id"):
            expected_task = tasks_col.find_one({"task_id": r["task_id"]})

        col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)

        # --- AI Mentor Review ---
        if col_ai1.button("🤖 Mentor Review", key=f"ai_review_{rid}"):
            review = ai_review_text(title, notes, link)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"mentor_review": review}},
                upsert=True,
            )
            st.success("AI mentor review generated.")
            st.write(review)

        # --- AI Accuracy Check ---
        if col_ai2.button("🎯 Check Accuracy", key=f"ai_acc_{rid}"):
            acc_report = ai_check_accuracy(title, notes, link, expected_task)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"ai_accuracy_report": acc_report}},
                upsert=True,
            )
            st.success("Accuracy evaluation completed.")
            st.write(acc_report)

        # --- Plagiarism Check ---
        if col_ai3.button("🧬 Plagiarism Check", key=f"plag_{rid}"):
            plag = ai_plagiarism_check(title, notes, link)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"plagiarism_report": plag}},
                upsert=True,
            )
            st.success("Plagiarism analysis completed.")
            st.write(plag)

        # --- Coaching Feedback ---
        if col_ai4.button("💬 Coaching Feedback", key=f"coach_{rid}"):
            soft = ai_soft_coaching_feedback(title, notes)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"coaching_feedback": soft}},
                upsert=True,
            )
            st.success("Coaching feedback generated.")
            st.write(soft)

        # ---------------- Manual Feedback ----------------
        st.markdown("#### ✍ Manual Mentor Feedback")

        fb_key = f"fb_text_{rid}"
        fb_text = st.text_area("Feedback:", key=fb_key)

        if st.button("Submit Feedback", key=f"fb_btn_{rid}"):
            progress = ensure_progress_record(project_id, submitter)

            progress_col.update_one(
                {"progress_id": progress["progress_id"]},
                {
                    "$push": {
                        "comments": {
                            "by": safe_get(user, "email"),
                            "text": fb_text,
                            "at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                            "resource_id": rid,
                        }
                    }
                },
            )

            resources_col.update_one(
                {"resource_id": rid},
                {
                    "$push": {
                        "feedbacks": {
                            "by": safe_get(user, "email"),
                            "text": fb_text,
                            "at": datetime.datetime.utcnow(),
                        }
                    }
                },
                upsert=True,
            )

            st.success("Mentor feedback saved successfully.")
            st.rerun()

        st.markdown("---")

st.divider()
st.success("Mentor Dashboard Loaded Successfully (Clean Academic Review Mode).")
