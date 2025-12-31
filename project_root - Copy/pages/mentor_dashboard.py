import streamlit as st
import datetime
import uuid
import pandas as pd
import requests
import altair as alt
from pymongo import MongoClient
from typing import Any, Dict, List, Optional
import re

# ------------------ DB CONNECTION ------------------
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

users_col = db["users"]
teams_col = db["teams"]
tasks_col = db["tasks"]
progress_col = db["progress"]
resources_col = db["resources"]

# ------------------ HELPERS ------------------
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj[key] if isinstance(obj, dict) and key in obj else default


def ai_chat(prompt: str, model: str = "llama3.2") -> str:
    """
    Generic AI chat wrapper using local Ollama /api/chat.
    Returns plain text, max ~4000 chars.
    Neutral academic tone by default.
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


def ai_review_text(title: str, notes: str, link: str) -> str:
    """AI mentor-style qualitative review (neutral academic)."""
    prompt = f"""
Act as a neutral but strict academic mentor. Evaluate the student's submission.

Title: {title}
Notes: {notes}
Link: {link}

Provide a structured review with the following sections:

1. Short Summary (3–4 lines)
2. Score out of 10 (justify briefly)
3. Strengths (3–5 bullet points)
4. Weaknesses / Gaps (3–5 bullet points)
5. Actionable Improvements (at least 5 clear steps)

Be honest, specific, and professional.
"""
    return ai_chat(prompt)


def ai_check_accuracy(
    title: str,
    notes: str,
    link: str,
    expected_task: Optional[Dict[str, Any]] = None,
) -> str:
    """
    AI agent: judge whether the submitted work is correct/incorrect/partial,
    and give an accuracy score % + reasoning.
    Neutral academic tone.
    """
    task_context = ""
    if expected_task:
        task_context = (
            f"\nExpected Task Title: {safe_get(expected_task, 'title', '')}"
            f"\nExpected Task Description: {safe_get(expected_task, 'description', '')}"
            f"\nExpected Status: {safe_get(expected_task, 'status', '')}"
        )

    prompt = f"""
You are an expert academic evaluator.

You must judge how correctly the student completed the assigned work.

Submission:
- Title: {title}
- Notes: {notes}
- Link: {link}

{task_context}

TASK:
1. Decide if the work is:
   - CORRECT
   - PARTIALLY_CORRECT
   - INCORRECT

2. Give an ACCURACY SCORE between 0 and 100 (integer only).
3. Explain WHY in 5–8 bullet points, focusing on:
   - alignment with expected task
   - technical correctness
   - completeness
   - originality / effort
   - missing pieces

Return your answer in the following clear structure:

Verdict: <CORRECT | PARTIALLY_CORRECT | INCORRECT>
Accuracy: <integer 0-100>
Reasoning:
- <point 1>
- <point 2>
- ...
"""
    return ai_chat(prompt, model="llama3.2")


def ai_plagiarism_check(title: str, notes: str, link: str) -> str:
    """
    Lightweight AI-based plagiarism / originality style check.
    (It cannot check the actual internet, but can judge style, redundancy, and obvious copy-paste patterns.)
    """
    prompt = f"""
You are an academic integrity assistant.

You will perform a qualitative plagiarism-style check for this submission:

Title: {title}
Notes: {notes}
Link: {link}

You CANNOT access the internet or external databases.
So you must judge based on:
- overly generic / textbook-style wording
- inconsistent style across sections
- sudden changes of vocabulary level
- presence of copy-paste indicators
- missing citations

Return:

1. Estimated Plagiarism Risk: Low / Medium / High
2. Originality Score (0-100)
3. Reasons (5 bullet points)
4. Suggestions to improve originality.
"""
    return ai_chat(prompt)


def ai_soft_coaching_feedback(title: str, notes: str) -> str:
    """
    Emotional / supportive feedback mode.
    """
    prompt = f"""
You are a kind and supportive academic mentor.

Student submission:
Title: {title}
Notes: {notes}

Provide:

1. Encouraging message (2–3 lines)
2. What the student did well (3 bullets)
3. What to improve next (3 bullets)
4. One motivational closing line.

Tone: warm, encouraging, but still realistic.
"""
    return ai_chat(prompt)


def ai_judge_project(
    project_name: str,
    tasks: List[Dict[str, Any]],
    members: List[Dict[str, Any]],
) -> str:
    """
    Judge entire project quality, risks, and improvement plan.
    """
    prompt = f"""
You are a neutral academic project examiner.

Project: {project_name}

TASK SNAPSHOT (JSON-like):
{tasks}

TEAM SNAPSHOT (JSON-like):
{[{'name': safe_get(m, 'name'), 'email': safe_get(m, 'email')} for m in members]}

Provide:

1. Overall Project Status (Excellent / Good / Average / Poor / Critical)
2. Technical Depth & Complexity (short analysis)
3. Progress & Completion Quality
4. Major Risks (5 bullets)
5. Team Coordination Analysis
6. Lead Effectiveness (based only on distribution of tasks & completion)
7. Improvement Plan (10 concrete actions)
8. Final Overall Score /100.
"""
    return ai_chat(prompt)


def ai_judge_member(
    member_email: str,
    project_name: str,
    tasks: List[Dict[str, Any]],
    submissions: List[Dict[str, Any]],
    progress: Dict[str, Any],
) -> str:
    """
    Judge one member: quality, reliability, risk, and coaching.
    """
    member_tasks = [t for t in tasks if safe_get(t, "assigned_to") == member_email]
    member_subs = [r for r in submissions if safe_get(r, "user_email") == member_email]

    prompt = f"""
You are an academic mentor evaluating a single team member.

Project: {project_name}
Member: {member_email}

TASKS (JSON-like for this member):
{member_tasks}

SUBMISSIONS (JSON-like for this member):
{member_subs}

PROGRESS RECORD (if any):
{progress}

Provide:

1. Summary of this member's contribution.
2. Reliability & responsibility assessment.
3. Technical quality (3–5 bullet points).
4. Problems / red flags (if any).
5. Suggested role fit (backend / frontend / testing / docs / management etc.).
6. Improvement plan (5–7 concrete actions).
7. Final member rating out of 10.
"""
    return ai_chat(prompt)


def ai_judge_lead(
    lead_email: str,
    project_name: str,
    tasks: List[Dict[str, Any]],
    members: List[Dict[str, Any]],
) -> str:
    """
    Judge lead: distribution of tasks, coordination, and project health.
    """
    prompt = f"""
You are an academic evaluator judging the PROJECT LEAD.

Project: {project_name}
Lead Email: {lead_email}

TASKS (JSON-like):
{tasks}

MEMBERS (JSON-like):
{[{'name': safe_get(m, 'name'), 'email': safe_get(m, 'email')} for m in members]}

Focus ONLY on patterns (do not assume unknown facts).

Provide:

1. How well work is distributed across members.
2. Are there signs of overload or unfair distribution?
3. Quality of planning (from start/end dates, statuses).
4. Risks caused by leadership (if any).
5. Positive aspects of leadership.
6. Suggestions to become a stronger project lead (6 bullets).
7. Final leadership rating /10.
"""
    return ai_chat(prompt)


def ai_risk_and_timeline_report(
    project_name: str,
    tasks: List[Dict[str, Any]],
) -> str:
    """
    Predictive + risk report (no real ML, but logical analysis).
    """
    prompt = f"""
You are a project scheduling and risk expert.

Project: {project_name}

TASKS (JSON-like):
{tasks}

For the schedule, you have: start_date, end_date, status.

Provide:

1. Estimated remaining days to finish the project.
2. Delay Risk % (0–100) with explanation.
3. Critical Path tasks (list 5–10 key tasks).
4. Bottlenecks (3–5 bullets).
5. Members or areas that may cause delay.
6. Risk heat summary (High / Medium / Low with reasons).
7. Concrete timeline improvement plan (8 actions).
"""
    return ai_chat(prompt)


def ai_rank_members_from_stats(member_stats: List[Dict[str, Any]]) -> str:
    """
    Use AI to summarize ranking and badge system from computed stats.
    """
    prompt = f"""
You are given member performance stats in JSON-like format:

{member_stats}

Each entry:
- Email
- Tasks
- Completed
- CompletionPercent
- ProgressPercent
- Badge (Gold/Silver/Bronze/Red Flag)
- CombinedScore (0–100)

Do:

1. Rank members from strongest to weakest.
2. Briefly explain why top 3 are strong.
3. Briefly explain why bottom 2 (if exist) are weak / need help.
4. Suggest one improvement focus for each member.
5. Summarize overall team health in 1 paragraph.
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


def normalize_date_input(val: Any) -> datetime.date:
    if isinstance(val, datetime.date):
        return val
    if isinstance(val, (list, tuple)) and val and isinstance(val[0], datetime.date):
        return val[0]
    return datetime.date.today()


def compute_member_stats(
    project_id: str,
    members: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build stats per member:
    - tasks count
    - completed count
    - completion %
    - progress %
    - badge
    - red flag
    - combined score
    """
    # aggregate task stats
    task_stats: Dict[str, Dict[str, float]] = {}
    for t in tasks:
        email = safe_get(t, "assigned_to", "unassigned")
        if not email:
            email = "unassigned"
        if email not in task_stats:
            task_stats[email] = {"total": 0, "completed": 0}
        task_stats[email]["total"] += 1
        if safe_get(t, "status") == "Completed":
            task_stats[email]["completed"] += 1

    rows: List[Dict[str, Any]] = []
    for m in members:
        email = safe_get(m, "email", "")
        name = safe_get(m, "name", "")
        prog = ensure_progress_record(project_id, email)
        progress_pct = float(safe_get(prog, "percentage", 0) or 0)

        stat = task_stats.get(email, {"total": 0, "completed": 0})
        tot = float(stat["total"])
        comp = float(stat["completed"])
        completion_pct = round((comp / tot) * 100, 1) if tot > 0 else 0.0

        # Combined simple score: 60% weight on completion, 40% on self progress
        combined_score = round(0.6 * completion_pct + 0.4 * progress_pct, 1)

        # Badge logic
        if combined_score >= 90:
            badge = "🥇 Gold"
        elif combined_score >= 75:
            badge = "🥈 Silver"
        elif combined_score >= 50:
            badge = "🥉 Bronze"
        else:
            badge = "🚩 Red Flag"

        red_flag = combined_score < 40

        rows.append(
            {
                "Name": name,
                "Email": email,
                "Tasks": int(tot),
                "Completed": int(comp),
                "CompletionPercent": completion_pct,
                "ProgressPercent": progress_pct,
                "Badge": badge,
                "RedFlag": red_flag,
                "CombinedScore": combined_score,
            }
        )

    return rows


# ------------------ PAGE START ------------------
st.set_page_config(page_title="Mentor Dashboard", page_icon="🧑‍🏫")

user = st.session_state.get("user")
active_project = st.session_state.get("active_project")

if not user:
    st.error("Session expired. Please log in again.")
    st.stop()

if not active_project:
    st.error("No active project selected.")
    st.stop()

project_id = safe_get(active_project, "project_id")
if not project_id:
    st.error("Active project is missing project_id.")
    st.stop()

# Validate role
role = None
for p in safe_get(user, "projects", []):
    if safe_get(p, "project_id") == project_id:
        role = safe_get(p, "role")

if role != "mentor":
    st.error("You are not authorized to view this page. (Role != mentor)")
    st.stop()

team_doc = teams_col.find_one({"project_id": project_id}) or {}
project_name = safe_get(team_doc, "project_name", "Untitled Project")
lead_email = safe_get(team_doc, "lead_email", "unknown")

st.title("🧑‍🏫 Mentor Dashboard")
st.subheader(f"Project: {project_name}")
st.caption(f"Lead: {lead_email}")
st.divider()

# Preload project data for all sections
members = list(users_col.find({"projects.project_id": project_id}))
task_docs = list(tasks_col.find({"project_id": project_id}))
subs = list(resources_col.find({"project_id": project_id}).sort("created_at", -1))

# ===============================================================
# 0️⃣ AI GLOBAL CONTROLS
# ===============================================================
st.header("🤖 AI Mentor — Global Tools (Neutral Academic Mode)")

col_g1, col_g2, col_g3, col_g4 = st.columns(4)

if col_g1.button("👨‍⚖️ Judge Project"):
    report = ai_judge_project(project_name, task_docs, members)
    st.subheader("Project Judgement Report")
    st.write(report)

if col_g2.button("🚦 Risk & Timeline Report"):
    report = ai_risk_and_timeline_report(project_name, task_docs)
    st.subheader("Risk & Timeline Analysis")
    st.write(report)

if col_g3.button("👑 Judge Lead"):
    report = ai_judge_lead(lead_email, project_name, task_docs, members)
    st.subheader("Lead Evaluation")
    st.write(report)

if col_g4.button("🏅 AI Rank Members"):
    member_stats = compute_member_stats(project_id, members, task_docs)
    st.subheader("AI Ranking Explanation")
    st.write(ai_rank_members_from_stats(member_stats))

st.divider()

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

    # ---- Gantt Chart ----
    st.subheader("📊 Gantt Chart (Tasks Timeline)")

    gantt_rows = []
    for t in task_docs:
        try:
            s = datetime.date.fromisoformat(safe_get(t, "start_date", ""))
            e = datetime.date.fromisoformat(safe_get(t, "end_date", ""))
        except Exception:
            continue
        gantt_rows.append(
            {
                "Task": safe_get(t, "title", ""),
                "Assignee": safe_get(t, "assigned_to", "unassigned"),
                "Status": safe_get(t, "status", "Not Started"),
                "Start": s,
                "End": e,
            }
        )

    if gantt_rows:
        gantt_df = pd.DataFrame(gantt_rows)
        # Convert to datetime
        gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
        gantt_df["End"] = pd.to_datetime(gantt_df["End"])

        # Use pure timedelta (no .dt to avoid Pylance warnings)
        gantt_df["DurationDays"] = (gantt_df["End"] - gantt_df["Start"]).apply(
            lambda td: int(td / pd.Timedelta(days=1))
        )

        chart = (
            alt.Chart(gantt_df)
            .mark_bar()
            .encode(
                x="Start:T",
                x2="End:T",
                y=alt.Y("Task:N", sort="-x"),
                color="Assignee:N",
                tooltip=["Task", "Assignee", "Status", "Start", "End", "DurationDays"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No valid start/end dates set on tasks, so Gantt chart cannot be drawn.")
else:
    st.info("No tasks found yet for this project. Gantt and progress charts will appear once tasks exist.")

st.divider()

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

if not subs:
    st.info("No submissions yet.")
else:
    for r in subs:
        rid = safe_get(r, "resource_id", str(uuid.uuid4()))
        title = safe_get(r, "title", "(Untitled)")
        submitter = safe_get(r, "user_email", "unknown")
        notes = safe_get(r, "notes", "")
        link = safe_get(r, "link", "")

        st.subheader(f"{title}  — by {submitter}")
        st.write(f"**Notes:** {notes}")
        if link:
            st.write(f"🔗 [Open Link]({link})")

        # Show previous AI review / accuracy if present
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

        # Try to fetch related task if task_id is stored
        expected_task = None
        if "task_id" in r:
            expected_task = tasks_col.find_one({"task_id": r["task_id"]})

        col_ai1, col_ai2, col_ai3, col_ai4 = st.columns(4)

        # --- AI Mentor Review (qualitative) ---
        if col_ai1.button("🤖 Mentor Review", key=f"ai_review_{rid}"):
            review = ai_review_text(title, notes, link)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"mentor_review": review}},
                upsert=True,
            )
            st.success("AI mentor review generated and saved.")
            st.write(review)

        # --- AI Accuracy Check (correct / incorrect) ---
        if col_ai2.button("🎯 Check Accuracy", key=f"ai_acc_{rid}"):
            acc_report = ai_check_accuracy(title, notes, link, expected_task)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"ai_accuracy_report": acc_report}},
                upsert=True,
            )
            st.success("AI accuracy evaluation generated and saved.")
            st.write(acc_report)

        # --- Plagiarism / Originality Check ---
        if col_ai3.button("🧬 Plagiarism Check", key=f"plag_{rid}"):
            plag = ai_plagiarism_check(title, notes, link)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"plagiarism_report": plag}},
                upsert=True,
            )
            st.success("Plagiarism-style analysis generated and saved.")
            st.write(plag)

        # --- Soft Coaching Feedback ---
        if col_ai4.button("💬 Coaching Feedback", key=f"coach_{rid}"):
            soft = ai_soft_coaching_feedback(title, notes)
            resources_col.update_one(
                {"resource_id": rid},
                {"$set": {"coaching_feedback": soft}},
                upsert=True,
            )
            st.success("Supportive coaching feedback generated and saved.")
            st.write(soft)

        # Manual feedback
        st.markdown("#### ✍ Manual Mentor Feedback")

        fb_key = f"fb_text_{rid}"
        fb_text = st.text_area("Feedback:", key=fb_key)

        if st.button("Submit Feedback", key=f"fb_btn_{rid}"):
            progress = ensure_progress_record(project_id, submitter)

            # Add feedback to progress comments
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

            # Also store feedback inside resource
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

            st.success("Feedback submitted and stored!")
            st.rerun()

        st.markdown("---")

st.divider()

st.success("Mentor Dashboard Loaded Successfully (Neutral Academic AI Mode).")
