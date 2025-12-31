"""
🧠 Agile Task Planner — Scrum Epics & Stories (llama3 via Ollama, Skill-Aware)
====================================================================

Hierarchy:
  EPIC
    → USER STORIES (with Acceptance Criteria, Definition of Done, Story Points)
        → TASKS (est. working days, assignee via skills)
            → SUBTASKS (optional, est. working days)

Data flow:
- Reads FINAL objectives from MongoDB 'project_objectives' (field: objectives, per project_id).
- (Optional legacy fallback: 'teams' collection, field: objectives.)
- Reads team members & their skills from 'users' + 'progress'.
- Uses Ollama (llama3:latest) to generate a Scrum plan based on:
    - project name
    - domain
    - selected final objectives
- Asks for project START + END date and sprint length.
- Converts plan into a WBS-like flat table: 1.EPIC.STORY.TASK.SUBTASK.
- Distributes work using SKILL MATCHING.
- Approximates weekend skipping in timeline.
- Persists leaf tasks into 'tasks' collection.

Assumes:
- st.session_state["user"] has 'email'.
- st.session_state["active_project"] has 'project_id', 'project_name', 'domain', 'lead_email' (optional).
"""

from __future__ import annotations

import os
import re
import json
import uuid
import datetime
from typing import Any, Dict, List

import requests
import streamlit as st
import pandas as pd
from pymongo import MongoClient


# ---------------- CONFIG ----------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Local llama3 model via Ollama (FREE / UNLIMITED locally)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "llama3:latest"
OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434/api/chat"
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

teams_col = db["teams"]
users_col = db["users"]
progress_col = db["progress"]
tasks_col = db["tasks"]
objectives_col = db["project_objectives"]


# ---------------- BASIC HELPERS ----------------

def safe_get(d: Any, k: str, default: Any = None) -> Any:
    return d.get(k, default) if isinstance(d, dict) else default


def normalize_date(d: Any) -> datetime.date:
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, (list, tuple)) and d and isinstance(d[0], datetime.date):
        return d[0]
    return datetime.date.today()


def is_weekend(day: datetime.date) -> bool:
    # Monday=0 ... Sunday=6
    return day.weekday() >= 5


def add_working_days(start: datetime.date, days: int) -> datetime.date:
    """
    Add N working days to a date (Mon–Fri).
    Simple approximation for skipping weekends.
    """
    if days <= 0:
        return start
    current = start
    remaining = days
    while remaining > 0:
        current += datetime.timedelta(days=1)
        if not is_weekend(current):
            remaining -= 1
    return current


# ---------------- OLLAMA HEALTH + CALL ----------------

def ollama_available() -> bool:
    """
    Ping Ollama /api/tags to see if server is alive.
    """
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_llm(prompt: str, max_tokens: int = 3000) -> str:
    """
    Call llama3:latest via Ollama /api/chat.
    Returns assistant content string.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.15,
        "stream": False,
    }
    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
    except Exception:
        return ""

    try:
        j = r.json()
    except Exception:
        return r.text or ""

    text = ""
    if isinstance(j, dict):
        # Standard Ollama /api/chat response
        text = j.get("message", {}).get("content", "") or ""
    else:
        text = str(j)

    return text.strip()


def parse_llm_json(raw: str) -> Any:
    """
    Try to extract a JSON list/dict from model output.
    (llama3 usually returns clean JSON if prompted properly.)
    """
    if not raw:
        return None

    raw = raw.strip()

    # If it's already a full JSON list/dict
    if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
        try:
            return json.loads(raw)
        except Exception:
            return None

    # Try to locate the first JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try to locate the first JSON object
    m2 = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    return None


# ---------------- SKILL MATCHING ----------------

def score_member_for_task(member_skills: List[str], task_title: str) -> float:
    if not member_skills:
        return 0.0
    words = re.findall(r"[A-Za-z]{4,}", (task_title or "").lower())
    if not words:
        return 0.0
    score = 0
    for w in words:
        for s in member_skills:
            sk = (s or "").lower()
            if not sk:
                continue
            if w in sk or sk in w:
                score += 1
    return score / max(1, len(words))


def pick_best_assignee(task_title: str, members: List[Dict[str, Any]], lead_email: str) -> str:
    if not members:
        return lead_email or "unassigned"
    best = lead_email
    best_score = -1.0
    for m in members:
        email = safe_get(m, "email", lead_email)
        skills = safe_get(m, "skills", [])
        sc = score_member_for_task(skills, task_title)
        if sc > best_score:
            best_score = sc
            best = email
    return best or lead_email or "unassigned"


# ---------------- SCRUM PLAN GENERATION ----------------

def generate_scrum_plan(objectives: List[str], project_name: str, domain: str) -> List[Dict[str, Any]]:
    """
    Returns a structured Scrum backlog as a list of epics.
    """
    if not objectives:
        return []

    joined_obj = "\n".join(f"- {o}" for o in objectives)

    prompt = f"""
You are a senior Scrum Master and academic project mentor.

Project Name: {project_name}
Domain: {domain}
Context: This is an academic student project, not a commercial product.
The project should follow good Agile/Scrum practices while staying realistic for students.

FINAL SELECTED OBJECTIVES:
{joined_obj}

TASK:
Convert these objectives into a full SCRUM backlog in JSON with the following structure:

[
  {{
    "epic": "string",
    "description": "string",
    "stories": [
      {{
        "id": "S1",
        "title": "As a <student/mentor/system/etc>, I want <capability> so that <educational or project value>.",
        "acceptance_criteria": [
          "Given ..., When ..., Then ...",
          "Given ..., When ..., Then ..."
        ],
        "definition_of_done": [
          "All acceptance criteria are met",
          "Unit tests written and passing",
          "Code reviewed and documented"
        ],
        "story_points": 3,
        "tasks": [
          {{
            "title": "Implement feature X",
            "description": "short technical description",
            "estimated_days": 2,
            "subtasks": [
              {{
                "title": "Design data model",
                "estimated_days": 1
              }},
              {{
                "title": "Implement API",
                "estimated_days": 1
              }}
            ]
          }}
        ]
      }}
    ]
  }}
]

Rules:
- 4 to 8 EPICS for the whole project.
- Each EPIC: 2–6 USER STORIES.
- Each story: 3–8 acceptance_criteria, 3–8 definition_of_done items.
- Each story: 2–6 tasks.
- Some tasks may have 0–3 subtasks.
- estimated_days must be small integers (1–5) representing WORKING DAYS for a student.
- Everything MUST be specific to the project objectives and domain.
- Return ONLY valid JSON. No explanations, no commentary.
"""

    raw = call_llm(prompt)
    data = parse_llm_json(raw)
    if not isinstance(data, list):
        return []
    return data


def fallback_scrum_plan(objectives: List[str]) -> List[Dict[str, Any]]:
    """
    Simple deterministic plan if llama3 is not available or JSON fails.
    """
    if not objectives:
        return []
    epics: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objectives[:4], start=1):
        epics.append(
            {
                "epic": f"Epic {idx}: {obj[:50]}",
                "description": obj,
                "stories": [
                    {
                        "id": f"S{idx}1",
                        "title": f"As a student, I want to achieve: {obj[:60]}",
                        "acceptance_criteria": [
                            "Given the project objectives, when core features are implemented, then the system should satisfy basic functional requirements."
                        ],
                        "definition_of_done": [
                            "Core functionality implemented",
                            "Basic tests added",
                            "Code pushed to repository",
                        ],
                        "story_points": 3,
                        "tasks": [
                            {
                                "title": "Plan implementation steps",
                                "description": "Break down objective into small tasks.",
                                "estimated_days": 2,
                                "subtasks": [],
                            },
                            {
                                "title": "Implement prototype",
                                "description": "Code & test basic version.",
                                "estimated_days": 3,
                                "subtasks": [],
                            },
                        ],
                    }
                ],
            }
        )
    return epics


# ---------------- FLATTEN SCRUM PLAN TO TASK ROWS ----------------

def flatten_scrum_plan(
    plan: List[Dict[str, Any]],
    members: List[Dict[str, Any]],
    lead_email: str,
    start_date: datetime.date,
    end_date: datetime.date,
    sprint_len_days: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convert nested epics→stories→tasks→subtasks into flat rows with:
    - WBS code
    - sprint number
    - approximate working-day schedule with weekend skip
    - best assignee based on skills
    """
    if not plan:
        return []

    rows: List[Dict[str, Any]] = []
    current = start_date
    sprint_num = 1

    for e_idx, epic in enumerate(plan, start=1):
        epic_code = f"{e_idx}"
        epic_name = epic.get("epic", f"Epic {e_idx}")
        epic_desc = epic.get("description", "")
        stories = epic.get("stories") or []

        for s_idx, story in enumerate(stories, start=1):
            story_code = f"{epic_code}.{s_idx}"
            story_title = story.get("title", "")
            ac_list = story.get("acceptance_criteria", []) or []
            dod_list = story.get("definition_of_done", []) or []
            story_points = int(story.get("story_points", 3))

            tasks = story.get("tasks") or []
            for t_idx, task in enumerate(tasks, start=1):
                task_code = f"{story_code}.{t_idx}"
                t_title = task.get("title", "")
                t_desc = task.get("description", "")

                est_days = max(1, int(task.get("estimated_days", 1)))
                # schedule in working days, skipping weekends
                start_t = current
                end_t = add_working_days(start_t, est_days)
                assignee = pick_best_assignee(t_title, members, lead_email)

                rows.append(
                    {
                        "wbs": task_code,
                        "epic": epic_name,
                        "epic_description": epic_desc,
                        "story_title": story_title,
                        "story_points": story_points,
                        "acceptance_criteria": ac_list,
                        "definition_of_done": dod_list,
                        "title": t_title,
                        "description": t_desc,
                        "estimated_days": est_days,
                        "assignee": assignee,
                        "start_date": start_t,
                        "end_date": end_t,
                        "sprint": sprint_num,
                        "parent_wbs": story_code,
                        "is_subtask": False,
                    }
                )

                current = end_t
                # advance sprint if we've passed a sprint window
                days_from_start = (current - start_date).days
                if days_from_start // sprint_len_days + 1 > sprint_num:
                    sprint_num += 1

                # ----- Subtasks -----
                subtasks = task.get("subtasks") or []
                for u_idx, sub in enumerate(subtasks, start=1):
                    sub_code = f"{task_code}.{u_idx}"
                    sub_title = sub.get("title", "")
                    sub_days = max(1, int(sub.get("estimated_days", 1)))

                    sub_start = current
                    sub_end = add_working_days(sub_start, sub_days)
                    sub_assignee = pick_best_assignee(sub_title, members, lead_email)

                    rows.append(
                        {
                            "wbs": sub_code,
                            "epic": epic_name,
                            "epic_description": epic_desc,
                            "story_title": story_title,
                            "story_points": story_points,
                            "acceptance_criteria": ac_list,
                            "definition_of_done": dod_list,
                            "title": sub_title,
                            "description": f"Subtask of: {t_title}",
                            "estimated_days": sub_days,
                            "assignee": sub_assignee,
                            "start_date": sub_start,
                            "end_date": sub_end,
                            "sprint": sprint_num,
                            "parent_wbs": task_code,
                            "is_subtask": True,
                        }
                    )

                    current = sub_end
                    days_from_start = (current - start_date).days
                    if days_from_start // sprint_len_days + 1 > sprint_num:
                        sprint_num += 1

                # clamp to project end date
                if current > end_date:
                    current = end_date

    return rows


# ---------------- MAIN STREAMLIT PAGE ----------------

def task_planner_page(user: Dict[str, Any], active_project: Dict[str, Any]):
    """Main task planner page function."""
    st.subheader("🧠 Agile Task Planner — Scrum Epics, Stories, Tasks, Subtasks")

    # --- Session context ---
    if not user:
        st.error("Please login first.")
        st.stop()

    if not active_project:
        st.error("No active project found in session_state['active_project'].")
        st.stop()

    project_id = safe_get(active_project, "project_id", "")
    project_name = safe_get(active_project, "project_name", "Untitled Project")
    domain = safe_get(active_project, "domain", "Unknown Domain")
    lead_email = safe_get(active_project, "lead_email", safe_get(user, "email", ""))

    if not project_id:
        st.error("Active project has no project_id.")
        st.stop()

    st.info(f"📌 Project: **{project_name}**   |   🧪 Domain: **{domain}**")
    st.markdown("---")

    # =====================================================
    # 🔷 OBJECTIVES
    # =====================================================
    pid = str(project_id)
    pn = project_name
    lead = lead_email

    obj_doc = objectives_col.find_one({"project_id": pid}) or {}
    objectives: List[str] = obj_doc.get("objectives") or []

    # optional legacy fallback: old storage in teams.objectives
    if not objectives:
        team_doc = (
            teams_col.find_one({"project_id": pid})
            or teams_col.find_one({"project_name": pn})
            or (teams_col.find_one({"lead_email": lead}) if lead else {})
            or {}
        )
        objectives = team_doc.get("objectives") or []

    if not objectives:
        st.error(
            "No final objectives found for this project.\n\n"
            "Please open the **Objectives Generator** page, generate objectives, "
            "select them, and save. Then come back here."
        )
        st.stop()

    st.success(f"{len(objectives)} final objectives loaded.")
    with st.expander("Show Final Objectives"):
        for i, obj in enumerate(objectives, start=1):
            st.markdown(f"**{i}.** {obj}")

    st.markdown("---")

    # --- Load members & skills ---
    members_docs = list(users_col.find({"projects.project_id": project_id}))
    members_info: List[Dict[str, Any]] = []

    for m in members_docs:
        email = safe_get(m, "email", "")
        prog = progress_col.find_one({"project_id": project_id, "user_email": email}) or {}
        skills = prog.get("skills", []) or []
        members_info.append({"email": email, "skills": skills})

    st.markdown("### 👥 Team Members & Skills")
    if members_info:
        st.dataframe(
            pd.DataFrame(
                [{"email": mi["email"], "skills": ", ".join(mi.get("skills", []))} for mi in members_info]
            ),
            use_container_width=True,
        )
    else:
        st.info("No team members found for this project. All work will be assigned to the project lead.")

    st.markdown("---")

    # --- Timeline + settings ---
    st.markdown("### 📅 Project Timeline & Sprint Settings")

    start_raw = st.date_input(
        "Project start date",
        value=datetime.date.today(),
        key="scrum_start_date",
    )
    end_raw = st.date_input(
        "Project end date",
        value=datetime.date.today() + datetime.timedelta(days=60),
        key="scrum_end_date",
    )

    start_date = normalize_date(start_raw)
    end_date = normalize_date(end_raw)

    if start_date >= end_date:
        st.error("End date must be strictly after start date.")
        st.stop()

    sprint_len = st.number_input(
        "Sprint length (calendar days, approx)",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        key="scrum_sprint_length",
    )

    st.caption(
        f"Total project duration: {(end_date - start_date).days} calendar days "
        "(weekend skipping is approximated)."
    )

    st.markdown("---")

    use_llm = st.checkbox("Use llama3 via Ollama for full Scrum plan", value=True)

    if st.button("🚀 Generate Scrum Backlog from Final Objectives"):
        with st.spinner("Generating Scrum Epics, Stories, Tasks, Subtasks..."):
            if use_llm and ollama_available():
                plan = generate_scrum_plan(objectives, project_name, domain)
                if not plan:
                    st.warning("llama3 returned no/invalid JSON. Using fallback Scrum plan.")
                    plan = fallback_scrum_plan(objectives)
            else:
                if use_llm:
                    st.warning("Ollama/llama3 not reachable. Using fallback Scrum plan.")
                plan = fallback_scrum_plan(objectives)

            flat_rows = flatten_scrum_plan(
                plan,
                members_info,
                lead_email,
                start_date,
                end_date,
                sprint_len_days=int(sprint_len),
            )
            st.session_state["scrum_tasks"] = flat_rows

        st.success("Scrum WBS generated!")
        st.rerun()

    tasks: List[Dict[str, Any]] = st.session_state.get("scrum_tasks", [])
    if not tasks:
        st.info("Click '🚀 Generate Scrum Backlog from Final Objectives' to create the plan.")
        return

    st.markdown("### ✏ Review & Edit Tasks (Epics / Stories / Tasks / Subtasks)")

    for idx, t in enumerate(tasks):
        c = st.columns([0.09, 0.3, 0.2, 0.12, 0.12, 0.07, 0.1])
        c[0].markdown(f"**{t['wbs']}**")
        c[1].markdown(f"**Epic:** {t['epic']}")
        c[2].markdown(f"**Story:** {t['story_title']}")
        t["title"] = c[3].text_input(f"title_{idx}", t["title"])
        t["estimated_days"] = c[4].number_input(
            f"days_{idx}",
            min_value=1,
            value=int(t["estimated_days"]),
        )

        assignee_options = [m["email"] for m in members_info]
        if lead_email and lead_email not in assignee_options:
            assignee_options.append(lead_email)
        if not assignee_options:
            assignee_options = ["unassigned"]

        try:
            default_index = assignee_options.index(t["assignee"])
        except Exception:
            default_index = 0

        t["assignee"] = c[5].selectbox(
            f"assign_{idx}",
            assignee_options,
            index=default_index,
        )

        c[6].markdown(f"{t['start_date'].isoformat()} → {t['end_date'].isoformat()}")

    st.markdown("---")

    st.markdown("### 🗂 Scrum WBS Table Preview")

    df = pd.DataFrame(
        [
            {
                "WBS": t["wbs"],
                "Epic": t["epic"],
                "Story": t["story_title"],
                "Story Points": t["story_points"],
                "Task / Subtask": t["title"],
                "Assignee": t["assignee"],
                "Sprint": t["sprint"],
                "Start": t["start_date"],
                "End": t["end_date"],
                "Est. Days": t["estimated_days"],
                "Is Subtask": t["is_subtask"],
            }
            for t in tasks
        ]
    )
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Scrum WBS CSV",
        csv_bytes,
        file_name=f"{project_name}_scrum_wbs.csv",
    )

    if st.button("💾 Save Tasks to Database"):
        inserted = 0
        for t in tasks:
            try:
                tasks_col.insert_one(
                    {
                        "task_id": str(uuid.uuid4()),
                        "project_id": project_id,
                        "wbs": t["wbs"],
                        "epic": t["epic"],
                        "epic_description": t.get("epic_description", ""),
                        "story_title": t["story_title"],
                        "story_points": int(t["story_points"]),
                        "acceptance_criteria": t.get("acceptance_criteria", []),
                        "definition_of_done": t.get("definition_of_done", []),
                        "title": t["title"],
                        "description": t.get("description", ""),
                        "assigned_to": t["assignee"],
                        "start_date": t["start_date"].isoformat(),
                        "end_date": t["end_date"].isoformat(),
                        "estimated_days": int(t["estimated_days"]),
                        "sprint": int(t["sprint"]),
                        "is_subtask": bool(t.get("is_subtask", False)),
                        "parent_wbs": t.get("parent_wbs"),
                        "status": "Not Started",
                        "created_by": lead_email,
                        "created_at": datetime.datetime.utcnow(),
                        "updated_at": datetime.datetime.utcnow(),
                    }
                )
                inserted += 1
            except Exception:
                continue

        st.success(f"Saved {inserted} tasks into MongoDB 'tasks' collection.")


# ------------------ Dynamic Runner (for direct run) ------------------

if __name__ in ("__main__", "__page__"):
    st.set_page_config(page_title="Agile Task Planner (Scrum)", layout="wide")
    user = st.session_state.get("user")
    active = st.session_state.get("active_project")
    if user and active:
        task_planner_page(user, active)
    else:
        st.error("Please open this page from the main dashboard where user and active_project are set.")
