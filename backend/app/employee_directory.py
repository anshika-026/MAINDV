"""
employee_directory.py

Single source of truth for employee_id -> (name, company), and company ->
box color. Keyed by employee_id (never by name) because names have typos,
get re-spelled, and (007/009) have been mixed up before — the ID is the one
stable thing every other table (face_embeddings, face_training_captures,
the live overlay) already keys on.

This does NOT replace the local `employees` table in face_db.py (still used
to validate a typed-in ID against a known ID in the labeling UI) or the
external face-enrollment service's own People page. It is the corrected
values for both of those to be kept in sync with — see
face_training_routes.sync_employees(), which re-applies this directory's
name after every pull from the external service so a stale/misspelled name
there can never silently overwrite a corrected one here.

Updating an employee's name or company here never touches face_embeddings
or face_training_captures — both are keyed by employee_id, not name, so
existing embeddings/training samples stay attached to the same person.
"""

# name, company — company is "" for "no company written" in the source list.
EMPLOYEES: dict[str, dict] = {
    "001": {"name": "Kanishka Gangwar", "company": "Dewin"},
    "002": {"name": "Sushant Singh", "company": "OJI"},
    "003": {"name": "Rishash", "company": ""},
    "004": {"name": "Suraj", "company": ""},
    "005": {"name": "Salil Ahuja", "company": "Shaurrya"},
    "006": {"name": "Shilpa Chaudhary", "company": "OJI"},
    "007": {"name": "Rahul Jha", "company": "OJI"},
    "008": {"name": "Anshika Yadav", "company": "Dewin"},
    "009": {"name": "Ashwani Rana", "company": "Dewin"},
    "010": {"name": "Pankaj Gaurav", "company": "OJI"},
    "011": {"name": "Sonam", "company": "Shaurrya"},
    "012": {"name": "Aman Sharma", "company": "OJI"},
    "013": {"name": "Diksha Jyani", "company": "Dewin"},
    "014": {"name": "Divyanshu Joshi", "company": "OJI"},
    "015": {"name": "Ranjan", "company": "OJI"},
    "016": {"name": "Ajeet Yadav", "company": "Dewin"},
    "017": {"name": "Abhishek Kumar", "company": ""},
    "018": {"name": "Neha", "company": "Dewin"},
    "019": {"name": "Satyendra", "company": ""},
    "020": {"name": "Pankaj Chaudhary", "company": "Shaurrya"},
    "021": {"name": "Bidyadhar Nayak", "company": "Shaurrya"},
    "022": {"name": "Ankit", "company": ""},
    "023": {"name": "Neeraj", "company": ""},
    "024": {"name": "Anish", "company": ""},
    "025": {"name": "Tarushi", "company": "Shaurrya"},
    "026": {"name": "Mahesh Chaudhary", "company": ""},
    "027": {"name": "Aarti", "company": "OJI"},
    "028": {"name": "Jitendra SCM", "company": "OJI"},  # existing name kept, per correction list
    "029": {"name": "Rajesh Shukla", "company": "Shaurrya"},
    "030": {"name": "Vineet", "company": "OJI"},
    "031": {"name": "Sachin", "company": ""},
    "032": {"name": "Sanjeev", "company": "Shaurrya"},
    "033": {"name": "Digesh Sharma", "company": ""},  # existing name kept, per correction list
    "034": {"name": "Hemant", "company": ""},
    "035": {"name": "Deepanshu Kaushik", "company": ""},  # existing name kept, per correction list
    "036": {"name": "Alauddin", "company": ""},
    "037": {"name": "Shams", "company": "Shaurrya"},
    "038": {"name": "Nandini", "company": "Dewin"},
    "039": {"name": "Pandit Ji", "company": ""},
    "040": {"name": "Ganesh Sir", "company": ""},
    "041": {"name": "Sanket", "company": ""},
    "042": {"name": "Vivek", "company": ""},
    "043": {"name": "Rohit Rathore", "company": ""},
}

# "" (no company written) maps to the same box color as OJI, per spec.
COMPANY_COLORS: dict[str, str] = {
    "Dewin": "#2563eb",     # blue
    "OJI": "#f97316",       # orange
    "Shaurrya": "#000000",  # black
    "": "#f97316",          # no company -> orange
}

# An unrecognized person (no employee_id at all) shows "Person" and must
# never get a guessed company color — kept visually distinct from all three
# company colors above so it's never mistaken for one.
NEUTRAL_BOX_COLOR = "#6b7280"


def get_employee(employee_id: str | None) -> dict | None:
    if not employee_id:
        return None
    return EMPLOYEES.get(employee_id)


def get_display(employee_id: str | None) -> tuple[str | None, str]:
    """Returns (name_or_None, box_color). name is None for an unrecognized
    person (caller shows "Person") or an employee_id not in this directory
    (caller falls back to showing the raw ID) — in both cases the color is
    the neutral, non-company color, never a guess."""
    emp = get_employee(employee_id)
    if emp is None:
        return None, NEUTRAL_BOX_COLOR
    return emp["name"], COMPANY_COLORS.get(emp["company"], NEUTRAL_BOX_COLOR)
