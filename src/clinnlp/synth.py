"""Template-based SYNTHETIC clinical notes with gold entity assertions and gold PHI spans.

Everything produced here is synthetic: names, dates, identifiers and narratives are random draws; disease
mentions are real surface strings taken from the NCBI Disease dev/test splits (so the NER, trained on the train
split, has not seen those documents). Gold labels are exact by construction. Templates are split into
"train" and "test" families so that assertion/de-id models can be evaluated on phrasings they never saw.
"""
import random
from dataclasses import dataclass, field

from .corpus import load_ncbi

FIRST = """James Mary Robert Patricia John Jennifer Michael Linda William Elizabeth David Barbara Richard Susan Joseph Jessica
Thomas Sarah Charles Karen Daniel Nancy Matthew Lisa Anthony Betty Mark Margaret Donald Sandra Steven Ashley Paul Kimberly Andrew Emily
Joshua Donna Kenneth Michelle Kevin Carol Brian Amanda George Melissa Edward Deborah Ronald Stephanie Timothy Rebecca Jason Laura
Jeffrey Sharon Ryan Cynthia Jacob Kathleen Gary Amy Nicholas Angela Eric Shirley Jonathan Anna Stephen Brenda Larry Pamela Justin Emma
Scott Nicole Brandon Helen Benjamin Samantha Samuel Katherine Gregory Christine Frank Debra Raymond Rachel Alexander Carolyn Patrick Janet
Jack Maria Dennis Heather Jerry Diane Tyler Olivia Aaron Julie Jose Joyce Henry Victoria Adam Ruth Douglas Priya Omar Wei Fatima Ahmed
Yuki Mateo Aisha Ivan Chen Sofia Rahul Lucia Hassan Mei Dmitri Amara""".split()
LAST = """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor
Moore Jackson Martin Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young Allen King Wright Scott Torres
Nguyen Hill Flores Green Adams Nelson Baker Hall Rivera Campbell Mitchell Carter Roberts Gomez Phillips Evans Turner Diaz Parker
Cruz Edwards Collins Reyes Stewart Morris Morales Murphy Cook Rogers Gutierrez Ortiz Morgan Cooper Peterson Bailey Reed Kelly Howard
Ramos Kim Cox Ward Richardson Watson Brooks Chavez Wood James Bennett Gray Mendoza Ruiz Hughes Price Alvarez Castillo Sanders Patel
Myers Long Ross Foster Jimenez Powell Jenkins Perry Russell Sullivan Bell Coleman Butler Henderson Barnes Gonzales Fisher Vasquez
Simmons Romero Jordan Patterson Alexander Hamilton Graham Reynolds Griffin Wallace Moreno West Cole Hayes Bryant Herrera Gibson Ellis
Tran Medina Aguilar Stevens Murray Ford Castro Marshall Owens Harrison Fernandez McDonald Woods Washington Kennedy Wells Vargas
Henry Chen Freeman Webb Tucker Guzman Burns Crawford Olson Simpson Porter Hunter Gordon Mendez Silva Shaw Snyder Mason Dixon
Hunt Hicks Holmes Palmer Wagner Black Robertson Boyd Rose Stone Salazar Fox Warren Mills Meyer Rice Schmidt Garza Daniels Ferguson
Nichols Stephens Soto Weaver Ryan Gardner Payne Grant Dunn Kelley Spencer Hawkins Parkinson Huntington Crohn Graves Hodgkin Addison
Cushing Wilms Down Turner Marfan Raynaud Bright Paget Tourette Hashimoto Behcet Sjogren Gilbert Alport Barrett Bell Duchenne Becker""".split()
CITY = ["Springfield", "Riverton", "Fairview", "Madison", "Georgetown", "Clinton", "Salem", "Franklin", "Greenville", "Bristol", "Ashland",
        "Oxford", "Kingston", "Dayton", "Lexington", "Milton", "Newport", "Auburn", "Burlington", "Chester", "Lakewood", "Hudson", "Marion", "Dover"]
HOSP_FMT = ["{} General Hospital", "St. {} Medical Center", "{} Regional Hospital", "{} Memorial Hospital", "University of {} Health", "{} Community Clinic"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# (assertion, temporality, experiencer, section, text, family)   {0}/{1} disease slots, {P:TYPE} PHI slots
T = []
def add(a, t, e, sec, family, *texts):
    for x in texts:
        T.append((a, t, e, sec, x, family))

# --- present
add("present", "current", "patient", "HPI", "train", "Patient presents with {0}.", "Diagnosed with {0} on admission.", "{0} is noted on examination.",
    "Ongoing management of {0}.", "Continues to have {0}.", "Worsening {0} over the past week.", "Assessment: active {0}.", "He was found to have {0}.")
add("present", "current", "patient", "HPI", "test", "Admitted for acute {0}.", "Imaging confirms {0}.", "Reports persistent {0}.", "She is being treated for {0}.",
    "Evidence of {0} on today's labs.", "Known {0}, stable on current therapy.", "No change in {0}.", "{0} continues to progress.")
# --- absent
add("absent", "current", "patient", "ROS", "train", "Denies {0}.", "No evidence of {0}.", "Negative for {0}.", "Patient has no {0}.", "Without {0}.",
    "{0} was ruled out.", "No {0} noted.", "She denies any {0}.")
add("absent", "current", "patient", "ROS", "test", "There is no sign of {0} or {1}.", "Free of {0}.", "{0} is absent.", "Workup excluded {0}.",
    "No history of {0}.", "Neither {0} nor {1} was identified.", "Patient does not have {0}.", "Not consistent with {0}.", "No suspicion of {0}.")
# --- possible
add("possible", "current", "patient", "A/P", "train", "Possible {0}.", "Concern for {0}.", "Suspected {0}.", "Cannot rule out {0}.", "Likely {0}.",
    "Differential includes {0}.", "Probable {0}.")
add("possible", "current", "patient", "A/P", "test", "Findings suggestive of {0}.", "{0} is probable.", "Query {0}.", "Rule out {0}.", "Presentation consistent with {0}.",
    "May have {0}.", "Evaluate for {0}.", "{0} cannot be excluded.", "{0} versus {1}.")
# --- hypothetical
add("hypothetical", "current", "patient", "Plan", "train", "Return if {0} develops.", "Monitor for signs of {0}.", "Risk of {0} discussed.",
    "Counseled on prevention of {0}.", "Call if {0} occurs.")
add("hypothetical", "current", "patient", "Plan", "test", "Watch for {0}.", "Should {0} develop, seek care.", "Screening for {0} was discussed.",
    "At risk for {0}.", "Education provided regarding {0} precautions.")
# --- historical
add("present", "historical", "patient", "PMH", "train", "History of {0}.", "PMH: {0}.", "Previously treated for {0}.", "{0} resolved in {P:YEAR}.",
    "S/p treatment for {0}.", "Past history of {0}.")
add("present", "historical", "patient", "PMH", "test", "Remote {0} as a child.", "Had {0} years ago.", "Prior diagnosis of {0}.", "Status post {0}.",
    "Hx of {0} and {1}.", "{0} in remission since {P:YEAR}.")
# --- family
add("present", "current", "family", "FH", "train", "Mother had {0}.", "Family history significant for {0}.", "Father died of {0}.", "Sister diagnosed with {0}.",
    "Family history of {0}.")
add("present", "current", "family", "FH", "test", "Strong history of {0} in maternal grandmother.", "Brother has {0}.", "Her uncle was treated for {0}.",
    "Paternal aunt with {0}.", "Two siblings have {0}.")
add("absent", "current", "family", "FH", "train", "No family history of {0}.", "Mother denies {0}.", "Father without {0}.")
add("absent", "current", "family", "FH", "test", "No history of {0} in the family.", "Siblings are free of {0}.")
# --- two-entity scope tests (labels per slot)
TWO = [  # (text, [labels0, labels1], family)
    ("Denies {0} but reports {1}.", [("absent", "current", "patient"), ("present", "current", "patient")], "train"),
    ("No {0}; presents with {1}.", [("absent", "current", "patient"), ("present", "current", "patient")], "train"),
    ("Negative for {0}, however {1} persists.", [("absent", "current", "patient"), ("present", "current", "patient")], "train"),
    ("Family history of {0}; patient has {1}.", [("present", "current", "family"), ("present", "current", "patient")], "train"),
    ("Possible {0} and likely {1}.", [("possible", "current", "patient"), ("possible", "current", "patient")], "train"),
    ("She has {0} although {1} was ruled out.", [("present", "current", "patient"), ("absent", "current", "patient")], "test"),
    ("Without {0}, but with {1}.", [("absent", "current", "patient"), ("present", "current", "patient")], "test"),
    ("History of {0} with current {1}.", [("present", "historical", "patient"), ("present", "current", "patient")], "test"),
    ("Mother had {0} while the patient has {1}.", [("present", "current", "family"), ("present", "current", "patient")], "test"),
    ("Denies {0}, {1} or any other symptoms.", [("absent", "current", "patient"), ("absent", "current", "patient")], "test"),
]
FILLER = ["BP 128/76, HR 82, Temp 98.6 F, SpO2 97% on room air.", "Creatinine 1.1 mg/dL, WBC 7.4, hemoglobin 13.2 g/dL.",
          "Started lisinopril 10 mg daily and metformin 500 mg twice daily.", "Chest x-ray reviewed with no acute changes noted by radiology.",
          "Patient is alert and oriented, in no acute distress.", "Lungs clear to auscultation bilaterally.", "Follow up in 2 weeks with primary care.",
          "Labs drawn at 06:30, results pending.", "He is a retired teacher and lives with his wife.", "Weight 82 kg, height 175 cm, BMI 26.8.",
          "The patient was last seen in {P:YEAR}.", "Discharge medications reconciled with the pharmacy.", "Telemetry showed normal sinus rhythm at 72 bpm.",
          "Recommended 150 minutes of exercise per week.", "Pain 3/10, improved after acetaminophen 650 mg."]
PHI_SENT = ["Seen by Dr. {P:NAME} on {P:DATE}.", "Spoke with the patient's daughter {P:NAME} at {P:PHONE}.", "Referred to Dr. {P:NAME} at {P:LOC}.",
            "Lives in {P:CITY}.", "Discharged on {P:DATE} to home.", "Contact {P:NAME} ({P:PHONE}) with questions.", "Follow-up scheduled with {P:LOC} on {P:DATE}.",
            "Please email results to {P:EMAIL}.", "Pharmacy callback number {P:PHONE}.", "Transferred from {P:LOC}.", "Records faxed to Dr. {P:NAME}, ID {P:ID}.", "{P:AGE} patient with multiple comorbidities."]
# held-out phrasings used only when template_family == "test" (de-identification generalisation)
PHI_SENT_TEST = ["Dr. {P:NAME}'s office called back at {P:PHONE}.", "Per {P:NAME}, the patient recently moved to {P:CITY}.", "Mr. {P:NAME} was admitted {P:DATE}.",
                 "Sent to {P:LOC} for further care.", "The patient's spouse {P:NAME} is the emergency contact.", "Report cc'd to {P:EMAIL}.",
                 "Case reviewed by {P:NAME}, MD on {P:DATE}.", "Now {P:AGE} and living independently.", "Insurance card number {P:ID} verified."]

SECTION_ORDER = ["HPI", "ROS", "PMH", "FH", "A/P", "Plan"]
HEADINGS = {"HPI": ["History of Present Illness", "HPI"], "ROS": ["Review of Systems", "ROS"], "PMH": ["Past Medical History", "PMH"],
            "FH": ["Family History", "FH"], "A/P": ["Assessment", "Impression"], "Plan": ["Plan", "Recommendations"]}


@dataclass
class Note:
    text: str
    phi: list = field(default_factory=list)       # [(start, end, type)]
    entities: list = field(default_factory=list)  # [dict]


def _pools(seed=0):
    rng = random.Random(seed)
    f, l = sorted(set(FIRST)), sorted(set(LAST))
    rng.shuffle(f); rng.shuffle(l)
    cf, cl = int(len(f) * .7), int(len(l) * .7)
    return {"train": (f[:cf], l[:cl]), "test": (f[cf:], l[cl:])}


def disease_pool():
    out, seen = [], set()
    for sp in ("dev", "test"):
        for d in load_ncbi(sp):
            for m in d.mentions:
                if m.kind in ("SpecificDisease", "DiseaseClass") and "OMIM" not in m.concept and "|" not in m.concept and len(m.text.split()) <= 5:
                    k = (m.text.lower(), m.concept)
                    if k not in seen:
                        seen.add(k); out.append((m.text, m.concept))
    return out


class Generator:
    def __init__(self, names_split="train", template_family="train", seed=0, headings=0.7, pool=None):
        self.rng = random.Random(seed)
        self.first, self.last = _pools()[names_split]
        self.tf = template_family
        self.headings_p = headings
        self.pool = pool or disease_pool()
        self.T = [t for t in T if t[5] == template_family]
        self.TWO = [t for t in TWO if t[2] == template_family]
        self.phi_sent = PHI_SENT_TEST if template_family == "test" else PHI_SENT

    # --- PHI value makers
    def name(self):
        r = self.rng
        return f"{r.choice(self.first)} {r.choice(self.last)}" if r.random() < .75 else r.choice(self.last)

    def date(self):
        r = self.rng; y, m, d = r.randint(1990, 2025), r.randint(1, 12), r.randint(1, 28)
        return r.choice([f"{m:02d}/{d:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{MONTHS[m-1]} {d}, {y}", f"{d} {MONTHS[m-1][:3]} {y}", f"{m}/{d}/{str(y)[2:]}", f"{MONTHS[m-1]} {y}"])

    def phone(self):
        r = self.rng; a, b, c = r.randint(201, 989), r.randint(200, 999), r.randint(1000, 9999)
        return r.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}"])

    def ident(self):
        r = self.rng
        return r.choice([f"MRN {r.randint(100000, 9999999)}", f"MRN: {r.randint(100000, 9999999)}", f"{r.randint(10, 99)}-{r.randint(1000, 9999)}-{r.randint(100, 999)}", f"ID {r.randint(10**7, 10**9)}"])

    def loc(self):
        r = self.rng
        return r.choice(HOSP_FMT).format(r.choice(CITY))

    def email(self):
        r = self.rng
        return f"{r.choice(self.first).lower()}.{r.choice(self.last).lower()}@{r.choice(['mail', 'example', 'clinic'])}.{r.choice(['com', 'org', 'net'])}"

    def phi_value(self, kind):
        k = {"NAME": ("NAME", self.name), "DATE": ("DATE", self.date), "PHONE": ("PHONE", self.phone), "ID": ("ID", self.ident),
             "LOC": ("LOCATION", self.loc), "CITY": ("LOCATION", lambda: self.rng.choice(CITY)), "EMAIL": ("EMAIL", self.email),
             "AGE": ("AGE", lambda: f"{self.rng.randint(90, 104)}-year-old"), "YEAR": (None, lambda: str(self.rng.randint(1985, 2024)))}[kind]
        return k[0], k[1]()

    # --- assembly
    def _fill(self, text, slots, buf, offset):
        """Expand {0},{1},{P:X}; return text and records relative to buf offset."""
        out, i, ents, phi = "", 0, [], []
        while i < len(text):
            if text[i] == "{":
                j = text.index("}", i)
                tok = text[i + 1:j]
                if tok.isdigit():
                    surf, concept = slots[int(tok)]
                    if out == "" or out.endswith(". "):
                        surf = surf[0].upper() + surf[1:]
                    ents.append((offset + len(out), offset + len(out) + len(surf), surf, concept, int(tok)))
                    out += surf
                else:
                    kind = tok.split(":")[1]
                    ptype, val = self.phi_value(kind)
                    if ptype:
                        phi.append((offset + len(out), offset + len(out) + len(val), ptype))
                    out += val
                i = j + 1
            else:
                out += text[i]; i += 1
        return out, ents, phi

    def note(self):
        r = self.rng
        by_sec = {s: [] for s in SECTION_ORDER}
        for _ in range(r.randint(7, 14)):
            if self.TWO and r.random() < .18:
                txt, labs, _ = r.choice(self.TWO)
                sec = "ROS" if labs[0][0] == "absent" else "HPI"
                by_sec[sec].append((txt, [(*labs[0], 0), (*labs[1], 1)], 2))
            else:
                a, t, e, sec, txt, _ = r.choice(self.T)
                n = 2 if "{1}" in txt else 1
                by_sec[sec].append((txt, [(a, t, e, k) for k in range(n)], n))
        note, pos = Note(""), 0
        headed = r.random() < self.headings_p
        pid_line = r.choice(["Patient: {P:NAME}   DOB: {P:DATE}   {P:ID}", "Name: {P:NAME}; {P:ID}; Admitted {P:DATE}", "{P:NAME} - {P:ID} - phone {P:PHONE}"])
        chunks = [("", pid_line, [], False)]
        for sec in SECTION_ORDER:
            items = by_sec[sec]
            if not items:
                continue
            if sec == "HPI" and r.random() < .6:
                items.insert(0, (r.choice(self.phi_sent), [], 0))
            body = []
            for it in items:
                body.append(it)
                if r.random() < .3:
                    body.append((r.choice(FILLER), [], 0))
                if r.random() < .12:
                    body.append((r.choice(self.phi_sent), [], 0))
            chunks.append((sec, None, body, True))
        text = ""
        for sec, line, body, is_sec in chunks:
            if not is_sec:
                s, ents, phi = self._fill(line, [], text, len(text))
                note.phi += phi; text += s + "\n\n"
                continue
            if headed:
                text += r.choice(HEADINGS[sec]) + ":\n"
            sentences = []
            for tpl, labs, n in body:
                slots = [r.choice(self.pool) for _ in range(max(n, 1))]
                if n == 2 and slots[0][0].lower() == slots[1][0].lower():
                    slots[1] = r.choice(self.pool)
                s, ents, phi = self._fill(tpl, slots, text, len(text))
                for (a, b, surf, concept, k) in ents:
                    lab = labs[k]
                    note.entities.append(dict(start=a, end=b, text=surf, mesh=concept, assertion=lab[0], temporality=lab[1], experiencer=lab[2]))
                note.phi += phi
                text += s + " "
            text = text.rstrip(" ") + "\n\n"
        note.text = text.rstrip() + "\n"
        # entities/phi were computed with trailing-space-sensitive offsets; recompute positions defensively
        for e in note.entities:
            assert note.text[e["start"]:e["end"]] == e["text"], (e, note.text[e["start"]:e["end"]])
        for s, e, t in note.phi:
            assert 0 <= s < e <= len(note.text)
        note.entities.sort(key=lambda x: x["start"]); note.phi.sort()
        return note

    def notes(self, n):
        return [self.note() for _ in range(n)]
