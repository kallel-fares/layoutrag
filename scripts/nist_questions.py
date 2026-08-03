"""Hand-written questions for the sampled NIST passages.

Each entry maps a passage index in the sampled set to a question that passage answers.
Written by reading the passages rather than generated, for one reason: a model shown a
passage and asked for a question reuses that passage's vocabulary, so retrieval then
succeeds on lexical overlap and every chunking strategy scores well for a reason that has
nothing to do with chunking.

So the wording here is deliberately *not* the passage's. "Chip-off" is asked as "physically
detaching memory", "microSD" as "removable storage", "Bluetooth" as "wireless personal-area
devices". What is being tested is whether retrieval finds the right passage from a question
phrased the way somebody would actually ask it.

Passages that cannot support a question are omitted rather than forced: glossaries, acronym
lists, revision change logs, ITL boilerplate that is identical across publications, and
passages where extraction mangled the notation past the point of meaning.
"""

from __future__ import annotations

QUESTIONS: dict[int, str] = {
    0: "Where might someone find written-down access codes when examining a seized handset?",
    1: "Why are carrier records often more valuable than what is stored on the device itself?",
    3: "What weakens the strength of derived keying material beyond the derivation step itself?",
    4: "How does address-based site blocking also help against hostile destinations?",
    5: "Why does having several browsers installed matter for staff working from home?",
    6: "What must every entry point share for badge readers to work across an organisation?",
    7: "Which card authentication method is called out as the most exposed to attack?",
    8: "When did link key generation move to elliptic curve cryptography in the wireless standard?",
    9: "How do two devices arrive at a shared key when the user types the same secret into both?",
    11: "What technique adds enterprise controls to an application the organisation did not build?",
    12: (
        "How can privileged instructions be handled when the processor offers no virtualisation "
        "support?"
    ),
    13: "Where should drivers controlling physical hardware run instead of inside the hypervisor?",
    14: "What switch setting lets one guest machine observe its neighbours' network traffic?",
    15: (
        "Which switching architecture removes the movement and network-span limits described "
        "earlier?"
    ),
    16: "How are candidate specification updates judged for inclusion in a release?",
    17: "What compatibility guarantee must a schema update preserve?",
    18: "Which accreditation programme ended, and what does that mean for the specification?",
    19: "Which scoring-system versions are permitted, and which prior ambiguity was resolved?",
    20: (
        "How is support provided for consumers who want only identifiers and results, not full "
        "content?"
    ),
    21: "Why was an annex created alongside the main specification?",
    22: "What should a tool do with results when it processes content from an older revision?",
    23: (
        "Why is a vulnerability identifier a weaker basis for a patch definition than other "
        "identifiers?"
    ),
    24: "Which activities happen at system level during the planning stage, and in what order?",
    25: "At which organisational levels may a monitoring strategy be developed?",
    26: "Why does exchanging keys on physical media stop working as an organisation grows?",
    27: (
        "Which duties fall to the senior official responsible for an agency's information "
        "technology?"
    ),
    28: "What are the legitimate reasons for extracting a key from a cryptographic module?",
    29: "What does protecting a key involve besides preventing disclosure?",
    30: "Why is data encrypted today at risk even if the algorithm is currently sound?",
    34: "Who holds a stake in an organisation's continuous monitoring capability?",
    35: "What outcomes can an assessment procedure produce, and when must it be annotated?",
    36: "How many parts must a tester confirm are needed to reconstruct a split secret?",
    37: "What must a tester confirm about hardware attached to each processor?",
    38: "What must a vendor list about the platforms a software module was validated on?",
    39: (
        "What must a security policy state about behaviour outside the permitted temperature range?"
    ),
    45: "Can firmware be written to flash while the machine keeps running?",
    47: "What accompanying information should be published with a shared threat indicator?",
    48: "What should sharing procedures say about data that may identify individuals?",
    50: (
        "How should a general-purpose machine be arranged when unvalidated code may also run on it?"
    ),
    51: "When is the nationality element mandatory in an enrolment record, and how is it written?",
    52: "Which standards govern the facial image captured during enrolment?",
    53: "What must an issuer do when a token holding a private key is lost or reassigned?",
    54: "Where is the key pair created when a hardware token is issued to an employee?",
    55: "Why should engineering focus on what could happen rather than what is expected to?",
    56: "What triggers a project to revisit its plan on security grounds?",
    57: "What determines how much of the guidance an organisation should apply?",
    58: (
        "Which defensive approach can mislead an adversary but also complicate one's own "
        "operations?"
    ),
    60: "What is claimed about the origin of the system and its use of external code?",
    61: "What does an access decision evaluate besides the identity of the requester?",
    62: "How do identity lists and role-based schemes relate to attribute-based access control?",
    63: "What can go wrong when an application requests more privileges than it needs?",
    64: "Why is advertising-borne malicious code so widespread on handsets?",
    65: "Which competing objectives must a smartphone or tablet satisfy at the same time?",
    66: (
        "Which group evaluated standards against the requirements in the referenced interagency "
        "report?"
    ),
    69: "What are the three uses of cryptography in application allow-listing?",
    70: "In which environments is allow-listing easiest to apply?",
    71: "Why would anyone test an approximate matching algorithm?",
    72: "What are the two kinds of similarity query, and how do they differ?",
    75: "Who can initiate a change request, and what else can prompt one?",
    76: "What activities help promote security awareness across an organisation?",
    77: "What does each cell of the forged-data table express?",
    78: "Which publications specify the key derivation methods built on a keyed hash?",
    79: "What happens to the contents of a deleted file on disk?",
    80: "What must a user do before encrypted content becomes readable?",
    81: "What does the section on transport-layer virtual private networks cover?",
    82: "How are redundancy and capacity growth handled for these gateways?",
    83: "What expertise should someone conducting a security assessment have?",
    84: "When does reporting happen relative to the other phases of a penetration test?",
    85: "How does the newer addressing scheme support route summarisation?",
    86: "Why should a concealed name server's address and software version stay unpublished?",
    87: "When might personal data carried off-site warrant a higher impact rating?",
    88: "What is the purpose of staff education about handling personal data?",
    89: "Why is preserving data integrity a central duty for whoever runs a server?",
    90: "When are conventional backups insufficient for a frequently changing service?",
    91: "What is the most common reason organisations adopt full virtualisation?",
    92: "What should happen to a machine's contents before it leaves the organisation?",
    94: "How does a compliance-class definition map onto a returned result?",
    95: "What does this suite of specifications standardise?",
    98: "What determines whether key derivation performance is acceptable?",
    99: "Which security model specifies key localisation for network management version 3?",
    100: "What two values come out of the transport-layer key exchange, and how are they used?",
    101: "What data sources can be combined to spot unusual activity?",
    102: "Which controls warrant more frequent automated checking, and which need it less?",
    103: "Which category of systems falls outside the scope of these standards?",
    104: "How does one decide how many interacting variables to cover when testing combinations?",
    105: "What must an organisation do before issuing a contract for external computing services?",
    106: "Which aspects of identity management involve handling personal data?",
    107: (
        "What does the customer control, and what remains with the provider, in a platform service?"
    ),
    108: "How do users reach applications in the hosted-software delivery model?",
    109: "Why should physical site security at a provider factor into supplier selection?",
    110: "Who is responsible for confirming a provider's website is genuine?",
    111: "Which firmware threat is described as among the hardest to prevent?",
    112: "Who develops the low-level firmware that starts a machine before the operating system?",
    113: "Can existing configuration management tooling be reused for wireless security settings?",
    114: (
        "What other kinds of simultaneous connection should be considered besides wired and "
        "wireless?"
    ),
}
