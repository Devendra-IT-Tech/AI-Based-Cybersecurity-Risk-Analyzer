"""
sample_cases.py
---------------
Built-in demo scenarios for immediate demonstration.
Each case includes a title, description, and expected risk profile.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class SampleCase:
    title: str
    description: str
    expected_risk: str  # LOW | MEDIUM | HIGH | CRITICAL


SAMPLE_CASES: List[SampleCase] = [
    SampleCase(
        title="🎣 Phishing Credential Compromise",
        description=(
            "An employee received an email that appeared to be from the company's IT department "
            "asking them to verify their account credentials. The email contained a link to a "
            "convincing but fake login page. The employee entered their username and corporate "
            "password. Shortly afterwards, several login attempts from foreign IP addresses were "
            "detected on the employee's account. The employee has access to the company's HR "
            "database which contains PII for over 500 staff members."
        ),
        expected_risk="CRITICAL",
    ),
    SampleCase(
        title="🔑 Suspicious Login Activity",
        description=(
            "The security team received an alert about multiple failed login attempts on an "
            "administrator account followed by one successful login at 3:15 AM from an unfamiliar "
            "geographic location (Eastern Europe). The administrator's account has read access to "
            "financial records and system configuration files. The legitimate account owner was "
            "not travelling and did not initiate the login."
        ),
        expected_risk="HIGH",
    ),
    SampleCase(
        title="🦠 Malware Alert on Workstation",
        description=(
            "The endpoint protection software on a developer's workstation triggered an alert for "
            "a detected Trojan. The developer had downloaded a software library from an unofficial "
            "forum earlier that morning. The workstation is connected to the internal development "
            "network and the developer has access to source code repositories. The malware appears "
            "to have been active for approximately 6 hours before detection."
        ),
        expected_risk="HIGH",
    ),
    SampleCase(
        title="📁 Unusual File Access Pattern",
        description=(
            "A monitoring system detected that a standard user account accessed and downloaded "
            "over 200 confidential documents from a shared drive in a 30-minute window. This is "
            "highly unusual behaviour for this account which normally accesses only 2-3 files per "
            "day. The files include project proposals, budget documents, and client contracts. "
            "The account belongs to an employee who submitted their resignation two weeks ago."
        ),
        expected_risk="HIGH",
    ),
    SampleCase(
        title="📧 Possible Sensitive Data Exposure",
        description=(
            "A staff member accidentally emailed an attachment containing a spreadsheet with "
            "customer names, email addresses, and partial credit card numbers to an external "
            "personal email address instead of an internal colleague. The error was noticed "
            "about 20 minutes after sending. The spreadsheet contained data for approximately "
            "150 customers. The staff member has reported the incident to their manager."
        ),
        expected_risk="MEDIUM",
    ),
]


def get_sample_titles() -> List[str]:
    """Return a list of sample case titles for display in the UI."""
    return [case.title for case in SAMPLE_CASES]


def get_sample_by_title(title: str) -> SampleCase | None:
    """Find and return a sample case by its title."""
    for case in SAMPLE_CASES:
        if case.title == title:
            return case
    return None
