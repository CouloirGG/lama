"""
LAMA — Jira CLI helper

Create, list, and manage Jira tickets from the command line.
Credentials loaded from .env in project root.

Usage:
    python scripts/jira_cli.py create "Summary" "Description" [--type Story|Bug|Task]
    python scripts/jira_cli.py list [--status "To Do"|"In Progress"|"Done"]
    python scripts/jira_cli.py view PT-44
    python scripts/jira_cli.py comment PT-44 "comment text"
    python scripts/jira_cli.py transition PT-44 "In Progress"
"""

import argparse
import sys
from pathlib import Path

# Load .env from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import os
from jira import JIRA


def _connect() -> JIRA:
    url = os.environ.get("JIRA_URL")
    user = os.environ.get("JIRA_USER")
    token = os.environ.get("JIRA_API_TOKEN")
    if not all([url, user, token]):
        print("ERROR: Set JIRA_URL, JIRA_USER, JIRA_API_TOKEN in .env")
        sys.exit(1)
    return JIRA(server=url, basic_auth=(user, token))


def _project() -> str:
    return os.environ.get("JIRA_PROJECT", "PT")


def cmd_create(args):
    jira = _connect()
    fields = {
        "project": {"key": _project()},
        "summary": args.summary,
        "description": args.description or "",
        "issuetype": {"name": args.type},
    }
    issue = jira.create_issue(fields=fields)
    print(f"Created: {issue.key} — {issue.fields.summary}")
    print(f"  URL: {os.environ.get('JIRA_URL')}/browse/{issue.key}")


def cmd_list(args):
    jira = _connect()
    project = _project()
    jql = f"project = {project}"
    if args.status:
        jql += f' AND status = "{args.status}"'
    jql += " ORDER BY created DESC"

    issues = jira.search_issues(jql, maxResults=args.limit)
    if not issues:
        print("No issues found.")
        return

    for issue in issues:
        status = issue.fields.status.name
        itype = issue.fields.issuetype.name
        print(f"  {issue.key:8s} [{status:12s}] ({itype:5s}) {issue.fields.summary}")


def cmd_view(args):
    jira = _connect()
    issue = jira.issue(args.key)
    print(f"{issue.key}: {issue.fields.summary}")
    print(f"  Type:     {issue.fields.issuetype.name}")
    print(f"  Status:   {issue.fields.status.name}")
    print(f"  Assignee: {issue.fields.assignee or 'Unassigned'}")
    print(f"  URL:      {os.environ.get('JIRA_URL')}/browse/{issue.key}")
    if issue.fields.description:
        print(f"\n  Description:\n    {issue.fields.description}")


def cmd_comment(args):
    jira = _connect()
    jira.add_comment(args.key, args.text)
    print(f"Comment added to {args.key}")


def cmd_transition(args):
    jira = _connect()
    transitions = jira.transitions(args.key)
    target = None
    for t in transitions:
        if t["name"].lower() == args.status.lower():
            target = t
            break
    if not target:
        available = [t["name"] for t in transitions]
        print(f"ERROR: '{args.status}' not available. Options: {available}")
        sys.exit(1)
    jira.transition_issue(args.key, target["id"])
    print(f"{args.key} -> {target['name']}")


def main():
    parser = argparse.ArgumentParser(description="LAMA Jira CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new issue")
    p_create.add_argument("summary", help="Issue summary/title")
    p_create.add_argument("description", nargs="?", default="",
                          help="Issue description")
    p_create.add_argument("--type", default="Story",
                          choices=["Story", "Bug", "Task"],
                          help="Issue type (default: Story)")

    # list
    p_list = sub.add_parser("list", help="List issues")
    p_list.add_argument("--status", help="Filter by status")
    p_list.add_argument("--limit", type=int, default=20,
                        help="Max results (default: 20)")

    # view
    p_view = sub.add_parser("view", help="View an issue")
    p_view.add_argument("key", help="Issue key (e.g. PT-44)")

    # comment
    p_comment = sub.add_parser("comment", help="Add a comment")
    p_comment.add_argument("key", help="Issue key")
    p_comment.add_argument("text", help="Comment text")

    # transition
    p_trans = sub.add_parser("transition", help="Move issue to status")
    p_trans.add_argument("key", help="Issue key")
    p_trans.add_argument("status", help="Target status name")

    args = parser.parse_args()

    cmds = {
        "create": cmd_create,
        "list": cmd_list,
        "view": cmd_view,
        "comment": cmd_comment,
        "transition": cmd_transition,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
