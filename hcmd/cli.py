import json
import sys
import os
import time

from hcmd.core.executor import CommandExecutor
from hcmd.core.context_resolver import resolve_context
from hcmd.core.nlp import interpret_plan
from hcmd.constants import OS
from hcmd.core.context import SystemContext
from hcmd.core.safety import validate_command
from hcmd.core.ambiguity import detect_ambiguity, ClarificationRequest
from hcmd.core.memory import memory
from hcmd.core.pattern_executor import resolve_pattern
from hcmd.core.rollback import build_rollback
from hcmd.core.help import show_help
from hcmd.core.metrics import benchmark


# --------------------------------------------------
# Path normalization
# --------------------------------------------------
def normalize_path(path: str | None, ctx: SystemContext) -> str | None:
    if not path:
        return None

    # Absolute paths must remain untouched
    if os.path.isabs(path):
        return os.path.abspath(path)

    if ctx.os_type == OS.WINDOWS:
        known = {
            "downloads": os.path.join(os.environ["USERPROFILE"], "Downloads"),
            "documents": os.path.join(os.environ["USERPROFILE"], "Documents"),
            "desktop": os.path.join(os.environ["USERPROFILE"], "Desktop"),
        }
        return os.path.abspath(known.get(path.lower(), path))

    known = {
        "downloads": os.path.expanduser("~/Downloads"),
        "documents": os.path.expanduser("~/Documents"),
        "desktop": os.path.expanduser("~/Desktop"),
    }
    return os.path.abspath(known.get(path.lower(), path))


# --------------------------------------------------
# Command builder
# --------------------------------------------------
def build_command(intent: str, data: dict, ctx: SystemContext) -> str | None:
    os_type = ctx.os_type

    # ---------- BASIC ----------
    if intent == "LIST_FILES":
        return "Get-ChildItem" if os_type == OS.WINDOWS else "ls -la"

    if intent == "PWD":
        return "Get-Location" if os_type == OS.WINDOWS else "pwd"

    if intent == "NAVIGATION":
        path = normalize_path(data.get("path"), ctx)
        return f'cd "{path}"' if path else None

    # ---------- FILE OPS ----------
    if intent in ("MOVE_FILE", "COPY_FILE"):
        src = normalize_path(data.get("src"), ctx)
        dst = normalize_path(data.get("dst"), ctx)
        if not src or not dst:
            return None

        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))

        if intent == "MOVE_FILE":
            return (
                f'Move-Item "{src}" "{dst}"'
                if os_type == OS.WINDOWS
                else f'mv "{src}" "{dst}"'
            )

        return (
            f'Copy-Item "{src}" "{dst}"'
            if os_type == OS.WINDOWS
            else f'cp "{src}" "{dst}"'
        )

    if intent == "CREATE_FILE":
        name = normalize_path(data.get("path"), ctx)
        return (
            f'New-Item -ItemType File "{name}"'
            if os_type == OS.WINDOWS
            else f'touch "{name}"'
        )

    if intent == "CREATE_DIR":
        name = normalize_path(data.get("path"), ctx)
        return (
            f'New-Item -ItemType Directory "{name}"'
            if os_type == OS.WINDOWS
            else f'mkdir "{name}"'
        )

    if intent == "READ_FILE":
        name = normalize_path(data.get("path"), ctx)
        return (
            f'Get-Content -LiteralPath "{name}"'
            if os_type == OS.WINDOWS
            else f'cat "{name}"'
        ) if name else None

    if intent == "DELETE_FILE":
        target = normalize_path(data.get("path"), ctx)
        if not target:
            return None
        if os.path.isdir(target):
            print(f"ERROR: '{target}' is a directory. Use 'delete directory'.")
            return None
        return (
            f'Remove-Item -LiteralPath "{target}" -Force'
            if os_type == OS.WINDOWS
            else f'rm "{target}"'
        )

    if intent == "DELETE_DIR":
        target = normalize_path(data.get("path"), ctx)
        if not target:
            return None
        return (
            f'Remove-Item -LiteralPath "{target}" -Recurse -Force'
            if os_type == OS.WINDOWS
            else f'rm -r "{target}"'
        )

    if intent == "RENAME_FILE":
        src = normalize_path(data.get("src"), ctx)
        dst = normalize_path(data.get("dst"), ctx)
        if not src or not dst:
            return None
        return (
            f'Rename-Item "{src}" "{dst}"'
            if os_type == OS.WINDOWS
            else f'mv "{src}" "{dst}"'
        )

    # ---------- SYSTEM ----------
    if intent == "SYSTEM_INFO":
        return "systeminfo" if os_type == OS.WINDOWS else "uname -a"

    if intent == "PROCESS_LIST":
        return "tasklist" if os_type == OS.WINDOWS else "ps aux"

    if intent == "NETWORK_INFO":
        return "ipconfig" if os_type == OS.WINDOWS else "ifconfig"

    if intent == "PROCESS_KILL":
        target = data.get("target")
        if not target:
            return None
        if os_type == OS.WINDOWS:
            return (
                f'taskkill /PID {target} /F'
                if target.isdigit()
                else f'taskkill /IM {target}.exe /F'
            )
        return f'kill {target}' if target.isdigit() else f'killall {target}'

    # ---------- GIT ----------
    if intent == "GIT_STATUS":
        return "git status"
    if intent == "GIT_BRANCH":
        return "git branch"
    if intent == "GIT_LOG":
        return "git log --oneline --decorate --graph"
    if intent == "GIT_PULL":
        return "git pull"
    if intent == "GIT_PUSH":
        return "git push"
    if intent == "GIT_RESET":
        return "git reset"
    if intent == "GIT_STASH":
        return "git stash"
    if intent == "GIT_COMMIT":
        return "git commit"
    if intent == "GIT_REPO_STATUS":
        return "git rev-parse --is-inside-work-tree"
    if intent == "GIT_ADD":
        return f'git add "{data.get("path")}"'
    if intent == "GIT_CHECKOUT":
        return f'git checkout "{data.get("branch")}"'
    if intent == "GIT_CLONE":
        return f'git clone "{data.get("repo")}"'

    return None


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main() -> int:
    explain = "--explain" in sys.argv
    if explain:
        sys.argv.remove("--explain")

    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")

    metrics_enabled = "--metrics" in sys.argv
    if metrics_enabled:
        sys.argv.remove("--metrics")

    if len(sys.argv) < 2:
        print("Usage: hcmd <command> [--dry-run] [--explain] [--metrics]")
        return 1

    text = " ".join(sys.argv[1:])

    # ---------- HELP ----------
    if text.strip().lower().startswith("help"):
        parts = text.split(maxsplit=1)
        show_help(parts[1] if len(parts) > 1 else None)
        return 0

    if text.strip().lower() == "benchmark":
        benchmark_cases = [
            "list files",
            "where am i",
            "create file benchmark.txt",
            "move benchmark.txt to backup",
            "git status",
            "git branch",
            "git log",
            "git checkout main",
            "stage app.py",
            "read file README.md",
        ]
        commands = benchmark_cases * 50
        result = benchmark(lambda command: interpret_plan(command).get("ok", False), commands)
        result["benchmark_type"] = "interpretation_only"
        result["unique_cases"] = len(benchmark_cases)
        result["repetitions_per_case"] = 50
        print(json.dumps(result, indent=2))
        return 0 if result["success_rate"] == 1.0 else 1

    ctx = resolve_context()
    started_at = time.perf_counter()

    # ---------- ROLLBACK ----------
    if text.strip().lower() == "rollback":
        if not memory.history:
            print("Nothing to rollback")
            return 0

        rollback_cmd = memory.history.pop()
        print(f"Rolling back: {rollback_cmd}")

        executor = CommandExecutor()
        ok, out = executor.execute(rollback_cmd)
        if not ok:
            print(f"ERROR during rollback: {out}")
            return 1

        memory.save()
        return 0

    # ---------- NLP ----------
    plan = interpret_plan(text)
    if not plan.get("ok"):
        print(f"ERROR: {plan.get('reason')}")
        return 1

    steps = plan["steps"]
    executor = CommandExecutor()
    explain_log = []

    if len(steps) > 1 and not explain:
        print("I will execute the following steps:")
        for i, s in enumerate(steps, 1):
            print(f"{i}. {s['intent']}")
        try:
            proceed = input("Proceed? [y/N]: ").lower()
        except EOFError:
            print("ERROR: Confirmation was unavailable in this environment")
            return 1
        if proceed != "y":
            return 1

    for step in steps:
        intent = step["intent"]

        skip_ambiguity = step.get("from_pronoun", False)
        clarification = None if skip_ambiguity else detect_ambiguity(intent, step, ctx)

        if isinstance(clarification, ClarificationRequest):
            print(f"CLARIFICATION REQUIRED: {clarification.reason}")
            for i, opt in enumerate(clarification.options, 1):
                print(f"{i}) {opt}")
            try:
                selection = int(input("> ")) - 1
            except (EOFError, ValueError):
                print("ERROR: A valid clarification choice is required")
                return 1
            if not 0 <= selection < len(clarification.options):
                print("ERROR: Clarification choice is out of range")
                return 1
            step["path"] = clarification.options[selection]

        cmd = build_command(intent, step, ctx)
        if not cmd:
            print("ERROR: Unsupported command")
            return 1

        safety = validate_command(intent, cmd, ctx)

        explain_log.append({
            "intent": intent,
            "command": cmd,
            "path": step.get("path"),
            "src": step.get("src"),
            "dst": step.get("dst"),
            "from_pronoun": step.get("from_pronoun", False),
            "safety": safety.warning if not safety.safe else "safe"
        })

        if explain:
            continue

        if not safety.safe and not dry_run:
            try:
                proceed = input(f"{safety.warning} Proceed? [y/N]: ").lower()
            except EOFError:
                print("ERROR: Confirmation was unavailable in this environment")
                return 1
            if proceed != "y":
                return 1

        if dry_run:
            print(f"[DRY RUN] {cmd}")
            continue

        ok, out = executor.execute(cmd)
        if not ok:
            print(f"ERROR: {out}")
            return 1
        if out:
            print(out)

        if intent.startswith("GIT_"):
            memory.last_git_intent = intent

        # ---------- MEMORY (SINGLE SOURCE OF TRUTH) ----------
        if intent in ("CREATE_FILE", "DELETE_FILE", "CREATE_DIR", "DELETE_DIR"):
            p = normalize_path(step.get("path"), ctx)
            if p:
                memory.push_object(p)

        elif intent == "MOVE_FILE":
            src = normalize_path(step.get("src"), ctx)
            dst = normalize_path(step.get("dst"), ctx)
            if src and dst:
                moved = os.path.join(dst, os.path.basename(src)) if os.path.isdir(dst) else dst
                memory.push_object(os.path.abspath(moved))

        elif intent == "COPY_FILE":
            src = normalize_path(step.get("src"), ctx)
            dst = normalize_path(step.get("dst"), ctx)
            if src and dst:
                copied = os.path.join(dst, os.path.basename(src)) if os.path.isdir(dst) else dst
                memory.push_object(os.path.abspath(copied))

        elif intent == "RENAME_FILE":
            dst = normalize_path(step.get("dst"), ctx)
            if dst:
                memory.push_object(os.path.abspath(dst))

        rb = build_rollback(intent, step, ctx)
        if rb:
            memory.history.append(rb)

    if explain:
        print("\nExplanation\n")
        print(f"Input: {text}\n")
        for i, e in enumerate(explain_log, 1):
            print(f"Step {i}:")
            for k, v in e.items():
                if v:
                    print(f"  {k}: {v}")
            print()
        print("No commands executed (--explain).")
        return 0

    memory.save()
    if metrics_enabled:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(json.dumps({
            "command": text,
            "steps": len(steps),
            "elapsed_ms": round(elapsed_ms, 3),
            "dry_run": dry_run,
            "executed": not explain and not dry_run,
        }, indent=2))
    return 0


def run():
    sys.exit(main())


if __name__ == "__main__":
    run()
