import json

from hcmd.cli import build_command
from hcmd.constants import OS
from hcmd.core.context import SystemContext
from hcmd.core.metrics import benchmark
from hcmd.core.nlp import interpret, interpret_plan


def context(os_type=OS.WINDOWS):
    return SystemContext(os_type=os_type, cwd="C:\\work")


def test_fallback_interprets_common_file_commands():
    create = interpret("create a file named report.txt")
    assert create["intent"] == "CREATE_FILE"
    assert create["path"] == "report.txt"
    move = interpret("move report.txt to backup")
    assert move["intent"] == "MOVE_FILE"
    assert move["src"] == "report.txt"
    assert move["dst"] == "backup"


def test_fallback_interprets_git_commands():
    result = interpret("git checkout main")
    assert result == {
        "ok": True,
        "intent": "GIT_CHECKOUT",
        "confidence": 1.0,
        "branch": "main",
        "pattern": None,
    }


def test_plan_splits_commands():
    plan = interpret_plan("create file one.txt and create file two.txt")
    assert plan["ok"] is True
    assert [step["intent"] for step in plan["steps"]] == [
        "CREATE_FILE",
        "CREATE_FILE",
    ]


def test_command_builder_is_cross_platform():
    assert build_command("LIST_FILES", {}, context()) == "Get-ChildItem"
    assert build_command("LIST_FILES", {}, context(OS.LINUX)) == "ls -la"


def test_benchmark_reports_latency_metrics():
    result = benchmark(lambda command: bool(command), ["a", "b", "c"])
    assert result["samples"] == 3
    assert result["successful_samples"] == 3
    assert result["success_rate"] == 1.0
    assert result["average_ms"] >= 0
    assert result["median_ms"] >= 0
    assert result["p95_ms"] >= 0
    assert result["throughput_per_second"] > 0
