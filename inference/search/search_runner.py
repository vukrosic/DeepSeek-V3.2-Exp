import json
from argparse import ArgumentParser
from datetime import datetime, UTC
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs"
REQUIRED_KEYS = {
    "id",
    "title",
    "scope",
    "goal",
    "depends_on",
    "primary_script",
    "target_metrics",
    "exactness_gate",
    "allowed_changes",
    "forbidden_changes",
    "candidate_variants",
}


def load_tasks():
    tasks = []
    for path in sorted(TASKS_DIR.glob("*.json")):
        with open(path) as f:
            task = json.load(f)
        task["_path"] = str(path.relative_to(ROOT.parent))
        tasks.append(task)
    return tasks


def get_task(task_id: str):
    for task in load_tasks():
        if task["id"] == task_id:
            return task
    raise SystemExit(f"Task not found: {task_id}")


def cmd_list() -> None:
    for task in load_tasks():
        deps = ",".join(task.get("depends_on", [])) or "-"
        print(f'{task["id"]}: {task["title"]}')
        print(f'  scope: {task["scope"]}')
        print(f'  deps: {deps}')
        print(f'  script: {task.get("primary_script", "-")}')


def validate_tasks():
    tasks = load_tasks()
    ids = set()
    errors = []

    for task in tasks:
        missing = sorted(REQUIRED_KEYS - task.keys())
        if missing:
            errors.append(f'{task.get("id", task["_path"])} missing keys: {", ".join(missing)}')
        task_id = task.get("id")
        if task_id in ids:
            errors.append(f"duplicate task id: {task_id}")
        ids.add(task_id)

    for task in tasks:
        for dep in task.get("depends_on", []):
            if dep not in ids:
                errors.append(f'{task["id"]} depends on unknown task: {dep}')

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)

    print(f"validated {len(tasks)} task manifests")


def cmd_graph() -> None:
    for task in load_tasks():
        print(task["id"])
        deps = task.get("depends_on", [])
        if not deps:
            print("  <- root")
            continue
        for dep in deps:
            print(f"  <- {dep}")


def cmd_show(task_id: str) -> None:
    print(json.dumps(get_task(task_id), indent=2))


def cmd_init_run(task_id: str, label: str) -> None:
    task = get_task(task_id)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe_label = label.strip().replace(" ", "-") if label else "run"
    run_dir = RUNS_DIR / f"{timestamp}-{task_id}-{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "task_snapshot.json", "w") as f:
        json.dump(task, f, indent=2)
        f.write("\n")
    with open(run_dir / "notes.md", "w") as f:
        f.write(f"# {task_id} Run\n\n")
        f.write(f"## Task\n{task['title']}\n\n")
        f.write("## Goal\n")
        f.write(f"{task['goal']}\n\n")
        f.write("## Exactness Gate\n")
        for gate in task.get("exactness_gate", []):
            f.write(f"- {gate}\n")
        f.write("\n## Candidate Variants\n")
        for variant in task.get("candidate_variants", []):
            f.write(f"- {variant}\n")
        f.write("\n## Findings\n- \n")
        f.write("\n## Decision\n- keep\n- reject\n- revisit\n")
    with open(run_dir / "results.json", "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "label": safe_label,
                "status": "initialized",
                "variants": [],
                "decision": "",
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(run_dir)


def main() -> None:
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    sub.add_parser("validate")
    sub.add_parser("graph")

    show = sub.add_parser("show")
    show.add_argument("task_id")

    init_run = sub.add_parser("init-run")
    init_run.add_argument("task_id")
    init_run.add_argument("--label", default="")

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "validate":
        validate_tasks()
    elif args.cmd == "graph":
        cmd_graph()
    elif args.cmd == "show":
        cmd_show(args.task_id)
    elif args.cmd == "init-run":
        cmd_init_run(args.task_id, args.label)


if __name__ == "__main__":
    main()
