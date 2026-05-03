from core.command_router import initialize_command_services, route_command


def main():
    initialize_command_services()

    ok1 = route_command("policy dry run on")
    r1 = route_command("open app notepad")
    ok2 = route_command("policy dry run off")

    print("policy_on:", ok1)
    print("dry_run_open:", r1)
    print("policy_off:", ok2)

    assert "dry-run" in r1.lower() or "simulation" in r1.lower() or "simulated" in r1.lower()


if __name__ == "__main__":
    main()
