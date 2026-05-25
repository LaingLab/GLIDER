# Security policy

GLIDER controls physical laboratory hardware — heaters, valves, optogenetic LEDs, motorized actuators — and runs as a kiosk on user-trusted Raspberry Pi appliances. Bugs that compromise hardware safety, data integrity, or operator control are treated as security issues, not just functional bugs.

## Reporting a vulnerability

**Do not open a public GitHub issue for security-sensitive disclosures.**

For:

- Hardware-safety bugs (outputs driving after STOP, hung emergency-stop, etc.)
- Data-integrity bugs that produce silently corrupted scientific output
- Privilege-escalation paths involving the Pi `sudoers-glider` rule or the in-app updater
- Code-execution paths through the plugin loader
- Anything else that you suspect is sensitive

email the maintainers privately. The maintainer contact is the GitHub repository owner (`LaingLab` on GitHub). If that contact is unresponsive within 7 days, you may open a public issue, but please give the maintainer the first week to respond privately.

When you report, please include:

- A clear description of the vulnerability
- Steps to reproduce
- The affected version (`glider --version`)
- The host platform (OS, Python version, board if relevant)
- Any proof-of-concept code or `.glider` file demonstrating the issue
- Your suggested mitigation, if you have one

## Supported versions

Security fixes are applied to the latest released minor version. Earlier versions are not patched.

| Version | Supported |
|---|---|
| 1.0.x (current) | ✅ |
| < 1.0 | ❌ |

## Disclosure timeline

Once a report is received, the maintainers will:

1. Acknowledge receipt within 7 days.
2. Investigate and confirm or rule out the issue within 14 days.
3. Develop a fix and ship it in a patch release within 30 days for confirmed issues, faster for actively exploited or hardware-safety-critical issues.
4. Coordinate disclosure timing with the reporter; default to public disclosure 30 days after a fix is released.

Reporters who follow this policy are credited in the release notes (with their permission).

## Hardening notes

A few areas of the codebase warrant extra reviewer attention:

- **Plugin loader** (`src/glider/plugins/plugin_manager.py`) — executes arbitrary Python from `~/.glider/plugins/`. Directory-discovery plugins are opt-in (`enable_directory_plugins=False` default). Plugins should be treated as trusted code; if you install a plugin from a third party, you are running their code with the same privileges as GLIDER itself.
- **Pi `sudoers-glider` rule** (`packaging/pi/stage-glider/files/sudoers-glider`) — grants `NOPASSWD` execution of one specific updater script to the `glider` user, used by the in-app updater. The script must remain root-owned and only modifiable by root, or the rule becomes a privilege-escalation path. The packaging stage script `00-run.sh` sets the correct ownership; if you modify the rule, preserve those invariants.
- **In-app updater** (`src/glider/updater.py`) — fetches GitHub Releases via HTTPS, verifies the tag matches the expected format, and on Pi invokes the sudoers-gated script to apply the update. Updates from forks or non-HTTPS sources are rejected.
- **Telemetrix and serial-port code** runs on a dedicated thread bridging to the qasync event loop. Untrusted serial input could in principle drive that thread into a bad state; in practice the only callers are physical USB Arduinos which are user-controlled.

## Out of scope

The following are tracked as bugs but not treated as security issues:

- The unsigned Windows installer (acknowledged in `CHANGELOG.md`).
- The unsigned macOS bundle (acknowledged in `CHANGELOG.md`).
- Issues that require physical access to the Pi to exploit.
- Issues that require an attacker to already have shell access as the `glider` user.
