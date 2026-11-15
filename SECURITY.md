# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Send a report to
**contact@openmvis.com** with the affected version (`mvis --version`), the
steps to reproduce and, if you have one, a proposed fix. Where the repository
offers GitHub's private vulnerability reporting, that channel works as well.

You will get an acknowledgement within a week. Fixes ship as a patch release
and are credited in the changelog unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---------|-----------|
| latest 0.x release | yes |
| older releases | no, upgrade to the latest release |

## Scope

MVIS is a calibration toolbox that reads recordings and configuration files you
provide. Reports about crafted inputs (recordings, YAML configs, MCAP or ROS
bags) that lead to code execution or data exfiltration are in scope. External
tools MVIS only invokes (Kalibr, Basalt, ffmpeg) are out of scope; report those
upstream.
