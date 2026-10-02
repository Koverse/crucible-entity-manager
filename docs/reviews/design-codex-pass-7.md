The nonblocking `os.write` resolves the full-stderr-pipe blocker.

**BLOCKER — [§5.7](docs/DESIGN.md:516):** The snapshot is published *after* an accounting change. At the deadline, the watchdog can read the previous snapshot and miss a newly outstanding authoritative write, yielding exit code 0. The revised text needs one published state that determines both the diagnostic and exit code.

NOT READY: accounting publication race