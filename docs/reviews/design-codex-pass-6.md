1. **Resolved — RESIDUAL closed:** The owner loop ticks without SSE input and runs retries and expiry on each iteration. [§5.3](docs/DESIGN.md:231)
2. **Partially resolved — BLOCKER:** `sigwait` starts the deadline independently of asyncio, but a single blocking `os.write` can still wait indefinitely on full stderr. [§5.7](docs/DESIGN.md:492) [Pipe behavior](https://man7.org/linux/man-pages/man7/pipe.7.html)
3. **Partially resolved — BLOCKER:** The exit-status table now distinguishes write roles, but reading separate counters without a coherent snapshot can miss an authoritative write as it moves from unwritten to in flight. [§5.7](docs/DESIGN.md:504)
4. **Resolved — RESIDUAL acknowledged:** Hook determinism is stated as a contract, and §5.5 states the consequence when a hook violates it. [§5.9](docs/DESIGN.md:600)

The stated early signal mask covers later executor threads; asyncio signal callbacks for those blocked signals will not run, so the watchdog must remain their handler. The timed work addresses the idle-stream case.

NOT READY: hard-deadline enforcement, exit-status counter consistency