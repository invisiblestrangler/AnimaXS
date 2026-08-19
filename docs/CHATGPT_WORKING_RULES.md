# ChatGPT Working Rules for AnimaXS

These are standing operating rules for ChatGPT when working on the AnimaXS project. Re-read this file before substantial investigation, repository mutation, CI work, or handoff planning.

## 1. Role separation

ChatGPT is the **planner/investigator**. It must do the investigative work itself: inspect the repository, determine root causes, identify exact files and code paths, define acceptance criteria, and independently verify results.

Execution agents are for **implementation of already-decided work**. Do not delegate open-ended investigation, diagnosis, bottleneck finding, or "figure out what to do" tasks to an execution agent unless it is genuinely unavoidable.

After an execution agent finishes, independently verify the actual repository state, diffs, tests, artifacts, and requirements rather than trusting the agent summary.

## 2. Investigation mode vs execution mode

Keep these phases separate.

### Investigation mode

Exploration is allowed and expected: inspect broadly, test hypotheses, compare implementations, run experiments, and collect evidence until the root cause and intended fix are sufficiently established.

### Execution mode

Once the cause/fix is established, become deliberately conservative. Do not re-investigate settled conclusions unless new evidence contradicts them.

Use this sequence:

1. state the exact objective;
2. identify the authoritative source of truth (existing patch/report/current repo state);
3. verify the expected branch/file state;
4. make only the already-decided change;
5. verify the resulting diff/state;
6. stop.

Execution should be boring and deterministic.

## 3. GitHub access rule

All GitHub reads and writes must use the **GitHub connector**.

Do **not** use shell `git`, `curl`, raw GitHub HTTP/network access, or other local-network workarounds to reach GitHub. Those routes are not part of the supported workflow here.

Local shell/container tools are fine for **local-only** work such as applying a patch, hashing files, comparing text, unpacking artifacts, or running local scripts. They must not be used to communicate with GitHub.

Prefer normal connector content operations such as `fetch_file`, `create_file`, `update_file`, PR/commit inspection, and compare operations. Do not silently switch to blob/tree plumbing or another write mechanism because the normal path encountered friction.

## 4. Clore rule

Clore is an **experimental workbench**, not the AnimaXS repository coding workspace.

Testing scripts and experiments may be created on Clore when useful. Do not implement or edit the actual AnimaXS application/repository code there.

Keep heavy or persistent project artifacts off low-storage VPS instances. Use the appropriate persistent storage/workflow instead.

## 5. Existing confirmed work is authoritative

When an existing patch, report, confirmation file, test artifact, or previously verified conclusion already defines the fix, treat it as the source of truth unless there is concrete contradictory evidence.

Do not turn a narrow "apply this known patch" task back into a fresh investigation.

If an exact patch already exists, apply and verify that exact patch rather than reconstructing or paraphrasing its changes from memory.

## 6. Pre-write checks

Before any repository mutation, verify the relevant current state through the GitHub connector, including branch/head and the target file/blob when applicable.

If the current state differs from the expected state, stop before writing and report the discrepancy.

For patch-based changes, prove that the patch applies cleanly to the expected source and that its scope matches the declared change before writing.

## 7. Hard stop conditions

After the **first unexpected failure during a repository mutation workflow, stop and report before trying a different mechanism**.

Stop immediately if any of the following occurs:

- a required GitHub connector read fails unexpectedly;
- the branch/head changed unexpectedly;
- the target file/blob does not match the expected pre-change state;
- an existing patch does not apply cleanly;
- the patch changes anything outside its declared scope;
- a supposedly untouched path or subsystem changes;
- the intended result cannot be proven exact enough before writing;
- the normal GitHub connector write rejects the change;
- the post-write GitHub state/diff does not match the intended result;
- a workaround would require switching to shell/network GitHub access or an unplanned mutation mechanism.

On stop, report:

1. the exact operation that failed;
2. the exact error or mismatch;
3. the current repository state as known;
4. the safest next option.

Do not silently retry many unrelated approaches.

## 8. Post-write verification

After every repository write, verify through the GitHub connector that:

- the branch advanced as expected;
- only intended files changed;
- the target diff matches the intended change;
- explicitly out-of-scope code remains unchanged;
- test/regression changes match the intended patch or plan.

When exactness matters, compare hashes or exact patch output rather than relying only on visual similarity.

## 9. Context discipline

Long context is not an excuse for forgetting project constraints. Before substantial work, re-read the relevant project rules, current task evidence, and existing confirmed conclusions.

Maintain a compact working contract for the current task:

- objective;
- source of truth;
- allowed systems/tools;
- forbidden detours;
- expected pre-state;
- acceptance criteria;
- hard stop conditions.

Do not ask the user to restate constraints that are already known from the project context or repository documentation.

## 10. User updates

During long-running work, provide regular concise progress updates so the user can see the current state, important findings, and whether anything has gone wrong.

If a task becomes unexpectedly difficult or a key operation fails, say so promptly rather than continuing through many hidden fallback attempts.

## 11. Planning quality

Implementation plans for execution agents must be concrete enough that the agent is primarily writing code, not discovering what to do. Include exact files, exact code paths, intended changes, constraints, validation steps, and relevant code snippets where useful.

Do not optimize for short-term convenience if it creates technical debt or forces repeated debugging later. Prefer doing the difficult but correct implementation once when the evidence supports it.

## 12. Default repository-mutation pattern

For narrow known changes, default to:

`verify current head -> verify target blob/file -> reproduce exact intended change locally if needed -> GitHub connector write -> GitHub connector diff/hash verification -> stop`

Do not expand the scope unless new evidence requires it.