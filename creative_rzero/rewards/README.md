# Reward strategies

A **solver reward strategy** answers one question: given one prompt group's
judged criterion scores, what per-sample `overall` reward does GRPO see?
Strategies are plugged in by name via `creative_rzero/rewards/registry.py`
(T3.3) and selected per experiment with `rewards.solver.type: <name>` in an
exp config — no changes to the caller that owns judge scoring, filters, and
logging (`examples/reward_function/creative_solver_caller.py`).

Two strategies exist today: `rank` ([solver/rank.py](solver/rank.py)) and
`raw` ([solver/raw.py](solver/raw.py)).

This only covers the **solver** role. The challenger reward
(`examples/reward_function/creative_writing_caller.py`, `uncertainty`) is
not on the registry yet — see "Adding a challenger strategy" below.

## Checklist: adding a new solver reward strategy

**1. Write the strategy — `creative_rzero/rewards/solver/<name>.py`**

Subclass `RewardStrategy` and implement `score_group(GroupScores) ->
GroupReward`. It must be self-contained: no judge calls, no env reads, no
I/O — pure math over the scores it's handed, so it's testable against a
plain `GroupScores` fixture with nothing to mock. `GroupScores.all_failed`
and `.low_quality` are precomputed with one shared definition for every
strategy (see registry.py's docstring on `GroupScores`) — read them, don't
recompute them; only decide what your strategy *does* about them (compare
`RankReward`, which collapses a low-quality group to a uniform 0.5, against
`RawReward`, which leaves it alone).

```python
from creative_rzero.rewards.registry import GroupReward, GroupScores, RewardStrategy, register

@register("zscore")
class ZScoreReward(RewardStrategy):
    def score_group(self, group: GroupScores) -> GroupReward:
        if group.all_failed:
            return GroupReward(overall={idx: 0.0 for idx in group.scores})
        ...
```

**2. Wire it into the package — `creative_rzero/rewards/solver/__init__.py`**

Import the new class so `@register` actually runs. Registration is a side
effect of import: a strategy file that's never imported is invisible to
`get_strategy`, no matter how correctly it's written.

**3. Allow it in config validation — `creative_rzero/config.py`**

Add `"<name>"` to `VALID_SOLVER_REWARDS`. Without this, an experiment
config with `rewards.solver.type: <name>` fails `ConfigError` at
`config.load()` — before any GPU or judge spend, which is the point.

**4. Allow it in verl_entry's dispatch gate — `creative_rzero/rewards/verl_entry.py`**

Add `"<name>"` to the `_ROLE_IMPLS["solver"]` tuple. **This is the step
that's easy to forget.** `verl_entry.py` doesn't call the registry
directly — it dispatches to `creative_solver_caller.compute_score`, which
is what actually resolves the strategy via the registry at run time.
`_ROLE_IMPLS` is a separate, independent allow-list of what that specific
caller is known to implement; skip this step and you'll get `rewards.
solver.type='<name>' is not available yet` even though `get_strategy
("<name>")` works fine on its own. (`verl_entry.py` has a standing TODO to
collapse this into a direct registry lookup once the caller itself is
registry-driven end to end — until then, both lists must agree.)

**5. Reference it from an experiment config — `configs/exp/<something>.yaml`**

```yaml
rewards:
  solver:
    type: <name>
```

**6. Test it — `tests/test_solver_reward_strategies.py`**

Add a test class alongside `TestRankReward`/`TestRawReward`: construct
`GroupScores` fixtures directly (ties, `G_eff == 1`, `all_failed`,
`low_quality`) and assert on `.score_group(...).overall`. No judge, no
mock agent, no network — that's the entire benefit of keeping strategies
self-contained.

## Why three separate allow-lists?

`registry._REGISTRY`, `config.VALID_SOLVER_REWARDS`, and `verl_entry.
_ROLE_IMPLS` each guard a different failure and a different time:

| List | Guards against | Fails at |
|---|---|---|
| `registry._REGISTRY` | strategy module never imported | first `get_strategy()` call |
| `config.VALID_SOLVER_REWARDS` | typo'd/unknown reward name in an exp config | `config.load()` (startup, before any run) |
| `verl_entry._ROLE_IMPLS["solver"]` | a name the registry knows but `creative_solver_caller.py` hasn't been told to expect | first training step's reward call |

They're redundant by design during this transitional period (the caller
still owns judge/logging/filters, not just the registry) — until
`verl_entry.py`'s TODO lands, adding a strategy means touching all three.

## What lives where

- `creative_rzero/rewards/registry.py` — `RewardStrategy` interface,
  `GroupScores`/`GroupReward`, `@register` / `get_strategy`.
- `creative_rzero/rewards/solver/` — solver strategies (`rank.py`,
  `raw.py` today).
- `creative_rzero/rewards/verl_entry.py` — verl's single
  `reward_function` entrypoint; dispatches by role, bridges the resolved
  experiment config into the env vars the legacy callers read.
- `examples/reward_function/creative_solver_caller.py` — solver
  orchestration: judge client, language/truncation filters, the
  `all_failed`/`low_quality` gate *definitions*, JSONL/W&B logging.
  Delegates only the reward *math* to the active strategy.
- `examples/reward_function/creative_writing_caller.py` — challenger
  reward (`uncertainty`); a monolith, not yet split into strategies.

## Adding a challenger strategy?

The registry only covers the solver role today. `creative_writing_caller.
py` doesn't call into `rewards/registry.py` at all — extending it means
the same shape of change (a `creative_rzero/rewards/challenger/` package,
a `RewardStrategy` per formula, the same three allow-lists) applied to the
challenger side. See `docs/REFACTOR_PLAN.md` T3.5 for the planned
strategies (`variance`, `target_band`) and rationale.
