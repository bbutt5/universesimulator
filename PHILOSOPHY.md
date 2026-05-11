# Philosophy: Emergence over hardcoding

This simulator aims to be as close to a true universe replica as compute
allows. When two implementations satisfy the same requirement, **always
pick the one that derives behaviour from more fundamental physics**, not
the one that encodes the outcome.

The goal is a sim of *the rules*, not a sim of *the consequences*.

## The four lenses

Apply these to every new mechanic you build:

1. **Could this property emerge from something more fundamental we already have?**
2. **If you're adding a table or lookup, can the values be computed from a formula?**
3. **If you're adding a boolean flag, can the behaviour emerge from a continuous threshold?**
4. **Are you encoding the *outcome* of physics, or the *physics itself*?**

## Concrete preferences

| Avoid                                      | Prefer                                                                 |
| ------------------------------------------ | ---------------------------------------------------------------------- |
| `can_fuse: True/False` flag                | Coulomb-barrier vs strong-force computation from Z and A               |
| Hand-tuned `de_relative` per element       | Pauling electronegativity formula for bond energy                      |
| Fixed `ionization_ke_threshold`            | Real first-ionization-energy data per element                          |
| `FUSION_REACTIONS` product lookup          | Binding-energy formula (semi-empirical mass formula → max-Q product)   |
| Phenomenological `radiation_kick = k/r`    | Explicit photon particles (if affordable)                              |
| Hardcoded "this is a star" flag            | Mass-ratio-based emergent classification (already done)                |
| Hardcoded "this is solid/liquid/gas/plasma"| Density + local-KE + bond + ionization classifier (already done)       |

## Things that are okay to hardcode

These are not "rules", they are either inputs or measurements:

- **Real measured physical data** — atomic masses, electronegativities,
  binding energies, ionization energies. These come from the lab.
  Always include the source / units in a comment.
- **Visualisation parameters** — colour ramps, point sizes, FOV. These
  affect what you see, not what happens.
- **The base laws we operate at** — Newtonian gravity, Coulomb, classical
  mechanics. We're not doing QFT; that's an acknowledged scale choice.
- **Initial conditions** — Hubble velocity, primordial seeds, injection
  rate. These are model inputs, not rules of physics.

## When fidelity costs too much

If a more-emergent approach would crash performance below interactivity or
destabilise the integrator, leave a comment naming the trade-off and
open / link an issue describing how to revisit it with more compute. Do
not silently revert to the easier model.

Example comment style:

```python
# Phenomenological radiation kick (issue #N): replaces explicit photon
# transport. Acceptable for now because photon count would be >1e6 per
# fusion event; full transport would dominate the step budget.
```

## Currently hardcoded — known debt

Tracked in [the de-hardcoding pass issue](https://github.com/billymahmood/universesimulator/issues)
labelled `dehardcoding`. New code that adds a constant, table, or flag
must justify why an emergent alternative is infeasible, or open an issue
describing how to remove it later. PR reviews should check this.
