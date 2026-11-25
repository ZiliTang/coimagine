# CAP C development notes

The supplied public stall prefixes showed two recurring patterns. The robots could move around the target without establishing contact, or they could report a push with little measured displacement. Some successful parent prefixes also started slowly, so C uses recent progress, robot movement and public execution stages before intervening.

| Revision | Joint failures rescued | Parent successes preserved | Metered ticks |
| --- | ---: | ---: | ---: |
| 1 | 2/16 | 6/8 | 38,444 |
| 2, selected | 2/16 | 6/8 | 37,689 |
| 3 | — | — | — |

Revision 1 ranks single-robot approach and push options using fitted faces, scan-derived clearance and observed failures. It stages one robot at a time and gives up on stale attempts. Revision 2 adds a shared directional cooldown when an observed contact produces little target and robot motion. It tightens the path grid, explores alternate stage points, and leaves developing approaches in place when the public feedback supports them. The two revisions rescued disjoint pairs on the shared panel. Their overall outcomes tied, so I selected revision 2 using the lower total tick count and its direct response to repeated same-direction stalls.

Revision 3 added a paired-push recovery and retained the frozen parent recovery as a fallback. Submission failed at the native 48 KiB source limit, before physical evaluation. Its probe slot was attempted once. The selected policy uses public observations and event history and has a native probe result. Its cooldown can still suppress a viable contact under pose reconstruction error or peer waiting. The 24 development outcomes are conditional on replayed prefixes. Standalone performance and held-out recovery remain unmeasured here.
