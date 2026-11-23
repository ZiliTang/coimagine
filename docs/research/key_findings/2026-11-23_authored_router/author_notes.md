# AB router author notes

The selected router is v3, SHA-256 `7a294a18caa44889e1eafad9ba1d1df88208833bcd24649d640cfac71a787942`. It keeps all state in episode memory and uses only validated action codes, duration, and the public tick, gap, contact, progress, and trigger.

Across 176 development rooms, A, B, the one-way handoff, and the random-switch control succeeded in 99, 98, 98, and 100 rooms. Among the 160 rooms with a trigger, A won 21 rooms that the one-way handoff lost, and the handoff won 20 that A lost. In the 33 rooms where the first post-trigger disagreement kept the pair and B requested fewer ticks, A won 24 and the handoff won 27. In the 13 rooms where the first disagreement changed the pair and the trigger had far contact at node 44, A won 3 and the handoff won 8.

The router accepts B's shorter duration on a matching pair. It leaves that decision separate from the persistent recovery choice. The v1 probe showed that the public gap can remain above the threshold for B's internal precision estimate, so v2 and v3 use the visible action difference directly. v3 starts B when far contact already exists at the trigger. It gives A more setup time during a far approach without contact, permits earlier B for clearance and short-range staging, and hands back after a sustained stall. The action guide motivated preserving productive PUSH and allowing setup time for contact or maneuvering.

| Revision | Success | Executed ticks |
| --- | ---: | ---: |
| v1 | 12/24 | 33,757 |
| v2 | 15/24 | 30,729 |
| v3 | 18/24 | 28,609 |

On the same 24 eligible rooms, A won 14, B and the one-way handoff each won 10, and the random-switch control won 13. v3 gained five rooms and lost one relative to A, and gained eight with zero losses relative to the one-way handoff. Relative to v2 it preserved all 15 wins and gained three. Two gains recovered A successes where early B had interrupted a far approach. The third came from a room where A, B, and the one-way handoff all failed. The fixed probe panel informed tuning and selection, so its result is development evidence. All 72 authorized probe episodes were used.
