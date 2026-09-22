# Changelog

## v0.9.1 - Live experiment progress and ETA

- added a dependency-free terminal progress reporter with case counts, generated
  episode counts, throughput, elapsed time, and ETA;
- enabled progress by default for both controlled and heterogeneous randomized
  suites, including multiprocessing runs via completion-order futures;
- added phase-level messages for aggregation, table export, figure rendering, and
  final output location;
- added `--no-progress` and `--progress-interval` command-line controls;
- retained deterministic result ordering after asynchronous worker completion.

## v0.9.0 - Heterogeneous randomized channel generalization suite

- added a separate unscreened multi-blocker channel generator with randomized map
  dimensions, robot poses, sensing radii, blocker counts, coalition requirements,
  anchor banks, sparse occluders, activation chains, candidate budgets, and
  episode budgets;
- retained the controlled causal families for mechanism isolation rather than
  conflating them with general-task statistics;
- added a 12-scene debug gallery showing initial observations, rays, anchors,
  robot layouts, walls, blocker counts, and per-scene budgets;
- added completion, partial task-potential, and planning-time summaries by easy,
  moderate, and hard scene strata;
- added episode-level diagnostic fields for heavy objects, occluders, candidate
  budget, episode budget, future-support requirement, and critical entity;
- added an active-action tie break to prevent receding-horizon policies from
  repeatedly selecting Idle when delaying a value-equivalent plan;
- added `main` and `full` randomized profiles, a fixed execution-budget cap, and
  `--skip-controlled` for practical local evaluation;
- all 32 tests pass.

## v0.8.0 - Family-stratified statistics and paper-scale randomized testing

- added an explicit family-stratified generator API so every open causal family
  receives an independent sample count rather than being coupled to a seed cycle;
- added a default 400-instance open benchmark (100 per family) and a separate
  100-instance unseen-wall surprise split;
- expanded initial-pose, setup-distance, map-size, translation, and reflection
  randomization without outcome screening;
- added multiprocessing support and a quick smoke profile;
- replaced Wald error bars with Wilson 95% intervals and added median/IQR
  summaries for timing, return, duration, and option count;
- added paired bootstrap intervals and exact McNemar tests on identical instances;
- replaced the binary dependency heatmap with confidence-interval curves and
  added return/timing violin plots plus paired-comparison forest plots;
- extended robot scaling to bounded factor arities P=2,3,4;
- exported complete episode, support, paired, aggregate, and scaling tables.

## v0.7.0 - Mixed randomized quantitative benchmark and scaling figures

- replaced the single fixed-dependency horizon distribution with four balanced,
  unscreened causal families: direct Push, Approach-Push, mixed setup-Push, and
  Reposition-Approach-Push;
- randomized map translation, reflection, and robot initial positions while
  preserving each family's information pattern;
- made CST/H=2 a nontrivial baseline that completes all current-support families,
  while IET/H=3 uniquely completes the future-support dependency;
- added separate completion and planning-time panels to remove twin-axis overlap;
- added a dependency-class heatmap, local-imagination fidelity/optimism diagnostics,
  and publication-ready object/robot structural scaling curves;
- exported episode-level CSV/JSON data plus horizon, dependency, support, and
  structural-scaling tables in CSV/JSON/Markdown formats.

## v0.6.1 - Compact publication figure and CST naming

- reduced the channel-story aspect ratio to approximately 5:3 and tightened row spacing;
- thinned and lengthened Reposition, Push, and outward channel-direction arrows;
- removed redundant white robot IDs and retained offset black robot labels below each robot;
- moved blocker IDs to the grid cells above the objects for improved legibility;
- changed the displayed abbreviation from COST to CST while retaining the full name
  Current-Observation-Supported Topology and the internal planner-mode identifier.

## v0.6.0 - Multi-blocker channel story and explicit factor topology

- added a four-blocker horizontal channel-clearing showcase with lateral wall gaps;
- added a two-row, five-boundary CST/IET main figure that shows shared progress
  before the critical heavy-gate divergence;
- visualized occlusion-terminated sensing rays, visible-region overlays, sparse
  reposition anchors, dashed channel boundaries, and outward clearing arrows;
- added compact robot-owned option strips grouped by navigation/entity factors,
  including new-option highlighting and the coordinated heavy-Push edge;
- retained the v0.5 quantitative horizon and optimism studies unchanged.

## v0.5.0 - Local-observation CST/IET Gate A

- separated privileged simulator truth from robot-owned local beliefs;
- added occlusion-aware dense ray observations and optimistic analytic local-belief imagination;
- added Reposition anchors and observation-grounded option templates;
- introduced CST, CoImagine-IET, and the privileged successor-observation oracle;
- added a three-boundary Reposition–Approach–Push benchmark and unscreened open/surprise variants;
- evaluated H=2 and H=3 under matched receding-horizon execution;
- added future-option/topology precision, recall, oracle agreement, and planning-time outputs;
- replaced the main Gate-A storyboard with a local-FoV CST-versus-IET sequence;
- retained prior topology-event, factorization, and scaling diagnostics as supplementary checks.


## v0.4.0 - TRO-oriented integrated Gate A

- replaced the main gallery of isolated mechanism tasks with one integrated
  three-robot, mixed-mass channel-clearing episode;
- added one shared universal option-template library used across all task suites;
- added weighted multi-blocker clearing, coalition requirements, direction
  constraints, and dependency-gated option activation;
- added exact receding-horizon episode rollouts for Evolving H=2,
  Root-Structure Reuse H=2, Evolving H=1, and symbolic local-belief H=2;
- added naturalized 3-5 blocker variants without post-selection by method gain;
- added a self-explaining double-column action-sequence storyboard showing the
  first decision, created coalition edge, delayed option activation, and final
  task difference;
- added one compact decision/scaling figure and one small main-result table;
- moved event taxonomy, hold-pass, reveal-act, and higher-order factor
  diagnostics to supplementary output;
- added clean, review-with-definitions, and independent-panel PDF/SVG/PNG files;
- added integrated-pipeline semantic tests.

## v0.3.0 - Pilot-ready diagnostic Gate A

- separated exact-state structural evidence from symbolic local-belief evidence;
- implemented natural and controlled diagnostic suites with explicit mechanism
  acceptance rules;
- replaced ambiguous fixed-factor logic with root option-template projection and
  root joint-template-edge reuse;
- added exclusive realized topology events and strict higher-order diagnostics;
- added review composites with nonoverlapping terminology regions.
