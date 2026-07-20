# CoImagine

**CoImagine** is a research codebase for world-model-based planning and embodied interaction in multi-robot systems. The project studies how robots can coordinate through local observations, structured action intents, communication, and imagined future interactions without relying on persistent global poses or shared metric maps.

The current paper developed within this project is:

> **CoLaM: Coordinate-Free Latent World-Model Rollout for Multi-Robot Planar Manipulation**

**CoImagine** is the long-term project and repository name. **CoLaM** is the method name and paper acronym used for the current study.

## Current research scope

The current implementation targets coordinate-free collaborative planar manipulation with multiple mobile robots. At each observation-complete option boundary, every robot:

1. encodes RGB-D, semantic masks, optional 3-D lidar, proprioception, and recent action responses into an egocentric belief;
2. enumerates a compact bank of push, reposition, and idle options;
3. communicates contextualized option intents with its teammates;
4. predicts future local beliefs with an option-level latent world model;
5. regenerates and renegotiates candidate options at each imagined depth; and
6. executes only the first selected joint option before observing the real scene again.

A push option specifies a desired planar-effect direction and an extent class. A reposition option specifies a target-centered behavior such as orbiting or backing off. Neither option contains an explicit contact point or pre-contact pose. Geometry-dominant recurrent policies use depth, lidar, target masks, and proprioception to acquire and maintain suitable contacts implicitly. Multi-robot pushes are synchronized through a prepare–ready–commit protocol.

## Main components

The planned codebase contains the following functional components:

- **Multimodal perception:** RGB-D semantic segmentation, target and task-entity masks, lidar preprocessing, and robot-centric observations.
- **Egocentric belief encoding:** spatial latent features, task and interaction memory, and response-conditioned dynamics context.
- **Structured options:** desired-effect push tokens, target-centered reposition tokens, idle actions, feasibility estimates, and behavior-support filtering.
- **Embodied skills:** recurrent push and reposition policies using geometry-dominant observations.
- **Communication:** finite-round exchange of compact belief summaries and contextualized option intents.
- **World model:** prediction of successor latents, robot-centric occupancy, option responses, task potential, execution risk, and duration.
- **Planning:** branch-wise communicative beam search with future candidate regeneration and renegotiation.
- **Evaluation:** matched Direct-Q baselines, one-step and multi-step rollouts, oracle look-ahead analysis, skill validation, and hardware tests.

## Repository organization

The implementation is expected to follow a modular structure similar to:

```text
coimagine/
├── configs/           # Training, planning, sensing, and evaluation configurations
├── scripts/           # Data collection, training, evaluation, and deployment entry points
├── src/coimagine/
│   ├── perception/    # RGB-D segmentation and lidar/depth preprocessing
│   ├── belief/        # Egocentric encoders, memory, and dynamics context
│   ├── skills/        # Push and reposition policies
│   ├── options/       # Option templates, feasibility, and candidate ranking
│   ├── communication/ # Intent encoding and finite-round coordination
│   ├── world_model/   # Latent transition and prediction heads
│   ├── planning/      # Communicative rollout and beam search
│   └── evaluation/    # Metrics, oracle analysis, and diagnostics
├── tests/
└── README.md
```

The exact structure may evolve as the simulator and hardware pipelines are integrated.

## Companion manuscript repository

The LaTeX manuscript, figures, research drafts, and paper-specific material are maintained separately in:

```text
coimagine-tex
```

This repository, `coimagine`, is reserved for implementation, experiment configuration, trained-model interfaces, evaluation tools, and reproducible deployment code.

## Development status

The project is under active development. The initial implementation sequence is:

1. train and validate the geometry-conditioned push and reposition skills;
2. validate synchronized prepare–ready–commit execution;
3. collect option-boundary transition data;
4. train the real-boundary belief encoder and one-step world model;
5. verify oracle two-step advantages on delayed-benefit tasks;
6. train and evaluate communicative rollout with horizon \(H=2\); and
7. integrate the complete pipeline on two quadruped robots.

Installation instructions, simulator versions, pretrained checkpoints, datasets, and reproducible experiment commands will be added as the corresponding components stabilize.

## Working conventions

- Use **CoImagine** for the long-term project, repository, and software framework.
- Use **CoLaM** only for the current paper method and paper-specific configurations.
- Keep paper-specific files in `coimagine-tex`; do not duplicate the complete manuscript in this repository.
- Prefer task-independent module names such as `world_model`, `skills`, and `planning` over names tied to a paper acronym.
- Store CoLaM-specific experiment settings under paths such as `configs/colam/` or `experiments/colam/`.
- Do not commit large datasets, simulator caches, checkpoints, rendered videos, or generated training artifacts.
- Record final environments, dependencies, random seeds, and configuration files required to reproduce reported results.

## Citation

The paper is currently under development. Citation information will be added after publication.

## License and release

This is currently a private research repository under active development. The software license, pretrained models, datasets, and public release policy will be specified when the implementation and experimental pipeline are ready for release.
