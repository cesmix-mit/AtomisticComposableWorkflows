# [WIP] Composable workflows

Composable workflows

<img aling="center" src="composable_workflows.png" alt="Composable workflows" width="68%"/>

Composable workflow 3

<img aling="center" src="workflow3.png" alt="Componsable workflow 3" width="68%"/>

Current examples:

| CW | Type  | Potential                               | MD                               | Location                                                                                                       |
|----|-------|-----------------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------|
|  1 | Na    |                                         | LAMMPS.jl                        | current repo                                                                                                   |
|  1 | Argon |                                         | LAMMPS.jl                        | current repo                                                                                                   |
|  1 | Argon | InteratomicPotentials.jl / LennardJones | Atomistic.jl / Molly.jl          | [link](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/molly_lj_simulation.jl)             |
|  1 | Argon | InteratomicPotentials.jl / LennardJones | Atomistic.jl / NBodySimulator.jl | [link](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/nbs_lj_simulation.jl)               |
|  2 | Argon | Atomistic.jl / DFTKPotential / DFTK.jl  | Atomistic.jl / Molly.jl          | [link](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/molly_dftk_ab_initio_simulation.jl) |
|  2 | Argon | Atomistic.jl / DFTKPotential / DFTK.jl  | Atomistic.jl / NBodySimulator.jl | [link](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/nbs_dftk_ab_initio_simulation.jl)   |
|  3 | Argon | InteratomicBasisPotentials.jl / ACE     | LAMMPS.jl                        | current repo                                                                                                   |
|  3 | HfO2  | InteratomicBasisPotentials.jl / ACE     | Atomistic.jl / Molly.jl          | current repo                                                                                                   |
|  3 | HfO2  | InteratomicBasisPotentials.jl / ACE     | Atomistic.jl / NBodySimulator.jl | current repo                                                                                                   |



Developed as part of [CESMIX](https://cesmix.mit.edu).
