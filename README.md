# [WIP] Atomistic Composable Workflows

The Center for the Exascale Simulation of Materials in Extreme Environments ([CESMIX](https://computing.mit.edu/cesmix/)) is a new MIT research effort to advance the state-of-the-art in predictive simulation. It seeks to connect quantum and molecular simulations of materials with state-of-the-art programming languages, compiler technologies, and software performance engineering tools, underpinned by rigorous approaches to statistical inference and uncertainty quantification.

This repository aims to gather case studies of interest to CESMIX implemented using the latest developments of the growing Julia atomistic ecosystem, integrated with other state-of-the-art tools. This is a work in progress and is not yet ready for production, however some of our examples can already be used.

## Atomistic composable workflows

A series of composable workflows is guiding our design and development. We analyzed three of the most representative workflows: classical MD, Ab initio MD, and classical MD with active learning. 

<img src="composable_workflows.png" alt="Composable workflows" width="75%" />

- CW1 describes  the  software  components  and  relationships  of  a  classical  MD  simulation. Essentially, at each time step, the force associated with each atom is calculated based on the interatomic potential, and then used to calculate each new atomic position.
The correct functioning of the calculators is analyzed and reported, via the curved arrow components to the dynamics model or control component.  E.g. a UQ analysis is performed on the force and MD calculations so that the control module can take corrective action.  
In addition, a composable design must guarantee the communication of all the processes, for this purpose  "wrapper"  components,  represented  by  small circles in the figure, are included in the design. The wrappers are key in this design because they allow  heterogeneous  software  to  coexist  in  the  same workflow.  Each wrapper implements a set of interfaces associated  with a  particular  component. 
- CW2 depicts an Ab initio MD process.  It is mostly analogous  to  the  workflow described above,  but  in this  case  the  force  calculation  is  provided  by  a  DFT simulation.
- CW3 presents a combination of the latter workflows.  Here, potentials/forces are fitted with respect  to  the  data  generated  by  the  DFT  simulator. The fitting process is complex and therefore requires a dedicated software component, as well as analysis of its inputs in terms of error, sensitivity, etc.  Furthermore,  the  dynamics model component,  based  on the analysis of the potential, forces and molecular dynamics,  can  re-fit these  forces  in  a  process called active learning.

## Atomistic suite for CESMIX in Julia

This composable approach allowed us to characterize each software component involved, which can be associated with one or more atomistic tools, as well as their interactions with other components. In particular, an increasing number of Julia packages dedicated to atomistic simulations are currently being developed. These packages combine the dynamic and interactive nature of Julia with its high-performance capabilities.

<img src="workflow3.png" alt="Componsable workflow 3" width="75%"/>


- [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) is a lightweight abstract interface for representation of atomic geometries. It helps in the operability of diverse atomistic tools. Furthermore, [AtomIO.jl](https://github.com/mfherbst/AtomIO.jl) is a standard IO package for atomic structures integrating with FileIO, AtomsBase, and others.
- [DFTK.jl](https://docs.dftk.org/stable/), the density-functional toolkit, is a library for playing with plane-wave density-functional theory (DFT) algorithms. In its basic formulation it solves periodic Kohn-Sham equations.
- [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl) and [InteratomicBasisPotentials.jl](https://github.com/cesmix-mit/InteratomicBasisPotentials.jl) are responsible for providing the methods to calculate the energies, forces and virial tensors of the potentials that we use in CESMIX.
- [PotentialLearning.jl](https://github.com/cesmix-mit/PotentialLearning.jl) aims to facilitate the learning/fitting of interatomic potentials and forces, ensuring fast execution, leveraging state-of-the-art tools. The code of this tool will be refactored in the near future.
- [Atomistic.jl](https://github.com/cesmix-mit/Atomistic.jl) provides an integrated workflow for MD simulations.
- [LAMMPS.jl](https://github.com/cesmix-mit/LAMMPS.jl) provides the bindings to the LAMMPS API, allowing other modules to access interatomic potentials, such as SNAP.



## Case studies

Listed here is a subset of the case studies we are developing. We are gradually adding new cases as well as improving and increasing the complexity of the current ones.

| CW |            Type           |        DFT       | UQ | Fitting/ Learning |               Potential / Forces               |    Molecular Dynamics   | Location          |
|:--:|:-------------------------:|:----------------:|:--:|:-----------------:|:----------------------------------------------:|:-----------------------:|-------------------|
| 1  | Ar                        |                  |  ✓ |         ✓         | InteratomicPotentials.jl → Lennard Jones / ACE | LAMMPS.jl → LAMMPS      | Current repo      |
| 1  | Ar                        |                  |  ✓ |         ✓         | InteratomicPotentials.jl → LennardJones / ACE  | Atomistic.jl → Molly.jl | Current repo      |
| 1  | Ar                        |                  |    |                   | InteratomicPotentials.jl → LennardJones        | Atomistic.jl → Molly.jl | [Atomistic.jl repo](https://github.com/cesmix-mit/Atomistic.jl/blob/main/examples/argon/molly_lj_simulation.jl) |
| 1  | Na                        |                  |    |                   | LAMMPS.jl → EAM                                | LAMMPS.jl  → LAMMPS     | Current repo      |
| 3  | Na                        | DFTK.jl          |  ✓ |         ✓         | InteratomicPotentials.jl  → SNAP / ACE         | LAMMPS.jl  → LAMMPS     | Current repo      |
| 3  | HfO2 , a-HfO2, TiO2, etc. | Multiple sources |    |         ✓         | InteratomicPotentials.jl → ACE                 | Atomistic.jl → Molly.jl | Current repo      |

Atomistic.jl also provides abstractions for using NBodySimulator.jl, however we are currently focusing on Molly.jl, which provides more flexibility.

Examples of use can be found on the websites or github repositories of each tool mentioned. If you want to start with a basic integrated example that allows you to fit DFT data with ACE and run an MD simulation you can follow this [link](https://github.com/cesmix-mit/AtomisticComposableWorkflows/tree/master/ACE). 



## Installation

#### Install Julia on Ubuntu

1.  Open terminal and download Julia from https://julialang.org/downloads/
    ```bash
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz
    ```
2.  Extract file
    ```bash
    $ tar xvzf julia-1.7.3-linux-x86_64.tar.gz
    ```
3. Copy to `/opt` and create link
    ```bash
    $ sudo mv  ./julia-1.7.3 /opt/
    $ sudo ln -s /opt/julia-1.7.3/bin/julia /usr/local/bin/julia
    ```
4. Alternative: add line to `.bashrc`
    ```bash
    $ nano .bashrc
    PATH=$PATH:/home/youruser/julia-1.7.3 /bin/
    ```
5. Restart the terminal

#### Add registries and install dependencies

1. Open a Julia REPL
    ```bash
    $ julia
    ```
2. Add General registry
    ```bash
    pkg>  registry add https://github.com/JuliaRegistries/General
    ```
3. Add CESMIX registry
    ```bash
    pkg>  registry add https://github.com/cesmix-mit/CESMIX.git 
    ```
4. Add MolSim registry
    ```bash
    pkg> registry add https://github.com/JuliaMolSim/MolSim.git
    ```
5. Install common packages. E.g.
    ```bash
    pkg> add LinearAlgebra
    pkg> add Random
    pkg> add StaticArrays
    pkg> add Statistics
    pkg> add StatsBase
    pkg> add Flux
    pkg> add BSON
    pkg> add CUDA
    pkg> add Zygote
    pkg> add UnitfulAtomic
    pkg> add Unitful
    pkg> add BenchmarkTools
    pkg> add Plots
    ```
6. Install CESMIX packages
    ```bash
    pkg> add AtomsBase
    pkg> add InteratomicPotentials
    pkg> add InteratomicBasisPotentials
    pkg> add Atomistic
    ```
7. Install ACE (see: https://acesuit.github.io/ACE.jl/dev/gettingstarted/#Installation)
    ```bash
    pkg> add PyCall IJulia
    pkg> add ACE
    pkg> add JuLIP ASE ACEatoms
    pkg> add IPFitting
    ```
