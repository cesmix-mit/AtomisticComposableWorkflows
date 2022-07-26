var documenterSearchIndex = {"docs":
[{"location":"#[WIP]-Atomistic-Composable-Workflows","page":"Home","title":"[WIP] Atomistic Composable Workflows","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The Center for the Exascale Simulation of Materials in Extreme Environments (CESMIX) is a new MIT research effort to advance the state-of-the-art in predictive simulation. It seeks to connect quantum and molecular simulations of materials with state-of-the-art programming languages, compiler technologies, and software performance engineering tools, underpinned by rigorous approaches to statistical inference and uncertainty quantification.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This repository aims to gather easy-to-use CESMIX-aligned case studies, integrating the latest developments of the Julia atomistic ecosystem with state-of-the-art tools. This is a work in progress and is not ready for production, however some of our examples can already be used.","category":"page"},{"location":"#Atomistic-composable-workflows","page":"Home","title":"Atomistic composable workflows","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A series of composable workflows is guiding our design and development. We analyzed three of the most representative workflows: classical MD, Ab initio MD, and classical MD with active learning. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img src=\"composable_workflows.png\" alt=\"Composable workflows\" width=\"75%\" />","category":"page"},{"location":"","page":"Home","title":"Home","text":"CW1 describes  the  software  components  and  relationships  of  a  classical  MD  simulation. Essentially, at each time step, the force associated with each atom is calculated based on the interatomic potential, and then used to calculate each new atomic position.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The correct functioning of the calculators is analyzed and reported, via the curved arrow components to the dynamics model or control component.  E.g. a UQ analysis is performed on the force and MD calculations so that the control module can take corrective action.   In addition, a composable design must guarantee the communication of all the processes, for this purpose  \"wrapper\"  components,  represented  by  small circles in the figure, are included in the design. The wrappers are key in this design because they allow  heterogeneous  software  to  coexist  in  the  same workflow.  Each wrapper implements a set of interfaces associated  with a  particular  component. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"CW2 depicts an Ab initio MD process.  It is mostly analogous  to  the  workflow described above,  but  in this  case  the  force  calculation  is  provided  by  a  DFT simulation.\nCW3 presents a combination of the latter workflows.  Here, potentials/forces are fitted with respect  to  the  data  generated  by  the  DFT  simulator. The fitting process is complex and therefore requires a dedicated software component, as well as analysis of its inputs in terms of error, sensitivity, etc.  Furthermore,  the  dynamics model component,  based  on the analysis of the potential, forces and molecular dynamics,  can  re-fit these  forces  in  a  process called active learning.","category":"page"},{"location":"#Atomistic-suite-for-CESMIX-in-Julia","page":"Home","title":"Atomistic suite for CESMIX in Julia","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This composable approach allowed us to characterize each software component involved, which can be associated with one or more atomistic tools, as well as their interactions with other components. In particular, an increasing number of Julia packages dedicated to atomistic simulations are currently being developed. These packages combine the dynamic and interactive nature of Julia with its high-performance capabilities.","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img src=\"workflow3.png\" alt=\"Componsable workflow 3\" width=\"75%\"/>","category":"page"},{"location":"","page":"Home","title":"Home","text":"AtomsBase.jl is a lightweight abstract interface for representation of atomic geometries. It helps in the operability of diverse atomistic tools. Furthermore, AtomIO.jl is a standard IO package for atomic structures integrating with FileIO, AtomsBase, and others.\nDFTK.jl, the density-functional toolkit, is a library for playing with plane-wave density-functional theory (DFT) algorithms. In its basic formulation it solves periodic Kohn-Sham equations.\nInteratomicPotentials.jl and InteratomicBasisPotentials.jl are responsible for providing the methods to calculate the energies, forces and virial tensors of the potentials that we use in CESMIX.\nPotentialLearning.jl aims to facilitate the learning/fitting of interatomic potentials and forces, ensuring fast execution, leveraging state-of-the-art tools. The code of this tool will be refactored in the near future.\nAtomistic.jl provides an integrated workflow for MD simulations.\nLAMMPS.jl provides the bindings to the LAMMPS API, allowing other modules to access interatomic potentials, such as SNAP.","category":"page"},{"location":"#Case-studies","page":"Home","title":"Case studies","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Listed here is a subset of the case studies we are developing. We are gradually adding new cases as well as improving and increasing the complexity of the current ones.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CW Type DFT UQ Fitting/ Learning Potential / Forces Molecular Dynamics Location\n1 Ar  ✓ ✓ InteratomicPotentials.jl → Lennard Jones / ACE LAMMPS.jl → LAMMPS Current repo\n1 Ar  ✓ ✓ InteratomicPotentials.jl → LennardJones / ACE Atomistic.jl → Molly.jl Current repo\n1 Ar    InteratomicPotentials.jl → LennardJones Atomistic.jl → Molly.jl Atomistic.jl repo\n1 Na    LAMMPS.jl → EAM LAMMPS.jl  → LAMMPS Current repo\n3 Na DFTK.jl ✓ ✓ InteratomicPotentials.jl  → SNAP / ACE LAMMPS.jl  → LAMMPS Current repo\n3 HfO2 , a-HfO2, TiO2, etc. Multiple sources  ✓ InteratomicPotentials.jl → ACE Atomistic.jl → Molly.jl Current repo","category":"page"},{"location":"","page":"Home","title":"Home","text":"Atomistic.jl also provides abstractions for using NBodySimulator.jl, however we are currently focusing on Molly.jl, which provides more flexibility.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Examples of use can be found on the websites or github repositories of each tool mentioned.","category":"page"},{"location":"#Example:-Fit-different-DFT-datasets-using-ACE,-run-multiple-serial/parallel-fitting-experiments,-and-run-an-MD-simulation.","page":"Home","title":"Example: Fit different DFT datasets using ACE, run multiple serial/parallel fitting experiments, and run an MD simulation.","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In the folder ACE, you will find a basic integrated example that allows you to fit DFT datasets with ACE and run an MD simulation.","category":"page"},{"location":"#Chose-a-DFT-dataset","page":"Home","title":"Chose a DFT dataset","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Choose a DFT dataset. Currently, this code accepts either two xyz files, one for training and one for testing, or a single xyz file, which is automatically split into training and testing. Example datasets can be downloaded from the following urls.","category":"page"},{"location":"","page":"Home","title":"Home","text":"a-HfO2 dataset: \"Machine-learned interatomic potentials by active learning:","category":"page"},{"location":"","page":"Home","title":"Home","text":"amorphous and liquid hafnium dioxide\". Ganesh Sivaraman,  Anand Narayanan Krishnamoorthy, Matthias Baur, Christian Holm,  Marius Stan, Gábor Csányi, Chris Benmore & Álvaro Vázquez-Mayagoitia.  DOI: 10.1038/s41524-020-00367-7.  Dataset url","category":"page"},{"location":"","page":"Home","title":"Home","text":"FitSNAP: A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package. Datasets url\nCESMIX training data repository. Datasets url","category":"page"},{"location":"#Fit-ACE","page":"Home","title":"Fit ACE","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The input parameters are listed below:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Input parameter Description E.g.\nexperiment_path Experiment path TiO2/\ndataset_path Dataset path data/\ntrainingset_filename Training datasets filename TiO2trainingset.xyz\ntestset_filename Test datasets filename TiO2testset.xyz\nntrainsys No. of atomic configurations in training dataset 80\nntestsys No. of atomic configurations in test dataset 20\nn_body Body order 3\nmax_deg Maximum polynomial degree 3\nr0 An estimate on the nearest-neighbour distance for scaling 1.0\nrcutoff Outer cutoff radius 5.0\nwL See run-experiments.jl 1.0\ncsp See run-experiments.jl 1.0\nw_e Energy weight 1.0\nw_f Force weight 1.0","category":"page"},{"location":"","page":"Home","title":"Home","text":"Run fitting process","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ julia fit-ace.jl  experiment_path         TiO2/ \\\n                    dataset_path            data/ \\\n                    trainingset_filename    TiO2trainingset.xyz \\\n                    testset_filename        TiO2testset.xyz \\\n                    n_train_sys             80 \\\n                    n_test_sys              20 \\\n                    n_body                  3 \\\n                    max_deg                 3 \\\n                    r0                      1.0 \\\n                    rcutoff                 5.0 \\\n                    wL                      1.0 \\\n                    csp                     1.0 \\\n                    w_e                     1.0 \\\n                    w_f                     1.0","category":"page"},{"location":"","page":"Home","title":"Home","text":"In addition, you can run the experiments with the default parameters (the parameters shown above).","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ julia fit-ace.jl","category":"page"},{"location":"#Run-multiple-fitting-experiments-in-serial/parallel-using-the-wrapper-to-ACE1.jl-in-InteratomicBasisPotentials.jl","page":"Home","title":"Run multiple fitting experiments in serial/parallel using the wrapper to ACE1.jl in InteratomicBasisPotentials.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modify the file run-experiments.jl to specify the parameter ranges needed to generate the experiments. E.g.","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Parallel execution. Warning: a high number of parallel experiments may degrade system performance.\nparallel = true\n\n# n_body: body order. N: correlation order (N = n_body - 1)\nn_body = 2:5\n\n# max_deg: maximum polynomial degree\nmax_deg = 3:6","category":"page"},{"location":"","page":"Home","title":"Home","text":"Run the script:","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ julia run-experiments.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"Each experiment is run in a separate process (using nohup to facilitate its execution in a cluster). The results are stored in the folder experiments/. After all experiments have been completed, run the following script to gather the results into a single csv.","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ ./gather-results.sh","category":"page"},{"location":"#Run-an-MD-simulation-using-the-wrapper-to-Molly.jl-or-NBodySimulator.jl-in-Atomistic.jl","page":"Home","title":"Run an MD simulation using the wrapper to Molly.jl or NBodySimulator.jl in Atomistic.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"$ run-md-ahfo2-ace-nbs.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"or","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ run-md-ahfo2-ace-molly.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Note: currently there is a bug in the second script) ","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#Install-Julia-on-Ubuntu","page":"Home","title":"Install Julia on Ubuntu","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Open terminal and download Julia from https://julialang.org/downloads/  bash  $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz\nExtract file  bash  $ tar xvzf julia-1.7.3-linux-x86_64.tar.gz\nCopy to /opt and create link  bash  $ sudo mv  ./julia-1.7.3 /opt/  $ sudo ln -s /opt/julia-1.7.3/bin/julia /usr/local/bin/julia\nAlternative: add line to .bashrc  bash  $ nano .bashrc  PATH=$PATH:/home/youruser/julia-1.7.3 /bin/\nRestart the terminal","category":"page"},{"location":"#Add-registries-and-install-dependencies","page":"Home","title":"Add registries and install dependencies","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Open a Julia REPL  bash  $ julia\nAdd registries: General, CESMIX, and MolSim  bash  pkg> registry add https://github.com/JuliaRegistries/General  pkg> registry add https://github.com/cesmix-mit/CESMIX.git   pkg> registry add https://github.com/JuliaMolSim/MolSim.git\nInstall general packages your workflow is likely to require. E.g.  bash  pkg> add LinearAlgebra  pkg> add StaticArrays  pkg> add UnitfulAtomic  pkg> add Unitful  pkg> add Flux  pkg> add Optimization  pkg> add OptimizationOptimJL  pkg> add BenchmarkTools  pkg> add Plots\nInstall CESMIX packages  bash  pkg> add AtomsBase  pkg> add InteratomicPotentials  pkg> add InteratomicBasisPotentials  pkg> add https://github.com/cesmix-mit/PotentialLearning.jl  pkg> add Atomistic\nInstall other important dependencies\nMD simulators\npkg> add Molly  pkg> add NBodySimulator\nACE (see: https://acesuit.github.io/ACE.jl/dev/gettingstarted/#Installation)\nbash  pkg> add PyCall IJulia  pkg> add ACE  pkg> add JuLIP ASE ACEatoms  pkg> add IPFitting","category":"page"}]
}
