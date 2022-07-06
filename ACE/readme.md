## [WIP] Fit different DFT data sets using ACE/NeuralACE, run multiple serial/parallel fitting experiments, and run an MD simulation.


### Chose a DFT dataset

Choose a DFT dataset. Currently, this code accepts two `xyz` format files, one for training and one for testing. Examples can be obtained from the following urls.

- a-HfO2 dataset: "Machine-learned interatomic potentials by active learning:
 amorphous and liquid hafnium dioxide". Ganesh Sivaraman,
 Anand Narayanan Krishnamoorthy, Matthias Baur, Christian Holm,
 Marius Stan, Gábor Csányi, Chris Benmore & Álvaro Vázquez-Mayagoitia.
 DOI: 10.1038/s41524-020-00367-7.
 [Dataset url](https://github.com/argonne-lcf/active-learning-md/tree/master/data)
- FitSNAP: A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package. [Datasets url](https://github.com/FitSNAP/FitSNAP/tree/master/examples)
- CESMIX training data repository. [Datasets url](https://github.com/cesmix-mit/TrainingData)


### Fit ACE/NeuralACE

The input parameters are listed below:

| Input parameter      | Description                                               | E.g.                |
|----------------------|-----------------------------------------------------------|---------------------|
| experiment_path      | Experiment path                                           | TiO2/               |
| dataset_path         | Dataset path                                              | data/               |
| trainingset_filename | Training datasets filename                                | TiO2trainingset.xyz |
| testset_filename     | Test datasets filename                                    | TiO2testset.xyz     |
| n_train_sys          | No. of atomic configurations in training dataset          | 80                  |
| n_test_sys           | No. of atomic configurations in test dataset              | 20                  |
| n_batches            | No. of batches per dataset                                | 8                   |
| n_body               | Body order                                                | 3                   |
| max_deg              | Maximum polynomial degree                                 | 3                   |
| r0                   | An estimate on the nearest-neighbour distance for scaling | 1.0                 |
| rcutoff              | Outer cutoff radius                                       | 5.0                 |
| wL                   | See run-experiments.jl                                    | 1.0                 |
| csp                  | See run-experiments.jl                                    | 1.0                 |
| w_e                  | Energy weight                                             | 1.0                 |
| w_f                  | Force weight                                              | 1.0                 |

Run fitting process

```
$ julia fit-ace.jl  experiment_path         TiO2/ \
                    dataset_path            data/ \
                    trainingset_filename    TiO2trainingset.xyz \
                    testset_filename        TiO2testset.xyz \
                    n_train_sys             80 \
                    n_test_sys              20 \
                    n_body                  3 \
                    max_deg                 3 \
                    r0                      1.0 \
                    rcutoff                 5.0 \
                    wL                      1.0 \
                    csp                     1.0 \
                    w_e                     1.0 \
                    w_f                     1.0
```

To fit with `fit-neural-ace.jl` the process is analogous. In this case the number of batches must also be defined.

In addition, you can run the experiments with the default parameters (the parameters shown above).

```bash
$ julia fit-ace.jl
```
or
```bash
$ julia fit-neural-ace.jl
```


### Run multiple fitting experiments in serial/parallel using ACE/NeuralACE

Modify the file `run-experiments.jl` to specify the parameter ranges needed to generate the experiments. E.g.
```julia

# Parallel execution. Warning: a high number of parallel experiments may degrade system performance.
parallel = true

# n_body: body order. N: correlation order (N = n_body - 1)
n_body = 2:5

# max_deg: maximum polynomial degree
max_deg = 3:6
```

Run the script:

```bash
$ julia run-experiments.jl
```

Each experiment is run in a separate process (using `nohup` to facilitate its execution in a cluster).
The results are stored in the folder `experiments/`.
After all experiments have been completed, run the following script to gather the results into a single csv.

```shell
$ ./gather-results.sh
```

### Run an MD simulation using the wrapper to Molly.jl or NBodySimulator.jl in Atomistic.jl

```bash
$ run-md-hfo2-ace-nbs.jl
```
or
```bash
$ run-md-hfo2-ace-molly.jl
```
(Note: currently there is a bug in the second script) 


