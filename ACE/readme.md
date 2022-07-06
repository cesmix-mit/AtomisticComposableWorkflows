## [WIP] Fit different DFT datasets using ACE/NeuralACE

### Fit ACE/NeuralACE

```
$ julia fit-ace.jl  experiment_path         TiO2/ \
                    dataset_path            data/ \
                    trainingset_filename    TiO2trainingset.xyz \
                    testset_filename        TiO2testset.xyz \
                    n_train_sys             80 \
                    n_test_sys              20 \
                    n_batches               8 \
                    n_body                  3 \
                    max_deg                 3 \
                    r0                      1.0 \
                    rcutoff                 5.0 \
                    wL                      1.0 \
                    csp                     1.0 \
                    w_e                     1.0 \
                    w_f                     1.0
```

Analogous with ```fit-neural-ace.jl```

In addition, you can run the experiments with the default parameters (the parameters shown above).

```$ julia fit-ace.jl``` or ```$ julia fit-neural-ace.jl```


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


### Run an MD simulation using the wrapper to Molly.jl or NBodySimulator.jl in Atomistic.jl (TODO: update this)

```$ run-md-hfo2-ace-molly.jl``` or ```$ run-md-hfo2-ace-nbs.jl```

