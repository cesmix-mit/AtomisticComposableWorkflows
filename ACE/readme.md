# [WIP] Running experiments about a-HfO2/HfO2, ACE/ACE+NN, and MD.

#### Fit a-HfO2/HfO2 datasets using ACE

The file `run-experiments.jl` specifies the parameters of each experiment. The results are stored in the `experiments/` folder.

```$ julia run-experiments.jl```

Each experiment is run in a separate process (using `nohup` to facilitate its execution in a cluster).
After all experiments have been completed, run the following command to gather the results into a single csv.

```$ ./gather-results.sh```

#### Fit the a-HfO2 dataset using ACE or ACE+NN and a specific set of parameters

Input parameters: `experiment_path dataset_path dataset_file n_body max_deg r0 rcutoff wL csp`

```$ julia fit-hfo2-ace.jl fit-ahfo2-ace-nn/ data/ a-Hfo2-300K-NVT.extxyz 1000 2 3 1 5 1 1```

```$ julia fit-hfo2-ace-nn.jl fit-ahfo2-ace-nn/ data/ a-Hfo2-300K-NVT.extxyz 1000 2 3 1 5 1 1```

In addition, you can run the experiments with the default parameters (the parameters shown above).

```$ julia fit-hfo2-ace.jl``` or ```$ julia fit-hfo2-ace-nn.jl```


### Run an MD simulation using the wrapper to Molly.jl or NBodySimulator.jl in Atomistic.jl

```$ run-md-hfo2-ace-molly.jl``` or ```$ run-md-hfo2-ace-nbs.jl```

