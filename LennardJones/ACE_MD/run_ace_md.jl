using LAMMPS
include("define_md_LAMMPS.jl")


## Run MD for different coefficients and different seeds
Tend = Int(0.5E6)
dT = 50
Temp = 0.65*120
for coeff = 1:20
    println("coeff $coeff")
    save_dir = "ACE_MD/coeff_$coeff/"
    # for seed = 1:100
        run_md(Tend, save_dir; seed = 1, dT = dT, dt = 0.005, Temp = Temp)
        run(`python3 ACE_MD/to_extxyz.py ACE_MD/coeff_$(coeff)/1/`)
    # end
end
