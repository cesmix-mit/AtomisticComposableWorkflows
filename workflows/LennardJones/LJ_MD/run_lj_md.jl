using LAMMPS


include("define_md_LAMMPS_lj.jl")

Tend = Int(0.5E6)
dT = 50
Temp = 0.65*120
for seed = 1:1
    save_dir = "LJ_MD/TEMP/"
    run_md(Tend, save_dir; seed = seed, dT = dT, dt = 0.005, Temp = Temp)
    run(`python3 LJ_MD/to_extxyz.py LJ_MD/TEMP/`)
end
