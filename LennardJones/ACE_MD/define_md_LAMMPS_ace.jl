## This function runs an MD trajectory using the ACE potential with potential file saved at 
## file_dir\parameters.ace, using LAMMPS.jl see LAMMPS documentation for more information about 
## particular commands

using LAMMPS
function run_md(Tend::Int, file_dir::String, save_dir::String; seed = 1, Temp = 0.5, dt = 0.001, dT = 10)
    try
        mkdir(save_dir_seed)
    catch

    end
    d = LMP(["-screen", "none"]) do lmp
        for i = 1
            # command(lmp, "log none")
            command(lmp, "log $(save_dir)log.lammps")
            command(lmp, "units metal")
            command(lmp, "dimension 3")
            command(lmp, "atom_style atomic")
            command(lmp, "atom_modify map array")
            command(lmp, "boundary f f f")

            # Setup box
            command(lmp, "region mybox block -5 5 -5 5 -5 5")
            command(lmp, "read_data starting_configuration.lj")

            command(lmp, "mass 1 39.948")
            command(lmp, "velocity all create $(Temp) $seed mom yes rot yes dist gaussian")
            # Setup Forcefield
            command(lmp, "pair_style pace")
            command(lmp, "pair_coeff * * $(file_dir)parameters.ace Ar")
            
            # computes
            command(lmp, "compute S all pressure NULL virial") # Stress tensor without kinetic energy component (only the virial)
            command(lmp, "fix vir all ave/time 1 $dT $dT c_S[1] c_S[2] c_S[3] c_S[6] c_S[5] c_S[4] file $(save_dir)tmp.virial")
            command(lmp, "compute rdf all rdf 100")      # radial distribution function 
            command(lmp, "fix frdf all ave/time $dT $(Int(Tend/dT)-1) $Tend c_rdf[*] file $(save_dir)tmp.rdf mode vector")

            command(lmp, "compute pe all pe")                       # potential energy
            command(lmp, "fix fpe all ave/time 1 $dT $dT c_pe file $(save_dir)tmp.pe")
            
            command(lmp, "compute msd all msd com yes")
            command(lmp, "fix fmsd all ave/time 1 $dT $dT c_msd[4] file $(save_dir)tmp.msd")
            
            command(lmp, "thermo $dT")
            command(lmp, "fix l0 all nve langevin $Temp $Temp 1 $seed")

            command(lmp, "timestep $dt")
            command(lmp, "dump 1 all xyz $(dT) $(save_dir)dump.xyz")
            command(lmp, "dump 2 all custom $(dT) $(save_dir)dump.positions_and_forces type id x y z fx fy fz")
            command(lmp, "run $Tend")
            command(lmp, "unfix l0")
            command(lmp, "unfix frdf")
            command(lmp, "unfix fpe")
            command(lmp, "unfix fmsd")
            command(lmp, "undump 1")
            command(lmp, "undump 2")
            command(lmp, "clear")
        end
    end

end