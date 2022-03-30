using LAMMPS

function run_md(Tend::Int, save_dir::String; seed = 1, T1 = 100, T2 = 1000, dt = 0.005, dT = 10)
    nsteps_heat = Int(2E5)
    nsteps_hot = Int(5E4)
    nsteps_quench = Int(2E6)
    nsteps_equil = Int(1E5)
    nsteps_cool = Int(1E5)
    T_vel = 0.0 

    
    d = LMP(["-screen", "none"]) do lmp
        for i = 1
            # command(lmp, "log none")
            command(lmp, "units metal")
            command(lmp, "dimension 3")
            command(lmp, "atom_style atomic")
            command(lmp, "atom_modify map array")
            command(lmp, "boundary f f f")

            # Setup box
            command(lmp, "region mybox block -20 32 -20 32 -20 32")
            command(lmp, "read_data ./starting_configuration.na")

            # Setup Forcefield
            command(lmp, "pair_style eam/fs")
            command(lmp, "pair_coeff * * Na_v2.eam.fs Na
            ")
            
            
            # computes
            command(lmp, "compute rdf all rdf 100")      # radial distribution function 
            command(lmp, "compute ke all ke/atom")
            command(lmp, "compute pea all pe/atom")
            command(lmp, "compute pe all pe")
            command(lmp, "compute s all stress/atom NULL virial")  
            command(lmp, "compute S all pressure NULL virial")  
            command(lmp, "compute flux all heat/flux ke pea s")  
            command(lmp, "compute msd all msd com yes")

            # Compute center of mass (for RDF's)
            command(lmp, "compute COM all com")
            command(lmp, "region center sphere 0 0 0 5 side in")
            command(lmp, "group core dynamic all region center every $dT")

            # Initialize
            command(lmp, "velocity all create $(T_vel) $seed mom yes rot yes")
            command(lmp, "thermo $dT")
            command(lmp, "timestep $dt")

            # Output fixes 
            command(lmp, "fix fpe all ave/time 1 $dT $dT c_pe file $(save_dir)tmp.pe")
            command(lmp, "fix fvir all ave/time 1 $dT $dT c_S[1] c_S[2] c_S[3] c_S[6] c_S[5] c_S[4] file $(save_dir)tmp.virial")
            command(lmp, "fix fmsd all ave/time 1 $dT $dT c_msd[4] file $(save_dir)tmp.msd")
            command(lmp, "fix fflux all ave/time 1 $dT $dT c_flux[*] file $(save_dir)tmp.flux")



            # Heat to melting point 
            command(lmp, "thermo_style custom step etotal pe temp press vol")
            command(lmp, "fix 1 all nvt temp $(T1) $(T2) 10.0")
            command(lmp, "dump 1 all xyz $(dT) $(save_dir)dump_all.xyz")
            command(lmp, "dump 2 all custom $(dT) $(save_dir)dump_all.positions_and_forces id x y z fx fy fz")

            command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_heat.xyz")
            command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_heat.positions_and_forces id x y z fx fy fz")
            command(lmp, "run $(nsteps_heat)")
            command(lmp, "unfix 1")
            command(lmp, "undump 3")
            command(lmp, "undump 4")

            # Run at hot temperature
            command(lmp, "fix 1 all nvt temp $(T2) $(T2) 10.0")
            command(lmp, "fix frdf core ave/time $dT $(Int(nsteps_hot/dT)-1) $(nsteps_hot) c_rdf[*] file $(save_dir)tmp_hot.rdf mode vector")

            command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_hot.xyz")
            command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_hot.positions_and_forces id x y z fx fy fz")
            command(lmp, "run $(nsteps_hot)")
            command(lmp, "unfix 1")
            command(lmp, "unfix frdf")
            command(lmp, "undump 3")
            command(lmp, "undump 4")

            # Quench
            command(lmp, "fix 1 all nvt temp $(T2) 60.0 10.0")
            command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_quench.xyz")
            command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_quench.positions_and_forces id x y z fx fy fz")
            command(lmp, "run $(nsteps_quench)")
            command(lmp, "unfix 1")
            command(lmp, "undump 3")
            command(lmp, "undump 4")

            # Equilibrate
            command(lmp, "fix 1 all nvt temp 60.0 60.0 10.0")
            command(lmp, "fix frdf core ave/time $dT $(Int(nsteps_equil/dT)-1) $(nsteps_equil) c_rdf[*] file $(save_dir)tmp_equil.rdf mode vector")
            command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_equil.xyz")
            command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_equil.positions_and_forces id x y z fx fy fz")
            command(lmp, "run $(nsteps_equil)")
            command(lmp, "unfix 1")
            command(lmp, "unfix frdf")
            command(lmp, "undump 3")
            command(lmp, "undump 4")

            # Cool
            command(lmp, "fix 1 all nvt temp 60 0.01 10.0")
            command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_cold.xyz")
            command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_cold.positions_and_forces id x y z fx fy fz")
            command(lmp, "run $(nsteps_cool)")
            command(lmp, "unfix 1")
            command(lmp, "undump 3")
            command(lmp, "undump 4")

            # Outputs
            command(lmp, "unfix fpe")
            command(lmp, "unfix fmsd")
            command(lmp, "unfix fvir")
            command(lmp, "unfix fflux")
            command(lmp, "undump 1")
            command(lmp, "undump 2")
            command(lmp, "clear")
        end
    end

end

Tend = Int(50000)
save_dir = "TEMP/"
run_md(Tend, save_dir; seed = 2, T1 = 10.0, T2 = 300.0, dT = 10)