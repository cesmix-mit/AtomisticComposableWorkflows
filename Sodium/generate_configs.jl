using LAMMPS

function run_md(Tend::Int, save_dir::String; seed = 1, T1 = 100, T2 = 1000, dt = 0.0001, dT = 10)
    nsteps_heat = Int(2E6)
    nsteps_hot = Int(5E5)
    nsteps_quench = Int(5E6)
    nsteps_equil = Int(1E6)
    nsteps_cool = Int(1E6)
    T_vel = 0.0 

    lattice_constant = 4.28
    
    d = LMP(["-screen", "none"]) do lmp
        for i = 1
            # command(lmp, "log none")
            command(lmp, "units metal")
            command(lmp, "dimension 3")
            command(lmp, "atom_style atomic")
            command(lmp, "atom_modify map array")
            command(lmp, "boundary p p p")

            # Setup box
            command(lmp, "region total_box block $(-2lattice_constant) $(2lattice_constant) $(-2lattice_constant) $(2lattice_constant) $(-2lattice_constant) $(2lattice_constant)")
            
            
            command(lmp, "create_box 1 total_box")
            command(lmp, "lattice bcc $(lattice_constant)")
            command(lmp, "create_atoms 1 region total_box")

            # Setup Forcefield
            
            command(lmp, "pair_style hybrid/overlay eam/fs zero 10.0")
            command(lmp, "pair_coeff * * eam/fs Na_v2.eam.fs Na")
            command(lmp, "pair_coeff * * zero")
                        
            # computes
            command(lmp, "compute rdf all rdf 200")      # radial distribution function 
            command(lmp, "compute ke all ke/atom")
            command(lmp, "compute pea all pe/atom")
            command(lmp, "compute pe all pe")
            command(lmp, "compute s all stress/atom NULL virial")  
            command(lmp, "compute S all pressure NULL virial")  
            command(lmp, "compute flux all heat/flux ke pea s")  
            command(lmp, "compute msd all msd com yes")

            # Initialize
            command(lmp, "velocity all create $(T_vel) $seed mom yes rot yes")
            command(lmp, "thermo $dT")
            command(lmp, "timestep $dt")

            # Output fixes 
            command(lmp, "fix fpe all ave/time 1 $dT $dT c_pe file $(save_dir)tmp.pe")
            command(lmp, "fix fvir all ave/time 1 $dT $dT c_S[1] c_S[2] c_S[3] c_S[6] c_S[5] c_S[4] file $(save_dir)tmp.virial")
            command(lmp, "fix fmsd all ave/time 1 $dT $dT c_msd[4] file $(save_dir)tmp.msd")
            command(lmp, "fix fflux all ave/time 1 $dT $dT c_flux[*] file $(save_dir)tmp.flux")
            command(lmp, "fix remove_drift all recenter 0.5 0.5 0.5 units box")
            command(lmp, "dump 1 all xyz $(dT) $(save_dir)dump_all.xyz")
            command(lmp, "dump 2 all custom $(dT) $(save_dir)dump_all.positions_and_forces id x y z fx fy fz")

            # Heat to melting point 
            TEMPS = [20, 100, 300, 500, 1000, 100, 10]
            number_of_steps = Int.([1E6, 1E6, 1E6, 2E6, 5E6, 1E6])
            number_of_steps_rdf = Int(1E5)
            labels = ["warm", "melt", "heat", "hot", "quench", "equilibrate"]
            for j = 1:6
                command(lmp, "thermo_style custom step etotal pe temp press vol")
                if j <= 4
                    command(lmp, "fix 1 all npt temp $(TEMPS[j]) $(TEMPS[j+1]) $(100*dt) iso 0.0 0.0 $(100*dt)")
                else 
                    command(lmp, "fix 1 all nvt temp $(TEMPS[j]) $(TEMPS[j+1]) $(100*dt)")
                end

                command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_$(labels[j])_1.xyz")
                command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_$(labels[j])_1.positions_and_forces id x y z fx fy fz")
                command(lmp, "run $(number_of_steps[j])")
                command(lmp, "unfix 1")
                command(lmp, "undump 3")
                command(lmp, "undump 4")

                if j <= 4
                    command(lmp, "fix 1 all nvt temp $(TEMPS[j+1]) $(TEMPS[j+1]) $(100*dt)")
                    command(lmp, "fix frdf all ave/time 100 100 10000 c_rdf[*] file $(save_dir)tmp_$(labels[j]).rdf mode vector")
                    command(lmp, "dump 3 all xyz $(dT) $(save_dir)dump_$(labels[j])_2.xyz")
                    command(lmp, "dump 4 all custom $(dT) $(save_dir)dump_$(labels[j])_2.positions_and_forces id x y z fx fy fz")
                    command(lmp, "run $(number_of_steps_rdf)")
                    command(lmp, "unfix 1")
                    command(lmp, "unfix frdf")
                    command(lmp, "undump 3")
                    command(lmp, "undump 4")
                end
            end

            # Outputs
            command(lmp, "unfix fpe")
            command(lmp, "unfix fmsd")
            command(lmp, "unfix fvir")
            command(lmp, "unfix fflux")
            command(lmp, "unfix remove_drift")
            command(lmp, "undump 1")
            command(lmp, "undump 2")
            command(lmp, "clear")
        end
    end

end

Tend = Int(50000)
save_dir = "TEMP/"
run_md(Tend, save_dir; seed = 2, T1 = 10.0, T2 = 300.0, dT = 1000)