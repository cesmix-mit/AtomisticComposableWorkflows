using LAMMPS

function compute_lattice(lattice_constant, save_dir::String)
    d = LMP(["-screen", "none"]) do lmp
        for i = 1
            # command(lmp, "log none")
            command(lmp, "units metal")
            command(lmp, "dimension 3")
            command(lmp, "boundary p p p")

            # Setup box
            command(lmp, "region box block 0 13.0 0 13.0 0 13.0 units lattice")
            command(lmp, "create_box 1 box")
            command(lmp, "lattice bcc $(lattice_constant)")
            command(lmp, "create_atoms 1 box")
            

            command(lmp, "mass 1 22.989769")
            # Setup Forcefield
            command(lmp, "pair_style zero 10.0")
            command(lmp, "pair_coeff * *")
            
            
            # Outputs
            dt = 0.001
            dT = 1
            command(lmp, "thermo $(dT)")
            command(lmp, "timestep $dt")
            command(lmp, """run $dT every $dT "write_data $(save_dir)DATA.*" """)
            command(lmp, "clear")
        end
    end

end

save_dir = "./"
lattice_parameter = 4.28 # From Nichol and Auckland
# lattice_parameter = 3.71
compute_lattice(lattice_parameter, save_dir)