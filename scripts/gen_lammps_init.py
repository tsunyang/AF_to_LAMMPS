def gen_lammps_init(LAMMPS_data,out_name,bond_coeff,n_atom_types,pair_interactions):
    """
    Generates LAMMPS Script for generating restart file for simulations.
    --------------------------------------------------------------------
    
    INPUT
    -----
    LAMMPS_data = string path of LAMMPS data file specifying coordinates of each atom in the Simulation
    bond_coeff = string list of coefficients for fene bond
    n_atom_types = # of atom types
    pair_interaction = str list of lines to write under pair interactions section
    
    
    OUPUT
    -----
    Text file LAMMPS file file for initializing simulations.
    """
    
    with open(out_name, 'w') as f:
        f.write('# HEADER\n\n')
        
        # Set variables
        f.write('# VARIABLES\n')
        f.write('variable fname index %s\n\n'%LAMMPS_data)
        
        # Set units, atom_style, neighbor info and read_data 
        f.write('# PARAMETERS\n')
        f.write('units\tlj\n')
        f.write('boundary\tp p p\n')
        f.write('atom_style\tbond\n')
        f.write('neighbor\t0.4 bin\n')
        f.write('neigh_modify\tevery 1 delay 1\n')
        f.write('read_data\t${fname}\n\n')
        
        # Set bond information
        f.write('# BOND INTERACTIONS\n')
        f.write('special_bonds fene \n')
        f.write('bond_style fene/expand \n')
        for c in bond_coeff:
            f.write('bond_coeff ' + c + '\n')
        f.write('\n')
        
        # Set pair interactions
        f.write('# PAIR INTERACTIONS\n')
        for line in pair_interactions:
            f.write(line + '\n')
        f.write('\n')
        
        # set velocity
        f.write('# Set Velocity\n')
        f.write('velocity all create 1.0 17704\n\n')
        
        # minimize, run, and write restart file:
        f.write('minimize 0 0 500 10000\n')
        f.write('run 0\n')
        f.write('write_restart restart.${run_name}')