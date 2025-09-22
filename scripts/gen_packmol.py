import numpy as np
from numpy.linalg import norm
import pandas as pd
import re
import sys

##############################################################
# Functions for generating input scripts to initialize       #          
# polymer models for BD simulations in LAMMPS.               #
# Generation of initial structure done by Packmol            # 
##############################################################

#---------------FUNCTIONS-----------------------#
def get_unit(vector):
    '''
    Compute unit vector
    '''
    # return zero if vector is zero
    if vector[0] == 0 and vector[1] == 0 and vector[2] == 0:
        return vector
    else:
        return vector / norm(vector)

def gen_chain(bond_lengths,bond_angles,dihedral_angles):
    '''
    Generates 1D coordinates for one chain of a polymer.
    -------------------------------------------
    
    Inputs
    ------
    bond_lengths = list of distances between adjacent particle centers in radians (N-1)
    bond_angles = angle between every 3 atoms (N-2)
    dihedral_angles = torsion angle between every 4 atoms (N-3)
    '''
    
    # initialize coord array and compute first 3 positions, fixing first monomer at [0,0,0]
    N = len(bond_lengths) + 1
    coords = np.zeros((N,3))
    if N >= 2:
        coords[1,:] = [bond_lengths[0],0,0]
    if N >= 3:
        dx = bond_lengths[1] * np.cos(np.pi - bond_angles[0])
        dy = bond_lengths[1] * np.sin(np.pi - bond_angles[0])
        coords[2,:] = [bond_lengths[0] + dx, dy, 0]
    
    # Set rest of coordinates
    if N >= 4:
        for i in range(3,N):
            r = bond_lengths[i-1]
            theta = bond_angles[i-2]
            phi = dihedral_angles[i-3]
            
            # transform to cartesian frame
            xyz = r * np.array([-np.cos(theta),
                                np.cos(phi) * np.sin(theta),
                                np.sin(phi) * np.sin(theta)])
            
            # Use affine transformation based on last three monomers
            bc_norm = get_unit(coords[i-1,:] - coords[i-2,:])
            ab = coords[i-2,:] - coords[i-3,:]
            n = get_unit(np.cross(ab,bc_norm))
            M = np.array([bc_norm,np.cross(n,bc_norm),n]).T
            coords[i,:] = coords[i-1] + np.matmul(M,xyz)
     
    return coords

def gen_pqr_file(pos,filename):
    """
    Writes a PQR file.
    -------------------------------------------------------------
    INPUT
    -----
    pos: pd array with rows including the following information in order:
         record type, atom number, atom name, residue name, residue number, x, y, and z coordinates, charge, and                          radius.
    filename: str name of output file
    """
    N = np.shape(pos)[0]
    with open(filename, 'w') as f:
        for line in pos.values:
            for i, element in enumerate(line):
                if i == 0:
                    f.write(element)
                elif i == 1:
                    f.write(str(element).rjust(7))
                elif i == 2:
                    alphabet ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                    for p, char in enumerate(element):
                        if char in alphabet:
                            break
                    after = len(element)-(p+1)
                    f.write(str(element).rjust(3 + after))
                elif i == 3:
                    f.write(element.rjust(6 - after))
                elif i == 4:
                    f.write(str(element).rjust(6))
                elif i == 5:
                    f.write(('%.3f'%element).rjust(12))
                elif i > 5:
                    f.write(('%.3f'%element).rjust(8))
            f.write('\n')
        for i in range(len(pos)):
            f.write('CONECT')
            if i == 0:
                f.write('    1    2\n')
            elif i == N-1:
                f.write(str(N).rjust(5) + str(N-1).rjust(5) + '\n')
            else:
                f.write(str(i+1).rjust(5) + str(i+2).rjust(5) + str(i).rjust(5) + '\n')
        f.write('END')
        
def gen_pdb_file(coords,filename):
    """
    Generates a PDB file from N-mer polymer xyz coordinate array.
    -------------------------------------------------------------
    INPUT
    -----
    coords: N x 3 array containing xyz coordinates of each monomer in the polymer strand
    filename: str name of output file
    """
    N = np.shape(coords)[0]
    with open(filename, 'w') as f:
        f.write('HEADER\n')
        for i in range(N):
            if i == 0 or i == N-1:
                ct = 'C2 '
            elif i % 2 == 0:
                ct = 'C1A'
            elif i % 2 == 1:
                ct = 'C1B'
            f.write('ATOM' + str(i+1).rjust(7,' ') + ct.rjust(5,' ') + 'PE'.rjust(3,' ') + '0'.rjust(7, ' ')+('%.3f'%coords[i,0]).rjust(12,' ') + ('%.3f'%coords[i,1]).rjust(8,' ')+('%.3f'%coords[i,2]).rjust(8) + '1.00'.rjust(6) + '0.00'.rjust(6) + 'C'.rjust(10) + '\n')
        for i in range(N):
            f.write('CONECT')
            if i == 0:
                f.write('    1    2\n')
            elif i == N-1:
                f.write(str(N).rjust(5) + str(N-1).rjust(5) + '\n')
            else:
                f.write(str(i+1).rjust(5) + str(i+2).rjust(5) + str(i).rjust(5) + '\n')
        f.write('END')
        
def gen_packmol_inp(sequence,rad_dict,pdbs,in_name,out_name,n_polymers,volume):
    """
    Generate input file for bead-spring polymer in packmol, which respects volume exclusion between monomers.
    Currently packs polymers into a cube.
    ---------------------------------------------------------------------------------------------------------
    INPUT
    -----
    sequence: list of ints specifying order of monomers in a strand
    rad_dict: dictionary specifying radius of each monomer type
    pdb: str list of pdb filenames
    in_name: string of input file to be submitted to Packmol
    out_name: string of output pdb file
    n_polymers: int list number of polymers to place in simulation box corresponding to each pdb file
    volume: float volume of simulation box
    
    OUPUT
    -----
    out = ouput file to fun from command line as packmol < filename
    """
    
    # Find box coordinates
    l = (volume) ** (1/3)
    
    # group monomer IDs by radius
    rev_rad_dict = dict()
    keys =  np.unique(list(rad_dict.values()))
    for key in keys:
        rev_rad_dict[key] = [i for i, j in rad_dict.items() if j == key]
    pos_dict = dict(zip(rad_dict.keys(),[(np.where(np.array(sequence) == key)[0]+1) for key in rad_dict.keys()])) 
        
    with open(in_name, 'w') as f:
        
        # Set minimum distance tolerance, file type, and output file
        f.write('tolerance 2.0\n')
        f.write('filetype pdb\n')
        f.write('output %s\n\n'%out_name)
        
        # Define structure
        for i, pdb in enumerate(pdbs):
            f.write('structure %s\n'%pdb)
            f.write('  number %i\n'%n_polymers[i])
            f.write('  chain A\n')
            f.write('  resnumbers 0\n')
            f.write('  inside box 0. 0. 0. %f %f %f\n'%(l,l,l))

            # assign monomer sizes
            # find monomers without size 1:
            non_1_sizes = []
            for labels in np.unique(sequence):
                if rad_dict[labels] != 1:
                    non_1_sizes.append(labels)
            # assign sizes to monomers
            for label in non_1_sizes:
                f.write('  atoms ')
                for val in pos_dict[label]:
                    f.write(str(val) + ' ')
                f.write('\n')
                f.write('    radius %f\n'%rad_dict[label])
                f.write('  end atoms\n')        
            f.write('end structure\n\n')

def write_lammps_data(atom_info,out_name,sequence,n_molecules,atom_radii, grid_vol,bond_types,masses,angles=False,bond_write=[]):
    """
    Writes LAMMPS data file for FENE bonded molecules.
    
    INPUT
    -----
    atom_info: string list of values to place under header ATOM
    out_name: name of output LAMMPS initialization file
    sequence: str list specifying order of monomers in a strand
    n_molecules: int number of strands in simulation
    atom_radii: float list of atom radii
    grid_vol: float volume of simuation grid
    bond_types: str list of order of bond types for each polymer strand.
    masses: float list of masses for each atom type
    angles: optional boolean for writing angle types
    bond_write: optional numpy array for bond topology. If [], bond topology is inferred from linear sequence of atoms & n_molecules
    """
    
    with open(out_name, 'w') as f:
        
        #-----HEADER------#
        f.write('# Header\n\n')
        
        # Write total number of atoms, bonds, angles, and dihedrals
        n_mon = len(sequence)
        n_atoms = n_molecules * n_mon
        if len(bond_write) == 0:
            n_bonds = n_molecules * (n_mon - 1)
        else:
            n_bonds = len(bond_write)
        n_angles = n_molecules * (n_mon - 2)
#         n_dihedrals = n_molecules * (n_mon -3)
        f.write(('%i'%n_atoms).rjust(10) + ('atoms').rjust(10) + '\n')
        f.write(('%i'%n_bonds).rjust(10) + ('bonds').rjust(10) + '\n')
        
#         #---------For use in Harmonic Style Bonds--------------------#
        if angles != False:
            f.write(('%i'%n_angles).rjust(10) + ('angles').rjust(10) + '\n')
#         f.write(('%i'%n_dihedrals).rjust(10) + ('dihedrals').rjust(10) + '\n')

        f.write('\n')
        
        # Write number of types of atoms, bonds, angles, and dihedrals
        n_atom_types = len(masses)
        n_bond_types = len(np.unique(bond_types))
        f.write(('%i'%n_atom_types).rjust(10) + '     ' + ('atom types\n'))
        f.write(('%i'%n_bond_types).rjust(10) + '     ' + ('bond types\n'))
        
         #---------For use in Harmonic Style Bonds--------------------#
        if angles != False:
            f.write('1'.rjust(10) + '     ' + ('angle types\n'))
#         f.write('1'.rjust(10) + '     ' + ('dihedral types\n')) 

#         f.write('\n')
        
        # Write box dimensions
        l = (grid_vol) ** (1/3)
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' xlo xhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' ylo yhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' zlo zhi\n\n')
        
        #-----Masses----#
        
        # set densities of each particle to 1
        f.write('Masses\n\n')
        for i, mass in enumerate(masses):
            f.write(str(i+1).rjust(10) + '          ' + str(mass) + '\n')
        
        #-----Atoms----#
        f.write('\nAtoms\n\n')
        for line in atom_info:
            for val in line:
                f.write(str(val).rjust(10))
            f.write('\n')
            
        #----Bonds----#
        if bond_write == []:
            f.write('\nBonds\n\n')
            b_count = 1
            atom = 1    
            while atom <= n_atoms:
                if ((atom) %n_mon != 0):
                    bond_line = [str(b_count).rjust(10),
                                 ('%i'%bond_types[(b_count-1) % (n_mon-1)]).rjust(10),
                                 ('%i'%(atom)).rjust(10),
                                 ('%i'%(atom+1)).rjust(10) + '\n']
                    f.write(''.join(bond_line))
                    b_count += 1
                    atom += 1
                else:
                    atom += 1
        else:
            f.write('\nBonds\n\n')
            for line in bond_write:
                for val in line:
                    f.write(str(val).rjust(10))
                f.write('\n')
                
        #---------For use in Harmonic Style Bonds--------------------#                       
        #----Angles----#
        if angles:
            
            f.write('\nAngles\n\n')
            an_count = 1
            atom = 1
            while atom <= n_atoms:
                if ((atom+1) %n_mon) != 0:
                    ang_line = [str(an_count).rjust(10),
                                '1'.rjust(10),
                                ('%i'%(atom)).rjust(10),
                                ('%i'%(atom+1)).rjust(10),
                                ('%i'%(atom+2)).rjust(10) + '\n']
                    f.write(''.join(ang_line))
                    an_count += 1
                    atom += 1
                else:
                    atom += 2

#         #----Dihedrals----#
#         f.write('\nDihedrals\n\n')
#         di_count = 1
#         atom = 1
#         while atom <= n_atoms:
#             if ((atom+2) %n_mon) != 0:
#                 di_line =  [str(di_count).rjust(10),
#                             '1'.rjust(10),
#                             ('%i'%(atom)).rjust(10),
#                             ('%i'%(atom+1)).rjust(10),
#                             ('%i'%(atom+2)).rjust(10),
#                             ('%i'%(atom+3)).rjust(10) + '\n']
#                 f.write(''.join(di_line))
#                 di_count += 1
#                 atom += 1
#             else:
#                 atom += 3

def write_lammps_dihedral(atom_info,out_name,sequence,n_molecules,atom_radii, grid_vol,bond_types,dihedral_types,masses):
    """
    Writes LAMMPS data file for dihedral polymers
    
    INPUT
    -----
    atom_info: string list of values to place under header ATOM
    out_name: name of output LAMMPS initialization file
    sequence: str list specifying order of monomers in a strand
    n_molecules: int number of strands in simulation
    atom_radii: float list of atom radii
    grid_vol: float volume of simuation grid
    bond_types: str list of order of bond types for each polymer strand.
    dihedral_types: str list of order of dihedral types in polymer strand
    masses: float list of masses for each atom type
    """
    
    with open(out_name, 'w') as f:
        
        #-----HEADER------#
        f.write('# Header\n\n')
        
        # Write total number of atoms, bonds, angles, and dihedrals
        n_mon = len(sequence)
        n_atoms = n_molecules * n_mon
        n_bonds = n_molecules * (n_mon - 1)
        n_angles = n_molecules * (n_mon - 2)
        n_dihedrals = n_molecules * (n_mon -3) * len(dihedral_types)
        f.write(('%i'%n_atoms).rjust(10) + ('atoms').rjust(10) + '\n')
        f.write(('%i'%n_bonds).rjust(10) + ('bonds').rjust(10) + '\n')
        
        #---------For use in Harmonic Style Bonds--------------------#    
        f.write(('%i'%n_angles).rjust(10) + ('angles').rjust(10) + '\n')
        f.write(('%i'%n_dihedrals).rjust(10) + ('dihedrals').rjust(10) + '\n')

        f.write('\n')
        
        # Write number of types of atoms, bonds, angles, and dihedrals
        n_atom_types = len(np.unique(sequence))
        n_bond_types = len(np.unique(bond_types))
        f.write(('%i'%n_atom_types).rjust(10) + '     ' + ('atom types\n'))
        f.write(('%i'%n_bond_types).rjust(10) + '     ' + ('bond types\n'))
        
        #---------For use in Harmonic Style Bonds--------------------#    
        f.write('1'.rjust(10) + '     ' + ('angle types\n'))
        f.write(str(len(dihedral_types)).rjust(10) + '     ' + ('dihedral types\n')) 

        f.write('\n')
        
        # Write box dimensions
        l = (grid_vol) ** (1/3)
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' xlo xhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' ylo yhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' zlo zhi\n\n')
        
        #-----Masses----#
        
        # set densities of each particle to 1
        f.write('Masses\n\n')
        for i, mass in enumerate(masses):
            f.write(str(i+1).rjust(10) + '          ' + str(mass) + '\n')
        
        #-----Atoms----#
        f.write('\nAtoms\n\n')
        for line in atom_info:
            for val in line:
                f.write(str(val).rjust(10))
            f.write('\n')
            
        #----Bonds----#
        atom = 1
        f.write('\nBonds\n\n')
        b_count = 1
        while atom <= n_atoms:
            if ((atom) %n_mon != 0):
                bond_line = [str(b_count).rjust(10),
                             ('%i'%bond_types[(b_count-1) % (n_mon-1)]).rjust(10),
                             ('%i'%(atom)).rjust(10),
                             ('%i'%(atom+1)).rjust(10) + '\n']
                f.write(''.join(bond_line))
                b_count += 1
                atom += 1
            else:
                atom += 1
        
        # delete this (for PEG + VTS1)
        atom = 52824
        n_atoms = 159023
        n_mon = 180
        n_molecules = 590
        while atom <= n_atoms:
            if ((atom-52823) %n_mon != 0):
                bond_line = [str(b_count).rjust(10),
                             ('2').rjust(10),
                             ('%i'%(atom)).rjust(10),
                             ('%i'%(atom+1)).rjust(10) + '\n']
                f.write(''.join(bond_line))
                b_count += 1
                atom += 1
            else:
                atom += 1
                
        #---------For use in Harmonic Style Bonds--------------------#                       
        #----Angles----#
        f.write('\nAngles\n\n')
        an_count = 1
        atom = 52824
        while atom <= n_atoms:
            if ((atom-52822)%n_mon) != 0:
                ang_line = [str(an_count).rjust(10),
                            '1'.rjust(10),
                            ('%i'%(atom)).rjust(10),
                            ('%i'%(atom+1)).rjust(10),
                            ('%i'%(atom+2)).rjust(10) + '\n']
                f.write(''.join(ang_line))
                an_count += 1
                atom += 1
            else:
                atom += 2
        
         #----Dihedrals----#
        f.write('\nDihedrals\n\n')
        remove_start = np.arange(n_mon-3,n_mon)
        to_remove = np.array([remove_start + n_mon*i for i in range(n_molecules)]).flatten()
        all_sets = np.array([1 + i for i in range(n_mon * n_molecules)])
        dihedral_starts = np.delete(all_sets,to_remove,axis=0)
        dihedral_starts += 52823
        count = 1
        for i,di_start in enumerate(dihedral_starts):
            for di_type in dihedral_types:
                line = [str(count).rjust(10), 
                        str(di_type).rjust(10),
                        str(di_start).rjust(10),
                        str(di_start+1).rjust(10),
                        str(di_start+2).rjust(10),
                        str(di_start+3).rjust(10) + '\n']
                count += 1
                f.write(''.join(line))

def pdb_to_lammps(pdb_file,out_name,sequence,n_molecules,atom_radii, grid_vol,bond_types,masses):
    """
    Converts pdb file to LAMMPS data file.
    
    INPUT
    -----
    pdb_file: string file name of pdb file containing initialization
    out_name: name of output LAMMPS initialization file
    sequence: str list specifying order of monomers in a strand
    n_molecules: int number of strands in simulation
    atom_radii: float list of atom radii
    grid_vol: float volume of simuation grid
    bond_types: str list of order of bond types for each polymer strand.
    masses: float list of masses for each atom type
    """
    
    # read pdb file
    read_csv = pd.read_csv(pdb_file)
    with open(out_name, 'w') as f:
        
        #-----HEADER------#
        f.write('# Header\n\n')
        
        # Write total number of atoms, bonds, angles, and dihedrals
        n_mon = len(sequence)
        n_atoms = n_molecules * n_mon
        n_bonds = n_molecules * (n_mon - 1)
        n_angles = n_molecules * (n_mon - 2)
        n_dihedrals = n_molecules * (n_mon -3)
        f.write(('%i'%n_atoms).rjust(10) + ('atoms').rjust(10) + '\n')
        f.write(('%i'%n_bonds).rjust(10) + ('bonds').rjust(10) + '\n')
        
        #---------For use in Harmonic Style Bonds--------------------#    
#         f.write(('%i'%n_angles).rjust(10) + ('angles').rjust(10) + '\n')
#         f.write(('%i'%n_dihedrals).rjust(10) + ('dihedrals').rjust(10) + '\n)

        f.write('\n')
        
        # Write number of types of atoms, bonds, angles, and dihedrals
        atom_types = len(np.unique(sequence))
        f.write(('%i'%atom_types).rjust(10) + '     ' + ('atom types\n'))
        f.write(('%i'%(1/2 *atom_types * (atom_types+1))).rjust(10) + '     ' + ('bond types\n'))
        
         #---------For use in Harmonic Style Bonds--------------------#    
#         f.write('1'.rjust(10) + '     ' + ('angle types\n'))
#         f.write('1'.rjust(10) + '     ' + ('dihedral types\n')) 

        f.write('\n')
        
        # Write box dimensions
        l = (grid_vol) ** (1/3)
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' xlo xhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' ylo yhi\n')
        f.write('0.0000'.rjust(10) + '   ' + '%.4f'%l + ' zlo zhi\n\n')
        
        #-----Masses----#
        f.write('Masses\n\n')
        for i, mass in enumerate(masses):
            f.write(str(i+1).rjust(10) + '          ' + str(mass) + '\n')
        
        #-----Atoms----#
        f.write('\nAtoms\n\n')
        count = 0
        for i in range(np.shape(pdb)[0]):
            if 'ATOM' in pdb.iloc[i,0]:
                pdb_line = re.findall('(\d+(?:\.\d{3})?)',pdb.iloc[i,0])
                lammps_line = [pdb_line[0].rjust(10),
                               pdb_line[2].rjust(10), 
                               ('%s'%(sequence[count%n_mon])).rjust(10),
                               pdb_line[3].rjust(10),
                               pdb_line[4].rjust(10),
                               pdb_line[5].rjust(10) + str('\n')]
                f.write(''.join(lammps_line))
                count += 1
        
        #----Bonds----#
        f.write('\nBonds\n\n')
        b_count = 1
        atom = 1    
        while atom <= n_atoms:
            if ((atom) %n_mon != 0):
                bond_line = [str(b_count).rjust(10),
                             ('%i'%bond_types[(b_count-1) % (n_mon-1)]).rjust(10),
                             ('%i'%(atom)).rjust(10),
                             ('%i'%(atom+1)).rjust(10) + '\n']
                f.write(''.join(bond_line))
                b_count += 1
                atom += 1
            else:
                atom += 1
                
          #---------For use in Harmonic Style Bonds--------------------#                       
#         #----Angles----#
#         f.write('\nAngles\n\n')
#         an_count = 1
#         atom = 1
#         while atom <= n_atoms:
#             if ((atom+1) %n_mon) != 0:
#                 ang_line = [str(an_count).rjust(10),
#                             '1'.rjust(10),
#                             ('%i'%(atom)).rjust(10),
#                             ('%i'%(atom+1)).rjust(10),
#                             ('%i'%(atom+2)).rjust(10) + '\n']
#                 f.write(''.join(ang_line))
#                 an_count += 1
#                 atom += 1
#             else:
#                 atom += 2
        
#         #----Dihedrals----#
#         f.write('\nDihedrals\n\n')
#         di_count = 1
#         atom = 1
#         while atom <= n_atoms:
#             if ((atom+2) %n_mon) != 0:
#                 di_line =  [str(di_count).rjust(10),
#                             '1'.rjust(10),
#                             ('%i'%(atom)).rjust(10),
#                             ('%i'%(atom+1)).rjust(10),
#                             ('%i'%(atom+2)).rjust(10),
#                             ('%i'%(atom+3)).rjust(10) + '\n']
#                 f.write(''.join(di_line))
#                 di_count += 1
#                 atom += 1
#             else:
#                 atom += 3

def lammpstrj_to_pqr(lammpstrj_file,out_name, timestep, charge_dict,radius_dict,mol_IDs):
    """
    Convert lammpstrj file to a pqr file.
    
    INPUT
    -----
    lammpstrj_file = str name of lammps trajectory file
    out_name = str name of output pqr file
    timestep = int number of timestep from which to extract coordinates
    charge_dict = dict containing charge of particle, e.g. {1 : 0, 2 : 0, 3 : -1.0, 4 : 1.0}
    radius_dict = dict containing radius of particle, e.g. {1 : 1.5, 2 : 10, 3 : 1.5, 4 : 1.5}
    mol_IDs = int list containing molecule IDs of each atom
    
    OUPUT
    -----
    a pqr file with rows containing the following info:
    record type, atom number, atom name, residue name, residue number, x, y, and z coordinates, charge, and radius.
    """
    
    t_index = -1
    n_atoms = -1
    with open(lammpstrj_file,'r') as f:
        for i, line in enumerate(f):
            
            # find lines corresponding to timestep:
            if line == str(timestep) + '\n':
                t_index = i
                continue
            if t_index != -1 and i == (t_index+2):
                n_atoms = int(line)
                pqr = np.zeros((n_atoms,10),dtype='object')
                atom_types = np.zeros(n_atoms)
                start = t_index + 8
                end = t_index + 7 + n_atoms

            
            # read data
            if n_atoms != -1 and i >= start and i <= end:
                k = i - start
                vals = line.split()
                pqr[k,1] = int(vals[0])
                pqr[k,5:8] = [float(vals[2]),float(vals[3]),float(vals[4])]
                atom_types[k] = int(vals[1])
            
        # fill rest of values
        pqr[:,0] = ['ATOM'] * n_atoms
        pqr[:,2] = ['C'] * n_atoms
        pqr[:,3] = ['VAL'] * n_atoms
        pqr[:,8] = [charge_dict[atom_type] for atom_type in atom_types]
        pqr[:,9] = [radius_dict[atom_type] for atom_type in atom_types]
    
    # Create pqr file
    pqr = pqr[np.argsort(pqr[:, 1])]
    pqr[:,4] = mol_IDs
    gen_pqr_file(pd.DataFrame(pqr),out_name)
