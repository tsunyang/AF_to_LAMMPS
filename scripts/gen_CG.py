import numpy as np
import numpy.linalg
from copy import copy
import pandas as pd

# Libraries
elements_dict = {'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012, \
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999, 'P':94.9714, \
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                 'OG' : 294}
one_to_three = {'A': 'ALA',     
                'C': 'CYS',     
                'D': 'ASP',     
                'E': 'GLU',     
                'F': 'PHE',     
                'G': 'GLY',     
                'H': 'HIS',     
                'I': 'ILE',     
                'K': 'LYS',     
                'L': 'LEU',     
                'M': 'MET',     
                'N': 'ASN',     
                'P': 'PRO',            
                'Q': 'GLN',     
                'R': 'ARG',     
                'S': 'SER',     
                'T': 'THR',     
                'V': 'VAL',     
                'W': 'TRP',     
                'Y': 'TYR'}
three_to_one = { 'ALA': 'A',
                 'CYS': 'C',
                 'ASP': 'D',
                 'GLU': 'E',
                 'PHE': 'F',
                 'GLY': 'G',
                 'HIS': 'H',
                 'ILE': 'I',
                 'LYS': 'K',
                 'LEU': 'L',
                 'MET': 'M',
                 'ASN': 'N',
                 'PRO': 'P',
                 'GLN': 'Q',
                 'ARG': 'R',
                 'SER': 'S',
                 'THR': 'T',
                 'VAL': 'V',
                 'TRP': 'W',
                 'TYR': 'Y'}
def combine_AF_PDB_PQR(pdb,pqr,out):
    """
    Helper function to combine header of Alphafold PDB files with the coordinates data supplied by the PQR file generated from
    https://server.poissonboltzmann.org/
    
    INPUT
    -----
    pdb = str name of input pdb file
    pqr = str name of input pqr file
    out = str name of ouput pqr file
    
    OUPUT
    -----
    """
    out_lines = []
    with open(pdb,"r") as f:
        for i,line in enumerate(f):
            vals = line.split()
            if vals[0] == 'ATOM':
                break
            out_lines.append(line)
    with open(pqr,"r") as f:
        for i, line in enumerate(f):
            out_lines.append(line)
    with open(out,"w") as f:
        for line in out_lines: 
            f.write(line)
    
    
def parse_AF_PQR(filename):
    """
    Reads AlphaFold PQR files into a list of residues and an array containing atomic positions
    
    INPUT
    -----
    filename = str name of pdb file to read
    
    OUPUT
    -----
    seq = str list of residues in protein
    pos = pd array containing atomic positions ordered by the following columns:
          |AMINO ACID|RESIDUE #|ATOM TYPE|x|y|z|
    """
    
    
    seq = []
    pos = []
    with open(filename) as f:
        for i,line in enumerate(f):
            vals = line.split()
            
            # find seq
            if vals[0] == 'SEQRES':
                for res in vals[4:]:
                    seq.append(res)
        
            # find coordinates of residues
            if vals[0] == 'ATOM':
                pos.append([vals[3],int(vals[4]),vals[2],float(vals[5]),float(vals[6]),float(vals[7])])
    return seq, pos

def find_res_COM(coords, masses):
    """
    Finds the center of mass of N particles
    
    INPUT
    -----
    coords = N x 3 float array of coordinates of each particle
    masses = N length string list of masses of particles
    
    OUTPUT
    ------
    COM = float list contatining xyz coordinates
    """

    _masses = np.array(masses).reshape(len(masses),1)
    return sum(_masses * coords) / sum(_masses), sum(_masses)

def find_seq_COM(seq, pos):
    """
    Find COM for each residue in an amino acid sequence.
    
    INPUT
    -----
    output of parse_AF_PDB, i.e.:
    seq = str list of residues in protein
    pos = array containing atomic positions ordered by the following columns:
          |AMINO ACID|RESIDUE #|ATOM TYPE|x|y|z|    
    
    OUTPUT
    ------
    CsOM = N x 3 xyz array of amino acid centers of mass
    aa_masses = N list of amino acid masses
    """
    
    res = np.unique([x[1] for x in pos])
    N = len(res)
    CsOM = np.zeros((N,3))
    aa_masses = [0] * N
    n_iter = pos[0][1]
    coords_iter = []
    atoms_iter = []
    n_res = 0
    for i, p in enumerate(pos):
        if p[1] != n_iter:
            masses = [elements_dict[atom] for atom in atoms_iter]
            CsOM[n_res,:], aa_masses[n_res] = find_res_COM(np.array(coords_iter), masses)
            n_res += 1
            coords_iter = []
            atoms_iter = []
            n_iter = p[1]
        else:
            coords_iter.append(p[3:])
            atoms_iter.append(p[2])
    masses = [elements_dict[atom] for atom in atoms_iter]
    CsOM[n_res,:], aa_masses[n_res] = find_res_COM(np.array(coords_iter), masses)
    return CsOM, aa_masses

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
                    f.write(str(int(element)).rjust(6))
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

def write_atom_pdb(coords,path,
                   record_name=[],
                   index=[],
                   at_name=[],
                   alt_loc=[], 
                   res_name=[],
                   chain_id=[],
                   res_num=[],
                   insert=[],
                   occupancy=[],
                   temp_fact=[],
                   seg_id=[],
                   el_symb=[],
                   charge=[]):
    """
    writes pdb files.
    
    INPUT
    -----
    coords = N x 3 np.array of coordinates
    path = string path of created pdb file
    
    OPTIONAL INPUT
    --------------
    See INPUT columns for record type ATOM/HETATM in https://zhanggroup.org/SSIPe/pdb_atom_format.html
    """
    wrt = np.zeros((len(coords),16),dtype=object)
    if len(record_name) == 0:
        wrt[:,0] = ['ATOM']
    else:
        wrt[:,0] = record_name
    if len(index) == 0:
        wrt[:,1] = [str(x) for x in np.arange(len(coords)) + 1]
    else:
        wrt[:,1] = index
    if len(at_name) == 0:
        wrt[:,2] = ['CA' for x in range(len(coords))]
    else:
        wrt[:,2] = at_name
    
    if len(alt_loc) == 0:
        wrt[:,3] = [' ' for x in range(len(coords))]
    else:
        wrt[:,3] = alt_loc
    
    if len(res_name) == 0:
        wrt[:,4] = ['VAL' for x in range(len(coords))]
    else:
        wrt[:,4] = res_name
        
    if len(chain_id)== 0:
        wrt[:,5] = ['A' for x in range(len(coords))]
    else:
        wrt[:,5] = chain_id
         
    if len(res_num) == 0:
        wrt[:,6] = np.arange(len(coords)) + 1
    else:
        wrt[:,6] = [str(x) for x in res_num]
        
    if len(res_num) == 0:
        wrt[:,6] = np.arange(len(coords)) + 1
    else:
        wrt[:,6] = [str(x) for x in res_num]
                   
    if len(insert) == 0:
        wrt[:,7] = ' '
    else:
        wrt[:,7] = insert
                   
    wrt[:,8] = [str(x) for x in np.round(coords[:,0],3)] 
    wrt[:,9] = [str(x) for x in np.round(coords[:,1],3)] 
    wrt[:,10] = [str(x) for x in np.round(coords[:,2],3)] 
    
    if len(occupancy) == 0:
        wrt[:,11] = 1.00
    else:
        wrt[:,11] = [str(np.round(x,2)) for x in occupancy]
                   
    if len(temp_fact) == 0:
        wrt[:,12] = '0.00'           
    else:
        wrt[:,12] = [str(np.round(x,2)) for x in temp_fact]
    
    if len(seg_id) == 0:
        wrt[:,13] = ' '          
    else:
        wrt[:,13] = [str(x) for x in seg_id]
                   
    if len(el_symb) == 0:
        wrt[:,14] = 'C'          
    else:
        wrt[:,14] = [str(x) for x in el_symb]
    
    if len(charge) == 0:
        wrt[:,15] = ' '
    else:
        wrt[:,15] = str(np.round('charge'))

    with open(path,'a') as f:
        for line in wrt:
            f.write("{:<6}{:>5} {:<4}{}{:>3} {}{:>4}{}   {:<8}{:<8}{:<8}{:<6}{:<6}      {:<4}{:>2}{:<2}\n".format(*list(line)))
    return

def write_multimodel_atom_pdb(coord_set,path,
                              model_names=[],
                              record_name_set=[],
                              index_set=[],
                              at_name_set=[],
                              alt_loc_set=[], 
                              res_name_set=[],
                              chain_id_set=[],
                              res_num_set=[],
                              insert_set=[],
                              occupancy_set=[],
                              temp_fact_set=[],
                              seg_id_set=[],
                              el_symb_set=[],
                              charge_set=[]):
    """
    write multi-model pdb files.
    
    INPUT
    -----
    coord_set = list of N x 3 np array of coordinates in each model
    model_name = name of models, default is a 0-indexed list
    See Documentation for write_atom_pdb() for other parameters
    
    OUTPUT
    ------
    pdb files with each model
    """
    
    if len(model_names) == 0:
        model_names = [[] for x in range(len(coord_set))]
    if len(record_name) == 0:
        record_name_set = [[] for x in range(len(coord_set))]
    if len(index_set)==0:
        index_set = [[] for x in range(len(coord_set))]
    if len(at_name_set)==0:
        at_name_set = [[] for x in range(len(coord_set))]
    if len(alt_loc_set)==0:
        alt_loc_set = [[] for x in range(len(coord_set))]
    if len(res_name_set)==0:
        res_name_set = [[] for x in range(len(coord_set))]
    if len(chain_id_set)==0:
        chain_id_set = [[] for x in range(len(coord_set))]
    if len(res_num_set)==0:
        res_num_set = [[] for x in range(len(coord_set))]
    if len(insert_set)==0:
        insert_set = [[] for x in range(len(coord_set))]
    if len(occupancy_set)==0:
        occupancy_set = [[] for x in range(len(coord_set))]
    if len(temp_fact_set)==0:
        temp_fact_set = [[] for x in range(len(coord_set))]
    if len(seg_id_set)==0:
        seg_id_set = [[] for x in range(len(coord_set))]
    if len(el_symb_set)==0:
        el_symb_set = [[] for x in range(len(coord_set))]
    if len(charge_set)==0:
        charge_set = [[] for x in range(len(coord_set))]
        
    if len(index_set)==1:
        index_set = [index_set[0] for x in range(len(coord_set))]
    if len(at_name_set)==1:
        at_name_set = [at_name_set[0] for x in range(len(coord_set))]
    if len(alt_loc_set)==1:
        alt_loc_set = [alt_loc_set[0] for x in range(len(coord_set))]
    if len(res_name_set)==1:
        res_name_set = [res_name_set[0] for x in range(len(coord_set))]
    if len(chain_id_set)==1:
        chain_id_set = [chain_id_set[0] for x in range(len(coord_set))]
    if len(res_num_set)==1:
        res_num_set = [res_num_set[0] for x in range(len(coord_set))]
    if len(insert_set)==1:
        insert_set = [insert_set[0] for x in range(len(coord_set))]
    if len(occupancy_set)==1:
        occupancy_set = [occupancy_set[0] for x in range(len(coord_set))]
    if len(temp_fact_set)==1:
        temp_fact_set = [temp_fact_set[0] for x in range(len(coord_set))]
    if len(seg_id_set)==1:
        seg_id_set = [seg_id_set[0] for x in range(len(coord_set))]
    if len(el_symb_set)==1:
        el_symb_set = [el_symb_set[0] for x in range(len(coord_set))]
    if len(charge_set)==1:
        charge_set = [charge_set[0] for x in range(len(coord_set))]
        
    for i in range(len(coord_set)):
        with open(path,'a') as f:
            f.write('MODEL        %i\n'%i)
        f.close()
        
        write_atom_pdb(coord_set[i],path,
                       index=    index_set[i],
                       at_name=  at_name_set[i],
                       alt_loc=  alt_loc_set[i], 
                       res_name= res_name_set[i],
                       chain_id= chain_id_set[i],
                       res_num=  res_num_set[i],
                       insert=   insert_set[i],
                       occupancy=occupancy_set[i],
                       temp_fact=temp_fact_set[i],
                       seg_id=   seg_id_set[i],
                       el_symb=  el_symb_set[i],
                       charge=   charge_set[i])
        with open(path,'a') as f:
            f.write('ENDMDL\n')
        f.close()
    return
        
def pdb_to_df(path,start,end,index=[]):
    '''
    Convert ATOM Record type pdb coordinates into an pd array as described by
    https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    
    INPUT
    -----
    
    path = str path of pdb file
    start = 0-indexed line to start reading
    end = 0-indexed line to end reading
    index = list of indices to read (optional) if supplied, start and end are not used
    
    OUTPUT
    ------
    array = object array
    '''
    
    if len(index) > 0:
        all_lines = [x for i,x in enumerate(open(path).readlines()) if i in index]
        np_ar = np.zeros((len(all_lines),16),dtype=object)
    else:
        all_lines = open(path).readlines()[start:end+1]
        np_ar = np.zeros((end-start+1,16),dtype=object)
    
    for i, line in enumerate(all_lines):
        np_ar[i,:] = [line[0:6],
                           int(line[6:11]),
                           line[12:16],
                           line[16],
                           line[17:20],
                           line[21],
                           line[22:26],
                           line[26],
                           line[30:38],
                           line[38:46],
                           line[46:54],
                           line[54:60],
                           line[60:66],
                           line[72:76],
                           line[76:78],
                           line[78:80]]
    
    # initialize array
    array = pd.DataFrame(np_ar,columns=['Record Type',
                                      'Atom serial number',
                                      'Atom name',
                                      'Alternate location indicator',
                                      'Residue name',
                                      'Chain identifier',
                                      'Residue sequence number',
                                      'Code for insertions of residues',
                                      'X orthogonal Å coordinate',
                                      'Y orthogonal Å coordinate',
                                      'Z orthogonal Å coordinate',
                                      'Occupancy',
                                      'Temperature factor',
                                      'Segment identifier',
                                      'Element symbol',
                                      'Charge'])       
        
        
    return array

def write_lammps_data(out_name,atoms,n_atom_types,xbound,ybound,zbound,masses):
    
    with open(out_name,'a') as f:
        f.write('# Header\n\n')
        f.write('%i atoms\n'%len(atoms))
        f.write('%i atom types\n\n'%n_atom_types)
        f.write('%f %f xlo xhi\n'%(xbound[0],xbound[1]))
        f.write('%f %f ylo yhi\n'%(ybound[0],ybound[1]))
        f.write('%f %f zlo zhi\n\n'%(zbound[0],zbound[1]))
        f.write('Masses\n\n')
        for i,m in enumerate(masses):
            f.write('%i %f\n'%(i+1,m))
        f.write('\nAtoms\n\n')
        for i in atoms:
            f.write(' '.join([str(x) for x in i])+'\n')
            
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
    
def find_Rg(coords,masses):
    """
    finds radius of gyration of an N-particle ensemble
    
    INPUT
    -----
    coords = N x 3 float array of coordinates of each particle
    masses = N length string list of masses of particles
    
    OUTPUT
    ------
    Rg = float radius of gyration
    """
    
    # find COM
    COM, total_mass = find_res_COM(coords,masses)
    
    # find Rg^2 = avg squared distance of each monomer from COM
    Rg2 = np.sum((coords - COM) ** 2) / len(coords)
    
    return np.sqrt(Rg2)

def FENE(r,K,R0,eps,sig,Del):
    """
    Compute FENE/expand potential as calculated by LAMMPS
    
    INPUT
    -----
    r = float separation distance
    K = float energy factor
    R0 = float radius scale factor for attractive term
    eps = float energy
    sig = float radius scale factor for repulsive term
    Del = float shift factor
    
    OUTPUT
    ------
    fene/expand potential energy
    """
    if r < R0 + Del:
        att = -0.5 * R0 **2 * K * np.log(1 - ((r- Del)/R0)**2)
    else: 
        att = 0
    if r < 2**(1/6) * sig + Del:
        rep = 4 * eps * ((sig/(r-Del)) ** 12 - (sig / (r- Del)) ** 6) + eps
    else:
        rep = 0
    return att + rep

def LJ(r,eps,sig,Del,rc):
    """
    Compute lj/expand potential as calculated by LAMMPS
    
    INPUT
    -----
    r = float separation distance
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc = float constant such that potential cutoff is rc + Del
    
    OUTPUT
    ------
    lj/expand potential energy
    """
    if r < rc + Del:
        return 4 * eps * ((sig/(r-Del)) ** 12 - (sig / (r- Del)) ** 6)
    else:
        return 0

def dist(a,b):
    """
    Compute the distance between vectors.
    
    INPUT
    -----
    a = np vector
    b = np vector
    
    OUPUT
    -----
    float Distance
    """
    return np.sqrt(sum((a-b)**2))