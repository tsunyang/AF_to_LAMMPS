import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model

'''
Functions for creating single polymers and testing intramolecular bonding
'''

def coul_debye(r,qi,qj, kappa,D,rc_coul):
    """
    Calculate Coulombic potential with Debye Screeining
    
    INPUT
    -----
    r = float distance
    qi = float charge of particle 1
    qj = float charge of particle 2
    kappa = float Debye screening length
    D = float coefficient (energy/charge^2) to account for dielectric constant (epsilon) and energy conversion constant (C)
    rc_coul = cutoff radius
    
    OUTPUT
    ------
    float potential
    """
    
    if r < rc_coul:
        return D * qi * qj / r * np.exp(-kappa * r)
    else:
        return 0

def force_coul_debye(r,qi,qj, kappa,D,rc_coul):
    """
    Calculate Coulombic force with Debye Screeining
    
    INPUT
    -----
    r = float distance
    qi = float charge of particle 1
    qj = float charge of particle 2
    kappa = float Debye screening length
    D = float coefficient (energy/charge^2) to account for dielectric constant (epsilon) and energy conversion constant (C)
    rc_coul = cutoff radius
    
    OUTPUT
    ------
    float force
    """
    
    if r < rc_coul:
        return D * qi * qj / r **2 * np.exp(-kappa * r) * (kappa * r + 1)
    else:
        return 0    

def LJ(r,eps,sig,rc_lj):
    """
    Compute lj/cut potential as calculated by LAMMPS
    
    INPUT
    -----
    r = float separation distance
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc_lj = float constant such that potential cutoff is rc
    
    OUTPUT
    ------
    lj potential energy
    """
    
    if r < rc_lj:
        return 4 * eps * ((sig/r) ** 12 - (sig / r) ** 6)
    else:
        return 0
    
def force_LJ(r,eps,sig,rc_lj):
    """
    Compute lj/cut force as calculated by LAMMPS
    
    INPUT
    -----
    r = float separation distance
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc_lj = float constant such that potential cutoff is rc
    
    OUTPUT
    ------
    lj force
    """
    
    if r < rc_lj:
        return -24 * eps * sig ** 6 * (r ** 6 - 2 * sig ** 6) / r ** 13
    else:
        return 0

def VdW(r,eps,sig,rc_lj,lam):
    """
    Calculate VdW potential
    
    INPUT
    -----
    r = float distance
    eps = float energy
    sig = float radius scale factor
    rc_lj = float constant such that potential cutoff is rc
    lam = float in [0,1] hydrophobicity scale

    OUTPUT
    ------
    float potential
    """
    if r <= 2**(1/6) * sig:
        return LJ(r,eps,sig,rc_lj) + (1-lam) * eps
    else:
        return lam * LJ(r,eps,sig,rc_lj)

def force_VdW(r,eps,sig,rc_lj,lam):
    """
    Calculate VdW potential
    
    INPUT
    -----
    r = float distance
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc_lj = float constant such that potential cutoff is rc
    lam = float in [0,1] hydrophobicity scale
    
    OUTPUT
    ------
    float force
    """
    if r <= 2**(1/6) * sig:
        return force_LJ(r,eps,sig,rc_lj)
    else:
        return lam * force_LJ(r,eps,sig,rc_lj)
    
def HPS(r,qi,qj, kappa,D,rc_coul, eps,sig,rc_lj,lam):
    """
    calculate HPS potential
    
    INPUT
    -----
    r = float distance
    qi = float charge of particle 1
    qj = float charge of particle 2
    kappa = float Debye screening length
    D = float coefficient (energy/charge^2) to account for dielectric constant (epsilon) and energy conversion constant (C)
    rc_coul = cutoff radius
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc_lj = float constant such that potential cutoff is rc
    lam = float in [0,1] hydrophobicity scale
    
    OUTPUT
    ------
    float potential
    """
    return coul_debye(r,qi,qj, kappa,D,rc_coul) + VdW(r,eps,sig,rc_lj,lam)

def force_HPS(r,qi,qj, kappa,D,rc_coul, eps,sig,rc_lj,lam):
    """
    calculate HPS force
    
    INPUT
    -----
    r = float distance
    qi = float charge of particle 1
    qj = float charge of particle 2
    kappa = float Debye screening length
    D = float coefficient (energy/charge^2) to account for dielectric constant (epsilon) and energy conversion constant (C)
    rc_coul = cutoff radius
    eps = float energy
    sig = float radius scale factor
    Del = float radius shift
    rc_lj = float constant such that potential cutoff is rc
    lam = float in [0,1] hydrophobicity scale
    
    OUTPUT
    ------
    float force
    """
    return force_coul_debye(r,qi,qj, kappa,D,rc_coul) + force_VdW(r,eps,sig,rc_lj,lam)

def write_table(header,names, tables,filename):
    """
    Writes arbitrary table file for LAMMPS
    
    INPUT
    -----
    header: str list of lines to write in table file header
    names: str list of len(N) listing names for each table in file
    tables: np array of tables to write to table file organized as (M tables x N points x 3) where columns are
            radius|energy|force
    filename: name of output file
    
    OUTPUT
    ------
    table file for LAMMPS
    """
    
    with open(filename,'w') as f:
        for line in header:
            f.write(line + '\n')
        f.write('\n')
        for i, name in enumerate(names):
            f.write(name + '\n')
            f.write('N ' + str(tables.shape[1]) + ' R ' + str(tables[i,0,0]) + ' ' + str(tables[i,-1,0]))
            f.write('\n\n')
            for j in range(tables.shape[1]):
                f.write(str(j+1) + ' ' + str(tables[i,j,0]) + ' ' + str(tables[i,j,1]) + ' ' + str(tables[i,j,2]) + '\n')
            f.write('\n')

def gen_random_coord(c1,radius):
    """
    generates random coordinate of next atom in chain.
    
    INPUT
    -----
    c1 = float list coordinate of last atom
    raidus = float bond length between last and current atom
    
    OUPUT
    -----
    c2 = float list of coordinate of current atom
    """
    # generate random theta and pi
    theta, phi = np.array([np.pi,2 * np.pi]) * np.random.rand(2)
    
    return np.array(c1) + np.array([radius * np.cos(phi) * np.sin(theta),
                                    radius * np.sin(phi) * np.sin(theta),
                                    radius * np.cos(theta)])

def check_volume_exclusion(c1,coords,r):
    """
    Checks if two atoms have overlapping radii
    
    INPUT
    -----
    c1 = float array of coordinate of atom 1
    coords = coordinates to check
    r = pairwise radii of overlap
    
    OUTPUT
    ------
    returns True if atoms have overlapping volume and False otherwise.
    """
    return np.any(np.linalg.norm(c1 - coords,axis=1) < r)

def pqr_to_pdb(pqr):
    name = pqr.strip('.pqr') + '.pdb'
    lines = []
    with open(pqr,'r') as f:
        for line in f:
            if line.split()[0] == 'ATOM':
                charge = line.split()[-2]
                i_charge = line.rfind(charge)
                lines.append(line[:i_charge].rstrip())
            else:
                lines.append(line)
    with open(name,'w') as f:
        for line in lines:
            f.write(line + '\n')
            
def find_lin_segs(data,r2_thresh,len_thresh,nc_list):
    '''
    Finds linear segments of 3d dataset of polymers.
    
    INPUT
    -----
    data = Nx3 np array of coordinates
    r2_thresh = float r2 threshold above which points are deemed colinear
    len_thresh = minimum length of colinear points
    nc_list = optional()
    
    OUTPUT
    ------
    lin_segs = list of list of indices corresponding to linear segments in data.
    '''
    # create regression model
    model = sklearn.linear_model.LinearRegression()
    
    # initialize data
    lin_segs = []
    N = len(data)
    jump = 0
    
    for n in range(N-3):
        
        # jump amino acids already in a linear segment
        if n < jump:
            continue
        
        # check overlap of sequence of three consecutive points
        overlap = False
        for _n in [n,n+1,n+2]:
            if _n in nc_list:
                overlap = True
                break
                
        if overlap:
            continue
        
        # find linear regression of points
        stop = n+3
        X = data[n:stop,:2]
        Z = data[n:stop,2]
        current_fit = model.fit(X,Z)
        
        # check if points are linear and non-overlapping
        while current_fit.score(X,Z) >= r2_thresh:

            # stop at end of polymer
            if stop >= N:
                break
     
            # check overlap
            if stop in nc_list:
                break
            
            # update linear regression
            stop += 1
            X = data[n:stop,:2]
            Z = data[n:stop,2]
            current_fit = model.fit(X,Z)
        
        # check if colinear group is too short
        if len(np.arange(n,stop)) < len_thresh:
            continue
            
        # add segment to output
        else:
            lin_segs.append([i for i in range(n,stop)])
            jump = stop
    return lin_segs