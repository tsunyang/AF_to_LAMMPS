import numpy as np
import pandas as pd
import sys
from sklearn.metrics import pairwise_distances
from copy import deepcopy
import json
import os
import re
from numpy.random import rand, normal
import scipy
from scipy.spatial.transform import Rotation as R
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json
from itertools import combinations, compress

# Bespoke modules
from gen_packmol import *
import gen_packmol
from gen_CG import *
from gen_lammps_init import gen_lammps_init
from conversion import *
from HPS import *

# CLASSES
class AF2LAMMPS():
    '''
    This class reads alphafold cif and json files and classifies them as rigid, paired rigid, and disordered 
    '''
    def __init__(self,name,cif_file,json_file):
        self.name = name
        self.cif_file = cif_file
        self.json_file = json_file
    
    
    # Functions
    def read_coords(self):
        """
        Read coordinates from cif file
        """
        cif_reader = PDB.MMCIFParser()
        self.structure = cif_reader.get_structure(self.name,self.cif_file)
        self.coords = np.array([res['CA'].get_vector()[:] for res in self.structure.get_residues()])
        self.sequence = ''.join([three_to_one[x.resname] for x in self.structure.get_residues()])
        self.N = len(self.sequence)
        return self.structure, self.coords, self.sequence, self.N
    
    def get_plDDT(self):
        """
        Read plDDT from cif file
        """
        cif = MMCIF2Dict(self.cif_file)
        self.plDDT = np.array(cif['_ma_qa_metric_local.metric_value'],dtype=float)
        return cif, self.plDDT
    
    def get_pae(self):
        """
        Load PAE data givin json file
        """
        pae_f = open(self.json_file)
        data = json.load(pae_f)[0]
        self.pae = np.array(data['predicted_aligned_error'])
        #for i in range(self.N**2):
            #self.pae[int(data['residue1'][i])-1,int(data['residue2'][i])-1] = float(data['distance'][i])
        return self.pae
    
    def get_reg(self,config_thresh=70,lthresh=3,dthresh=10):
        """
        Return 0 index of regions that are ordered and disordered
        """
        # find amino acids where confidence is >= 70
        rig_idx = np.where(self.plDDT >= config_thresh)[0]
        diff = np.diff(rig_idx)
        rig_new = []

        # connect rigid segments less than lthresh in distance
        for i,idx in enumerate(rig_idx[:-1]):
            if (diff[i] > 1) and (diff[i] <= lthresh):
                rig_new.extend(np.arange(idx,rig_idx[i+1]).tolist())
            else:
                rig_new.append(idx)
        rig_new.append(rig_idx[-1])
        rig_new = np.array(rig_new)
        
        # remove rigid regions that are less than dthresh in length
        mask = rig_new - np.arange(len(rig_new))
        self.rig_segs = [rig_new[np.where(mask==x)[0]] for x in np.unique(mask) if len(np.where(mask==x)[0]) >= dthresh]
        pol_segs = np.setdiff1d(np.arange(self.N),np.concatenate(self.rig_segs))
        mask1 = pol_segs - np.arange(len(pol_segs))
        self.pol_segs = [pol_segs[np.where(mask1==x)[0]] for x in np.unique(mask1)]
        return self.rig_segs, self.pol_segs
    
    def get_paired_rig(self,pae_thresh=10):
        """
        Find rigid segments whose relative positions are correlated
        """
        self.paired_rig = []
        for combo in combinations(np.arange(len(self.rig_segs)),2):
            seg1 = self.rig_segs[combo[0]]
            seg2 = self.rig_segs[combo[1]]
            pae1 = self.pae[seg1[0]:seg1[-1],seg2[0]:seg2[-1]]
            pae2 = self.pae[seg2[0]:seg2[-1],seg1[0]:seg1[-1]]
            if np.mean([np.mean(pae1),np.mean(pae2)]) < pae_thresh:
                self.paired_rig.append(combo)
        
        # iteratively find paired segments
        def merge(paired,res=None):
            if paired == []:
                return []
            if res == None:
                res = [paired[0]]

            for p in paired[1:]:
                for r in res:
                    if len(set(r).intersection(set(p))) > 0:
                        r.extend(list(set(p).difference(r)))
                        break
                else:
                    res.append(p)
            if paired == res:
                return res
            else:
                return merge(res)
            
        self.paired_rig = merge(self.paired_rig)
        return self.paired_rig

# FUNCTIONS
def gen_c(c,r,n=10):
    """
    generate random coordinate which is a distnace r from coordinate c
    
    INPUT
    -----
    c = 3D refernce coordinate
    r = distance between c and new coordinates
    n = (optional) int number of coordinates to generate
    
    OUTPUT
    ------
    n x 3 array of new coordinates
    """
    # generate random theta and pi
    phi =  np.random.uniform(0,2 * np.pi,n)
    costheta = np.random.uniform(-1,1,n)
    theta = np.arccos(costheta)
    
    return np.array([c[0] + r * np.cos(phi) * np.sin(theta),
                     c[1] + r * np.sin(phi) * np.sin(theta),
                     c[2] + r* np.cos(theta)])

def ovecs(v):
    '''
    Find two mutually orthogonal unit vectors orthogonal to v
    '''
    # find direction vector of v
    n = v / np.linalg.norm(v)
    
    # rotate vector 90 degrees:
    if abs(n[0]) < 1e-10 and  abs(n[1]) < 1e-10:
        o1 = np.array([n[2],0,-n[0]])
    if abs(n[1]) < 1e-10 and abs(n[2]) < 1e-10:
        o1 = np.array([0,n[2],-n[1]])
    if abs(n[2]) < 1e-10 and abs(n[0]) < 1e-10:
        o1 = np.array([n[1],-n[0],0])
    else:
        o1 = np.array([n[1],-n[0],0])
        
    # find second orthogonal vector by cross product
    o2 = np.cross(o1,v)
    
    return o1/np.linalg.norm(o1),o2/np.linalg.norm(o2)

def sphere_intersect(r1,c1,r2,c2):
    """
    Find 50 unfiformly distributed points that are the intersection of two sphers
    
    INPUT
    -----
    r1 = float radius of circle 1
    c1 = 3d coordinate of center of circle 1
    r2 = float radius of circle 2
    c2 = 3d coordinate of center of circle 2
    
    OUTPUT
    ------
    50 x 3 array of points on intersecting circle
    
    """
    d = np.sqrt(np.sum((c2 - c1)**2))
    t = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(r1**2 - t**2)
    n = (c2-c1) / np.linalg.norm(c2-c1)
    center = c1 + t * n
    o1,o2 = ovecs(c2-c1)
    thetas = np.linspace(0,2*np.pi)
    
    intersects = np.array([center + h * (o1 * np.cos(theta) + o2 * np.sin(theta)) for theta in thetas])
    return intersects.reshape(50,3)

def dcheck(x,length,end,thresh=1):
    """
    Helper function used to find if a coordinate is less than length / thresh away from end.
    
    INPUT
    -----
    x = coordinate of query point
    length = maximum length between x and end
    end = coordinate of end point
    thresh = length treshold of checking function
    
    OUTPUT
    ------
    True if dist(x,end) is less than length
    """
    current_dist = pairwise_distances([x],[end])
    if length < current_dist * thresh:
        return False
    else:
        return True
    
    
def ocheck(x,idx,setpoint_idx,atom_radii,coords):
    """
    helper function used in walk_f and walk_b to find overlapping atoms
    
    INPUT
    -----
    x = coordinate of query point
    idx = sequence index of x
    setpoint_idx = sequence indices of coords
    atom_radii = float list of radii in entire sequence
    coords = N x 3 coordinates to avoid
    
    OUTPUT
    ------
    True if all pairwise distances are > minimum overlap distance given by sum of radii of each particle
    """
    min_dist = [atom_radii[idx] + atom_radii[atom] for atom in setpoint_idx]
    all_dist = pairwise_distances(x,coords)
    if np.any(all_dist < min_dist):
        return False
    else:
        return True
    
def ocheck1(x,r_x,coords,r_avoid):
    """
    helper function used in walk_f1 and walk_b1 to find overlapping atoms
    
    INPUT
    -----
    x = coordinate of query point
    r_x = radius of x
    coords = N x 3 coordinates to avoid
    r_avoid = radii of coordinates to avoid
    
    OUTPUT
    ------
    True if all pairwise distances are > minimum overlap distance given by sum of radii of each particle
    """
    min_dist = [r_x + r for r in r_avoid]
    all_dist = pairwise_distances(x,coords)
    if np.any(all_dist < min_dist):
        return False
    else:
        return True
    
def walk_f(paths,cpath,atom_radii,end,ocheck_args):
    '''
    Recursive function for generating random walk with set start and end nodes. walk_f creates new coordinates while walk_b
    backtracks. 
    
    INPUT
    -----
    paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    atom_radii = float list of atom radii
    end = list of last coordinate
    
    OUTPUT
    ------
    paths =  paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    '''
    
    # find index of current node
    for i,path in enumerate(paths):
        if path == 0:
            break
    
    # get length of remaining polymer
    rlength = np.sum(atom_radii[i:])

    # generate grid of candidate coordinates
    opts = gen_c(cpath[i-1],(atom_radii[i-1] + atom_radii[i]),10).T
    
    # get indices of coordinates to avoid
    if i == 1:
        to_avoid = [-1]
    if i > 1:
        to_avoid = [-1] + [pos for pos in range(i-1)]
    
    r_x = atom_radii[i]
    coords = cpath[:i-1] + [end] + ocheck_args[0]
    r_avoid = [atom_radii[q] for q in to_avoid] + ocheck_args[1] 
    
    # filter candidate coordinates via volume exclusion and distance from final coordinate
    new_path = [opt for opt in opts if dcheck(opt,rlength,end) and ocheck1([opt.tolist()],r_x,coords,r_avoid)]    
    
    # end recursion if at penultimate position
    if i == len(paths)-2:
        
        # get indices of coordinates to avoid
        to_avoid = [pos for pos in range(i-1)]
        
        # find circle on which penultimate position occurs
        opts1 = sphere_intersect(atom_radii[-3] + atom_radii[-2],
                                np.array(cpath[-3]),
                                atom_radii[-2] + atom_radii[-1],
                                end)
        
        # check for overlaps
        r_x = atom_radii[i]
        coords = cpath[:i-1] + ocheck_args[0]
        r_avoid = [atom_radii[q] for q in to_avoid]  + ocheck_args[1] 
        n_path = [o for o in opts1 if ocheck1([o.tolist()],r_x,coords,r_avoid)]
        
        if n_path:

            # fill penultimate node
            c = np.random.choice(len(n_path))
            cpath[i] = n_path[c].tolist()
            paths[i] = [opt.tolist() for opt in n_path]
            
            # fill in last node
            cpath[-1] = end
            paths[-1] = end
            return paths, cpath
        else:
            
            return walk_b(paths,cpath,atom_radii,end,ocheck_args)
        
    
    # randomly choose filtered coordinates
    if new_path:
        choice = np.random.choice(len(new_path))
        cpath[i] = new_path[choice].tolist()
        paths[i] = [opt.tolist() for opt in new_path]
        return walk_f(paths,cpath,atom_radii,end,ocheck_args)
    
    # if no options, backtrack
    else:
        return walk_b(paths,cpath,atom_radii,end,ocheck_args)
    
def walk_f1(paths,cpath,atom_radii,ocheck_args):
    '''
    Recursive function for generating random walk with set start node. walk_f creates new coordinates while walk_b
    backtracks.
    
    INPUT
    -----
    paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    atom_radii = float list of atoms
    
    OUTPUT
    ------
    paths =  paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    '''
    
    # find index of current node
    for i,path in enumerate(paths):
        if path == 0:
            break
    # get length of remaining polymer
    rlength = np.sum(atom_radii[i:])

    # generate grid of candidate coordinates
    opts = gen_c(cpath[i-1],(atom_radii[i-1] + atom_radii[i]),10).T
    
    # get indices of coordinates to avoid
    if i == 1:
        o_checked = [o for o in opts if ocheck1([o],atom_radii[1],ocheck_args[0],ocheck_args[1])]
        
        if len(o_checked) == 0:
                return walk_f1(paths,cpath,atom_radii,ocheck_args)
        choice = np.random.choice(len(o_checked))
        cpath[i] = o_checked[choice].tolist()
        paths[i] = [opt.tolist() for opt in o_checked]
        
        if len(paths) > 2:
            return walk_f1(paths,cpath,atom_radii,ocheck_args)
        
        else:
            return paths, cpath
    if i > 1:
        to_avoid = [pos for pos in range(i-1)]
    
    r_x = atom_radii[i]
    
    coords = cpath[:i-1] + ocheck_args[0]
    r_avoid = [atom_radii[q] for q in to_avoid] + ocheck_args[1] 
    
    # filter candidate coordinates via volume exclusion and distance from final coordinate
    new_path = [opt for opt in opts if ocheck1([opt.tolist()],r_x,coords,r_avoid)]     
        
    
    # randomly choose filtered coordinates
    if new_path:
        choice = np.random.choice(len(new_path))
        cpath[i] = new_path[choice].tolist()
        paths[i] = [opt.tolist() for opt in new_path]
        
        if i == len(paths)-1:
            return paths, cpath
        else:
            return walk_f1(paths,cpath,atom_radii,ocheck_args)
    
    # if no options, backtrack
    else:
        return walk_b1(paths,cpath,atom_radii,ocheck_args)
    
def walk_b1(paths,cpath,atom_radii,ocheck_args):
    '''
    Recursive function for generating random walk with set start and end nodes. walk_f1 creates new coordinates while walk_b1
    backtracks.
    
    INPUT
    -----
    paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    atom_radii = float list of atom radii
    end = list of last coordinate
    
    OUTPUT
    ------
    paths =  paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    '''
    # find index of current node
    for i,path in enumerate(paths):
        if path == 0:
            break
            
    # restart if we are at start
    if i == 1:
        return walk_f1(paths,cpath,atom_radii,ocheck_args)
    
    # remove path if only one option
    if len(paths[i-1]) == 1:
        paths[i-1] = 0
        cpath[i-1] = 0
        return walk_b1(paths,cpath,atom_radii,ocheck_args)
    # re-select path if more than one option
    else:
        paths[i-1].remove(cpath[i-1])
        choice = np.random.choice(len(paths[i-1]))
        cpath[i-1] = paths[i-1][choice]
    return walk_f1(paths,cpath,atom_radii,ocheck_args)

def walk_b(paths,cpath,atom_radii,end,ocheck_args):
    '''
    Recursive function for generating random walk with set start and end nodes. walk_f creates new coordinates while walk_b
    backtracks.
    
    INPUT
    -----
    paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    atom_radii = float list of atom radii
    end = list of last coordinate
    
    OUTPUT
    ------
    paths =  paths = list of list of candidate coordinates
    cpath = list of coordinates of current path
    '''
    # find index of current node
    for i,path in enumerate(paths):
        if path == 0:
            break
            
    # restart if we are at start
    if i == 1:
        return walk_f(paths,cpath,atom_radii,end,ocheck_args)
    
    # remove path if only one option
    if len(paths[i-1]) == 1:
        paths[i-1] = 0
        cpath[i-1] = 0
        return walk_b(paths,cpath,atom_radii,end,ocheck_args)
    # re-select path if more than one option
    else:
        paths[i-1].remove(cpath[i-1])
        choice = np.random.choice(len(paths[i-1]))
        cpath[i-1] = paths[i-1][choice]
    return walk_f(paths,cpath,atom_radii,end,ocheck_args)

def IDR_stats(L):
    '''
    Gives empirical statistics (mean, std) of the end to end distance of a self-avoiding random walk with beads of radius 3.81. 
    NOTE: USE WITH CAUTION. PARAMETERS ARE FROM OUR OWN ANALYSIS AND THUS NOT GENERAL.
    
    INPUT
    -----
    L = length (number of monomers in chain)
    
    OUTPUT
    ------
    mu: float mean end-to-end distance
    std: float std of end-to-end distance
    '''
    def func(x,a,b):
        return a * x ** b
    
    return func(L,4.67,0.53), func(L,1.26,0.61)

def pick_length(mu,sig,N):
    '''
    Picks a gaussian distributed end-to-end length between two points connected by an IDR of N monomers with bead radius 3.81
    NOTE: USE WITH CAUTION. PARAMETERS ARE FROM OUR OWN ANALYSIS AND THUS NOT GENERAL.
    '''
    mu,sig = IDR_stats(N+2)
    L = np.random.randn(10000) * sig + mu
    if len(L[(L < ((N+1) * 3.81)) & (L > 3.81)]) == 0:
        return pick_length(mu,sig,N)
    else:
        return L[(L < ((N+1) * 3.81)) & (L > 3.81)][0]
    
def rig_ends(rig,coords):
    """
    Helper function to find configuration for structured regions of a protein.
    
    INPUT
    -----
    rig: list of np array storing (90-indexed) indices of continous rigid segments
    coords: list of Nx3 array storing coordinates of each rigid region
    """
    
    # get start and end coordinates of paired regions
    start_end = [0 for x in range(len(rig))]
    for i, seg in enumerate(rig):
        start_end[i] = [coords[seg[0]],coords[seg[-1]]]
    return start_end

def get_rig_dist(rig,coords,paired,radii):
    """
    Helper function to find pairwise maximum distance betwen rigid segments that are paired or neighbors in the linear
    sequence.
    
    INPUT
    -----
    rig = list of np array storing (0-indexed) indices of continous rigid segments
    coords = N x3 np array of coordinates of full protein
    paired = list of tuples of indices of paired rigid segments
    radii = N sequence of radii of amino acids of full protein
    
    OUTPUT
    ------
    rig_dist = W x W np array of max distances, where 0 means no constraint. W = len(rig)
    """
    
    # get coordinate of rigid ends
    rig_lim = rig_ends(rig, coords)
    
    # find max distance constraint
    rig_dist = np.zeros((len(rig),len(rig)))
    for i in range(len(rig_dist)):
        for j in range(len(rig_dist)):
            i0 = rig[min(i,j)][-1]
            i1 = rig[max(i,j)][0]
            if (i,j) in paired or (j,i) in paired:
                rig_dist[i,j] = np.sqrt(np.sum((coords[i0] - coords[i1]) ** 2))
            elif j == i+1 or i == j+1:
                rig_dist[i,j] = np.sum(radii[i0+1:i1])
                
    return rig_dist

def get_pmat(rig,paired):
    """
    Helper function for generating boolean matrix for paired rigid segments.
    
    INPUT
    -----
    rig = list of np array storing (0-indexed) indices of continous rigid segments
    paired = list of tuples of indices of paired rigid segments
    
    OUTPUT
    ------
    rig_dist = W x W np boolean array, where 1 means paired. W = len(rig) 
    """
    pmat = np.zeros((len(rig),len(rig)))
    for x in paired:
        for i in combinations([*x], 2):
            pmat[i] = 1
            pmat[i[::-1]] = 1
    return pmat

def get_random_rot(nr=1000):
    '''
    Helper function for creating a list of random scipy.transform.rotation.Rotation objects from quaternions
    
    INPUT
    -----
    nr = int number of random objects
    
    OUTPUT
    ------
    list random scipy.transform.rotation.Rotation objects
    '''
    rs = []
    for i in range(nr):
        q = np.random.normal(size=(4))
        q /= np.linalg.norm(q)
        r = R.from_quat(q)
        rs.append(r)
    return rs

def place_rigid(coords,rig,paired,radii,nr=1000):
    '''
    Function for generating new orientations of proteins with partial disorder.
    
    INPUT
    -----
    coords = N x 3 np array of coordinates of full protein
    rig = list of np array storing (0-indexed) indices of continous rigid segmentss
    paired = list of tuples of indices of paired rigid segments
    radii = sequence of protein radii
    nr = (optional) number of random rotations to generate for each new rigid section
    
    OUTPUT
    ------
    c_rig = list of np arrays corresponding to rigd segments of the protein
    idx_placed = list of (0-indexed) indices that were placed
    '''
    
    # get rigid region parameters
    N = len(coords)
    rig_lim = rig_ends(rig,coords)
    rig_dist = get_rig_dist(rig,coords,paired,radii)
    pmat = get_pmat(rig,paired)
    
    # initialize output
    c_rig = [0 for x in range(len(rig))]
    idx_placed = []
    rig_placed = []
    
    # iterate through contiguous rigid regions
    for i in range(len(rig)):

        # set first rigid segment to origin
        if i == 0:

            # find pairs, translate first coordinate to origin
            _pairs = np.where(pmat[i] == 1)[0]
            c_rig[0] = coords[rig[0]] - rig_lim[0][0]
            for _p in _pairs:
                c_rig[_p] = coords[rig[_p]] - rig_lim[0][0]

            # book-keeping for indixes, segments placed
            rig_placed.extend(_pairs)
            idx_placed.extend(np.concatenate((rig[i],*[rig[x] for x in _pairs])))
            rig_placed.append(0)
        
        # set next rigid segments
        if i != 0:

            # skip if already placed
            if i in rig_placed:
                continue

            else:

                # find paired values
                _pairs = np.where(pmat[i] == 1)[0]
                to_trans = np.concatenate((rig[i],*[rig[x] for x in _pairs]))
                curr_rig = [i]
                curr_rig.extend(_pairs.tolist())

                # find length restrictions
                dbound = np.ones((len(curr_rig),2)) * np.inf
                for s,x in enumerate(curr_rig):

                    if x-1 in rig_placed:
                        dbound[s,0] = rig_dist[x,x-1]
                    if x+1 in rig_placed:
                        dbound[s,1] = rig_dist[x,x+1]

                # generate random first coordinate of current rigid segment
                flex_len = rig[i][0] - rig[i-1][-1] - 1
                l = pick_length(*IDR_stats(flex_len+2),flex_len)
                opts = gen_c(c_rig[i-1][-1],l,n=10)

                # check first coordinate for overlaps
                o_checked = [o for o in opts.T if ocheck1([o],
                                                          radii[rig[i][0]],
                                                          np.concatenate([c_rig[pl] for pl in rig_placed]),
                                                          [radii[j] for j in idx_placed])]
                
                # if no options, rerun function
                if len(o_checked) == 0:
                    return place_rigid(coords,rig,paired,radii,nr=nr)

                # generate random orientation of rigid body
                possible_coords = np.zeros((nr,len(curr_rig),2,3))
                choice = np.random.choice(np.arange(len(o_checked)))
                rs = get_random_rot(nr=nr)
                for _idx,r in enumerate(rs):
                    for iicx, icx in enumerate(curr_rig):
                        v = r.apply(np.array(rig_lim[icx]))
                        possible_coords[_idx,iicx,:] = v-v[0]+o_checked[choice]

                # get reference coordinates from which a max distance is defined for the current segment
                ref = np.zeros((len(curr_rig),2,3))
                for l, icr in enumerate(curr_rig):
                    if icr-1 in rig_placed:
                        ref[l,0] = c_rig[icr-1][-1]
                    if icr+1 in rig_placed:
                        ref[l,1] = c_rig[icr+1][-1]

                # find minimum pairwise distance
                pdist_min = np.zeros((len(to_trans),len(idx_placed)))
                for l in range(len(to_trans)):
                    for m in range(len(idx_placed)):
                        pdist_min[l,m] = radii[to_trans[l]] + radii[idx_placed[m]]

                # place new coordinates and check min and max distance constraints
                checked = False
                for icheck, _rs in enumerate(rs):
                    comp = []
                    for _icr,r in enumerate(curr_rig):
                        
                        # apply rotation and translate to chosen initial coordinate
                        v = _rs.apply(coords[rig[r]])
                        if _icr == 0:
                            _start = v[0]
                        v = v - _start + o_checked[choice]
                        comp.append(v)
                        
                        # check that ends are below max distance from adjacent rigid segments
                        _d = np.sqrt(np.sum((np.array([v[0],v[-1]]) - ref[_icr]) ** 2,axis=1))
                        c_rig[r] = v
                        if np.all(dbound-_d > -1e-10):
                            checked = True
                    
                    # check overlaps between current segment and placed segments
                    dists = pairwise_distances(np.concatenate(comp), np.concatenate([c_rig[rp] for rp in rig_placed]))
                    if np.all(dists > pdist_min) and checked:
                        break
                    
                    # if no options, rerun function
                    if icheck == nr-1:
                        return place_rigid(coords,rig,paired,radii,nr=nr)

                # book-keeping for indices, segments placed
                rig_placed.extend(curr_rig)
                idx_placed.extend(np.concatenate((rig[i],*[rig[x] for x in _pairs])))
                
    return c_rig,idx_placed

def place_brownian_brige(iflex,flex,c_rig,c_placed, idx_placed,radii):
    """
    Helper Function for generating coordinates of a brownian bridge between successive rigid regions of proteins
    
    INPUT
    -----
    iflex = int index of current flexible region
    flex = list of np arrays corresponding to (0-indexed) indices of flexible regions
    c_rig = list of np arrays corresponding to rigd segments of the protein
    c_placed = list of list of coordinates placed matching index of idx_placed
    idx_placed = list of (0-indexed) indices that were placed
    radii = np sequence of amino acid radii for entire protein sequence
    
    OUTPUT
    ------
    paths = tree of coordinates lists generated in recursive search process
    cpath = list of list of coordinates of IDR, including fixed ends
    idx_placed = updated list of indices placed
    """
    
    # get coordinates of fixed ends
    start = c_rig[iflex-1][-1]
    end = c_rig[iflex][0]
    
    # initalize search path
    paths = [list(start)] + [0 for x in range(len(flex[iflex])+1)]
    cpath = [list(start)] + [0 for x in range(len(flex[iflex])+1)]
    r_avoid = [radii[x] for x in idx_placed]
    idr_radii = radii[np.arange(flex[iflex][0]-1,flex[iflex][-1]+2)]
    
    try:    
        # generate brownian brige
        paths, cpath = walk_f(paths,cpath,idr_radii,end,[c_placed,r_avoid])
        idx_placed.extend(flex[iflex])
        c_placed.extend(cpath[1:-1])
        return paths,cpath,c_placed,idx_placed
    except:
        return place_brownian_brige(iflex,flex,c_rig,c_placed, idx_placed,radii)
    
def place_flex_start(flex,c_rig,idx_placed,radii):
    """
    Helper function for generating coordinates of flexible region before the first rigid region
    
    INPUT
    -----
    flex = list of np arrays corresponding to (0-indexed) indices of flexible regions
    c_rig = list of np arrays corresponding to rigd segments of the protein
    c_placed = list of list of coordinates placed matching index of idx_placed
    idx_placed = list of (0-indexed) indices that were placed
    radii = np sequence of amino acid radii for entire protein sequence
    
    OUTPUT
    ------
    paths = tree of coordinates lists generated in recursive search process
    cpath = list of list of coordinates of IDR, including fixed end
    c_placed = updated list of lists of coordinates placed matching index of idx_placed
    idx_placed = updated list of indices placed
    """
    
    # initialize tree for recursive search
    iflex = 0
    ff = flex[iflex]
    ffn = len(ff)
    paths = [[0,0,0]] + [0 for x in range(ffn)]
    cpath = [[0,0,0]] + [0 for x in range(ffn)]
    
    # get parameters for volume exclusion check
    atom_radii = radii[np.arange(0,ff[-1]+2)]
    r_avoid = [radii[x] for x in idx_placed]
    c_placed = np.concatenate(c_rig).tolist()
    # place coordinates
    paths, cpath = walk_f1(paths,cpath,atom_radii[::-1],[c_placed,r_avoid])
    idx_placed.extend(ff)
    c_placed.extend(cpath[1:][::-1])
    return paths[::-1], cpath[::-1],c_placed,idx_placed

def place_flex_end(iflex,flex,c_rig,c_placed,idx_placed,radii):
    """
    Helper function for generating coordinates of flexible region after last fixed region
    
    INPUT
    -----
    flex = list of np arrays corresponding to (0-indexed) indices of flexible regions
    c_rig = list of np arrays corresponding to rigd segments of the protein
    c_placed = list of list of coordinates placed matching index of idx_placed
    idx_placed = list of (0-indexed) indices that were placed
    radii = np sequence of amino acid radii for entire protein sequence
    
    OUTPUT
    ------
    c_placed = updated list of lists of coordinates placed matching index of idx_placed
    idx_placed = updated list of indices placed
    """
    
    # initialize tree for recursive search
    ff = flex[iflex]
    ffn = len(ff)
    paths = [c_rig[-1][-1].tolist()] + [0 for x in range(ffn)]
    cpath = [c_rig[-1][-1].tolist()] + [0 for x in range(ffn)]

    # get parameters for volume exclusion check
    atom_radii = radii[np.arange(ff[0]-1,ff[-1]+1)]
    r_avoid = [radii[x] for x in idx_placed]

    # place coordinates
    paths, cpath = walk_f1(paths,cpath,atom_radii,[c_placed,r_avoid])
    idx_placed.extend(ff)
    c_placed.extend(cpath[1:])
    return paths, cpath,c_placed,idx_placed

def generate_protein(coords,rig,flex,paired,radii,nr=1000):
    '''
    Function for generating random coordinates for proteins with partial disorder.
    
    INPUT
    -----
    coords = N x 3 np array of coordinates of full protein
    rig = list of np array storing (0-indexed) indices of continous rigid segments
    flex = list of np array storing (0-indexed) indices of continous rigid segments
    paired = list of tuples of indices of paired rigid segments
    radii = np sequence of protein radii
    nr = (optional) number of random rotations to generate for each new rigid section
    
    OUTPUT
    ------
    c_placed = N x 3 output coordinates of generation
    idx_placed = index order of coordinate output
    '''
    # resolve fully disordered case
    N = len(coords)
    if len(rig) == 0:
        print('fully disordered')
        paths = [[0,0,0]] + [0 for x in range(N-1)]
        cpath = [[0,0,0]] + [0 for x in range(N-1)]
        paths, cpath = walk_f1(paths,cpath,radii,[[[0,0,0]],[radii[0]]])
        return [] ,cpath
    
    else:  
        print('placing rigid regions')
        c_rig,idx_placed = place_rigid(coords,rig,paired,radii,nr=nr)
        c_placed = np.concatenate(c_rig).tolist()
        
    # place polymeric regions
    c_flex = [0 for x in range(len(flex))]
    
    for iflex in range(len(flex)):
        
        # check if protein begins with flexible region
        if flex[iflex][0] == 0:
            print('placing first IDR')
            _, c_flex[iflex],c_placed,idx_placed = place_flex_start(flex,c_rig,idx_placed,radii)
            continue
        
        # check if protein ends with flexible region
        if flex[iflex][-1] == len(coords) -1:
            print('placing final IDR')
            if flex[iflex][-1] == len(coords) -1:
                _, c_flex[iflex],c_placed,idx_placed = place_flex_end(iflex,flex,c_rig,c_placed,idx_placed,radii)
                continue
            else:
                pass

        # place flexbile region between structured regions
        else:
            print('placing IDR')
            _, c_flex[iflex],c_placed,idx_placed = place_brownian_brige(iflex,flex,c_rig,c_placed,idx_placed,radii)
    
    return idx_placed,c_placed