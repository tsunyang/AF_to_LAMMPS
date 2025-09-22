import freud
import scipy.spatial.transform
import scipy.spatial.distance
import numpy as np

def kabsch(coords):
    '''
    Implements the Kabsch algorithm for finding the rmsf after optimal rotation to
    the centroid of a set of coordinates.
    
    INPUT'
    -----
    coords: (N,n,3) np array of N timesteps, n atom positions
    
    OUTPUT
    ------
    all_rotated: (N,n,3) rotated coordinates centered at the centroid
    rmsf: (n,) array of Root mean square fluctuation.
    mean
    '''
    
    # center mean coordinate at centroid
    mean_coord = np.mean(coords,axis=0)
    mean_centroid = np.mean(mean_coord, axis=0)
    mean_coord -= mean_centroid
    
    # center each snapshot at centroid
    for i,c in enumerate(coords):

        # translate to centroid in the given timestep
        coords[i] -= np.mean(c, axis=0)
        
    # Apply optimal rotation via scipy
    all_rotated = np.zeros(np.shape(coords))
    for i in range(len(coords)):
        iter_c = coords[i]
        e_rot,rmsd = scipy.spatial.transform.Rotation.align_vectors(mean_coord,iter_c)
    
        # generate rotation
        all_rotated[i,:,:] = e_rot.apply(iter_c)
        
    # Find RMSF of rotated matrices
    mean_coord_rotated = np.mean(all_rotated,axis=0)
    rmsf = np.mean(np.sqrt(np.sum((all_rotated - mean_coord_rotated) ** 2,axis=2)),axis=0)
        
    return all_rotated, rmsf

def kabsch_align(c1,c2):
    '''
    Implements the Kabsch algorithm for finding optimal rotation and translation between two sets of coordinates in 3D.
    This algorithm aligns c1 to c2
    
    INPUT'
    -----
    c1: (n,3) np array of n atom positions
    c2: (n,3) np array of n atom positions
    
    OUTPUT
    ------
    rot = scipy rotation object for rotating centroid_shifted c1 to c2
    rmsd = root mean square deviation between c1 c2 after kabsch algorithm
    c1_aligned = c1 aligned to c2
    '''
    
    # center both structures at centroid
    c1_centroid = np.mean(c1,axis=0)
    c_cen1 = c1 - c1_centroid
    c2_centroid = np.mean(c2,axis=0)
    c_cen2 = c2 - c2_centroid
    rot,rmsd = scipy.spatial.transform.Rotation.align_vectors(c_cen2,c_cen1)
    return rot,rmsd, rot.apply(c1) - np.mean(rot.apply(c1),axis=0) + c2_centroid

def wrapper(coords,Lx,Ly,Lz):
    """
    wrap a set of coordinates into a box of defined lengths. Assumes coordinates of the box of [0,Lx],
    [0,Ly], and [0,Lz]
    
    INPUT
    -----
    coords = N x 3 array of coordinates
    Lx = float length of box along x axis
    Ly = float length of box along y axis
    Lz = float length of box along z axis
    
    OUTPUT
    ------
    N x 3 array of coordinates wrapped in box
    """
    return freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz).wrap(coords-np.array([Lx,Ly,Lz])/2)+np.array([Lx,Ly,Lz])/2

def translate_closest(coords,ref,Lx,Ly,Lz):
    """
    Translates a set of coordinates based on the displacement between the closest pairwise distance to a set of
    reference coordinates, under assumption of periodic boundary conditions. Assumes coordinates of the box of 
    [0,Lx], [0,Ly], and [0,Lz]
    
    INPUT
    -----
    coords = N x 3 array of coordinates to translate
    ref = M x 3 array of refence coordinates to base closest pairwise distance.
    Lx = float length of box along x axis
    Ly = float length of box along y axis
    Lz = float length of box along z axis
    
    OUTPUT
    ------
    coords_trans = N x 3 coordinates of translated coordinates
    disp = displacement vector for translation
    i1 = indices of coords with minimum pairwise distance with ref
    i2 = indices of ref with minimum pairwise distance with coords
    """
    # generate Freud periodic box instance
    box = freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
    
    # get indices in coords and ref that have minimum pairwise distance across PBC
    d = box.compute_all_distances(coords,ref)
    i1,i2 = np.where(d == np.min(d))
    if len(i1) > 1:
        print('WARNING: degenerate min distance found!')
    
    # get displacement vector between unwrapped and wrapped image of coord yielding closest distance to ref.
    disp = ref[i2[0]] + PBC_disp(ref[i2[0]],coords[i1[0]],np.array([Lx,Ly,Lz])) - coords[i1[0]]
    return coords + disp, disp, i1, i2

def PBC_disp(x0, x1, dimensions):
    """
    Find displacement between two vectors under periodic boundaries.
    
    INPUT
    -----
    x0 = np array of coordinates
    x1 = np array of coordinates
    dimensions = np array of box dimensions
    
    OUTPUT
    ------
    displacement between two vectors under periodic boundary conditions
    """
    disp = x1-x0
    delta = np.abs(disp)
    delta = np.where(delta > 0.5 * dimensions, np.sign(disp)* (delta - dimensions), disp)
    return delta

def COM_dist(ref,trial,ref_masses,trial_masses,Lx,Ly,Lz):
    """
    Find the distance between center of masses of two sets of coordinates under periodic boundary conditions
    
    INPUT
    -----
    ref = N x 3 array of coordinates as reference
    trial = M x 3 array of coordinates as trial
    ref_masses = len(N) list of particles in reference
    trial_masses = len(M) list of particles in reference
    Lx,Ly,Lz = length of simulation box
    
    OUTPUT
    ------
    distance between center of masses of ref and trial
    """
    com_ref, _ = find_res_COM(ref,ref_masses)
    com_trial, _ = find_res_COM(trial,trial_masses)
    
    # generate Freud periodic box instance
    box = freud.box.Box(Lx=Lx,Ly=Ly,Lz=Lz)
    
    return box.compute_all_distances(com_ref,com_trial)[0][0]

def write_lammps_data(out_name,atoms,n_atom_types,xbound,ybound,zbound,masses,n_bond_types=0,bonds=[],n_angle_types=0,angles=[],n_dihedral_types=0,dihedrals=[]):
    """
    Write LAMMPS data file with header and Atoms section
    
    INPUT
    -----
    out_name = str name of path for output
    atoms = np array of information to put under Atoms section. Refer to https://docs.lammps.org/read_data.html
            for format.
    n_atom_types = int number of atom types
    xbound = float list [xmin, xmax] of simulation box
    ybound = float list [ymin, ymax] of simulation box
    zbound = float list [zmin, zmax] of simulation box
    masses = float list of masses listed by atom type.
    
    OUTPUT
    ------
    LAMMPS data file
    """
    with open(out_name,'a') as f:
        f.write('# Header\n\n')
        f.write('%i atoms\n'%len(atoms))
        if n_bond_types > 0:
            f.write('%i bonds\n'%len(bonds))
        if n_angle_types > 0:
            f.write('%i angles\n'%len(angles))
        if n_dihedral_types > 0:
            f.write('%i dihedrals\n'%len(dihedrals))
        f.write('\n')
        
        f.write('%i atom types\n'%n_atom_types)
        if n_bond_types > 0:
            f.write('%i bond types\n'%n_bond_types)
        if n_angle_types > 0:
            f.write('%i angle types\n'%n_angle_types)
        if n_dihedral_types > 0:
            f.write('%i dihedral types\n'%n_dihedral_types)
        f.write('\n')
        f.write('%f %f xlo xhi\n'%(xbound[0],xbound[1]))
        f.write('%f %f ylo yhi\n'%(ybound[0],ybound[1]))
        f.write('%f %f zlo zhi\n\n'%(zbound[0],zbound[1]))
        f.write('Masses\n\n')
        for i,m in enumerate(masses):
            f.write('%i %f\n'%(i+1,m))
        f.write('\nAtoms\n\n')
        for i in atoms:
            f.write(' '.join([str(x) for x in i])+'\n')
                    
        if n_bond_types > 0:            
            f.write('\nBonds\n\n')
            for i in bonds:
                f.write(' '.join([str(x) for x in i])+'\n')
                    
        if n_angle_types > 0:            
            f.write('\nAngles\n\n')
            for i in angles:
                f.write(' '.join([str(x) for x in i])+'\n')
        
        if n_dihedral_types > 0:            
            f.write('\nDihedrals\n\n')
            for i in dihedrals:
                f.write(' '.join([str(x) for x in i])+'\n')
            
def get_damp(m,d,eta=0.853):
    """
    Get Langevin thermostat damping factor in units of femtoseconds.
    
    INPUT
    -----
    m = mass in g/mol
    d = diameter in Angstrom
    eta = viscosity of water,de fault = 0.853 MPa*s (300K)
    
    OUTPUT
    ------
    damping factor (fs)
    """
    
    return (m / 6.0221408e23/1000)/(3 * np.pi * eta/1e6 * d/1e10) * 1e15

def get_largest_dist(coords):
    """
    Get largest pairwise distance between two points in a set of coordinates
    
    INPUT
    -----
    coords = N x 3 array of points
    
    RETURNS
    -------
    max_d = max distance
    idxs = indices of farthest points
    """
    return np.max(cdist(coords,coords)),np.where(cdist(coords,coords) == np.max(cdist(coords,coords)))[0]


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
        r = scipy.spatial.transform.Rotation.from_quat(q)
        rs.append(r)
    return rs

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


def check_in_box(coords,box_x,box_y,box_z):
    """
    Check if the coordinates are in a box of specified dimensions
    
    INPUT
    -----
    coords = N x 3 float array of coordiantes of each particle
    box_x = float list of minimum and maximum x coordinate
    box_y = float list of minimum and maximum y coordinate
    box_z = float list of minimum and maximum z coordinate
    
    OUTPUT
    ------
    True if all coordinates in box. False otherwise
    """
    checkx = np.all((coords[:,0] > box_x[0]) & (coords[:,0] < box_x[1]))
    checky = np.all((coords[:,1] > box_y[0]) & (coords[:,1] < box_y[1]))
    checkz = np.all((coords[:,2] > box_z[0]) & (coords[:,2] < box_z[1]))
    if checkx & checky & checkz:
        return True
    else:
        return False

def check_overlap(c1,rad1,c2,rad2):
    """
    Check if two sets of coordinates are overlapping
    
    INPUT
    -----
    c1 = N1 x 3 array of coordinates
    c2 = N2 x 3 array of coordiantes
    rad1 = 1 x N1 array of radii of coordinates in set 1
    rad2 = 1 x N2 array of radii of coordiantes in set 2.
 
    OUTPUT
    ------
    True if all coordinates are not overlapping. False otherise.ß
    """
    r1,r2 = np.meshgrid(rad1,rad2)
    mins = r1+r2
    d = scipy.spatial.distance.cdist(c1,c2)
    return np.all(d.T  >= mins)