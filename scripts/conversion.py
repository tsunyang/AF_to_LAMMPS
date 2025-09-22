import numpy as np

def LJ_to_SI(mass, length, temperature):
    """
    Converts LJ units to SI units based on 3 fundamental quantities.
    
    INPUT
    -----
    mass = float mass in kg
    length = float distance in m
    temperature = float temperature in K
    
    OUTPUT
    ------
    mass = float mass in kg
    length = distance in m
    time = time in s
    energy = energy in kg m^2 s^-2
    velocity = velocity in m s^-1
    force = force in kg m s^-2
    torque = torque in kg m^2 s^-2
    temperature = temperature in K
    pressure = pressure in kg m^-1 s^-2
    viscosity = dynamic viscosity in kg m^-1 s^-1
    charge = charge in C
    dipole = dipole in C m
    e_field = electric field in N C^-1
    density = density kg m^-3 
    """
    kb = 1.38064852 * 10 ** -23
    eps0 = 8.85418782 * 10 ** -12
    energy = kb * temperature
    time = np.sqrt(mass * length ** 2 / energy)
    velocity = length / time
    force = energy / length
    torque = energy
    pressure = energy / length ** 3
    viscosity = energy * time / length ** 3
    charge = np.sqrt(4 * np.pi * eps0 * length * energy)
    dipole = np.sqrt(4 * np.pi * eps0 * length ** 3 * energy)
    e_field = force / charge
    density = mass / length ** 3
    
    names = 'mass,length,time,energy,velocity,force,torque,temperature,pressure,viscosity,charge,dipole,e_field,density'.split(',')
    values = [mass,length,time,energy,velocity,force,torque,temperature,pressure,viscosity,charge,dipole,e_field,density]
    units = 'kg,m,s,kg m^2 s^-2,m s^-1,kg m s^-2,kg m^2 s^-2,K,kg m^-1 s^-2,kg m^-1 s^-1,C,C m,N C^-1,kg m^-3'.split(',')
    
    for i, v in enumerate(values):
        print(names[i].ljust(20) + "{:.2e}".format(v).ljust(20) + str(units[i]))
    return values

def get_damp(m,d,eta=0.853):
    """
    Get Langevin thermostat damping factor in units of femtoseconds.
    
    INPUT
    -----
    m = mass in g/mol
    d = diameter in Angstrom
    eta = viscosity of water,de fault = 0.853 MPa*s (300K)
    """
    
    return (m / 6.0221408e23/1000)/(3 * np.pi * eta/1e6 * d/1e10) * 1e15