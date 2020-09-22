# coding: utf-8
# # Beckermann 1999 model

# This notebook will do coupled fluid-solid melting with a phase field method mixed with volume penalisation
import numpy as np
import dedalus.public as de 
from dedalus.extras import flow_tools
import time
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
import logging
logger = logging.getLogger(__name__)

from os.path import join
from dedalus.tools import post
 
def sigmoid(x,a=1): return 0.5*(np.tanh(x/a)+1)
d = de.operators.differentiate
from dedalus.core.operators import GeneralFunction
import seawater as sw

# Dimensional parameters
U = 3.5
T_B = 20 # C
c_p = 4.2 # J/g*C
L_T = 3.34e2 # J/g
C_B = 20 # g/kg
ν = 1.3e-2 # cm**2/s
κ = 1.3e-3 # cm**2/s
μ = 1.3e-4 # cm**2/s
m = 0.056 # C/(g/kg)
L,H = 50,15
l,h = 30,3

Re= 1/ν
Pr= 7
S = L_T/(c_p*T_B)
δ = 1e-4
ϵ = 1e-2
N = (m*C_B)/T_B
A = ϵ*(S*Re*Pr)*(5/6)
G = ϵ 
β = 4/2.648228
η = 1e-1*Re*(β*ϵ)**2 # not optimal

# Seawater equation of state Buoyancy funtion
rho0 = 1000
drho = 0 # density melt - density0
B = (-10)*100*drho/rho0

# Save parameters
Nx,Nz = 3*4096//2,3*1024//2
dt = 2e-4
sim_name = sys.argv[1]
restart = int(sys.argv[2])
steps = 2000000
save_freq = 2500
save_max = 20
print_freq = 200
wall_time = 23*60*60
save_dir = '.'

# Domain
xbasis = de.Fourier('x', Nx, interval=(0,L), dealias=3/2)
zbasis = de.SinCos('z',Nz, interval=(0,H), dealias=3/2)
domain = de.Domain([xbasis,zbasis],grid_dtype=np.float64)
x,z = domain.grids(domain.dealias)
xx, zz = x + 0*z, 0*x + z
kx, kz = domain.elements(0), domain.elements(1)
# Wall penalty boundary
wall = domain.new_field()
wall.set_scales(domain.dealias)
wall.meta['z']['parity'] = 1
wall['g'] = sigmoid(-(x-0.02*L),a=2*ε)+sigmoid(x-.98*L,a=2*ε)
wall['c'] *= np.exp(-kx**2/5e6) # spectral smoothing

# Buoyancy multiplier for parity constraints
par = domain.new_field() 
par.set_scales(domain.dealias)
par.meta['z']['parity'] = -1
par['g'] = np.tanh(-(z-H)/.1)*np.tanh(z/.1)
par['c'] *= np.exp(-kx**2/5e6) # spectral smoothing

# Mathematical problem
melting = de.IVP(domain,variables=['u','w','p','T','f'])
melting.meta['u','p','T','f']['z']['parity'] = 1
melting.meta['w']['z']['parity'] = -1

params = [Nx,Nz,S,A,G,N,δ,ε,Re,Pr,η,h,wall,par,U,L,H,B]
param_names = ['Nx','Nz','S','A','G','N','δ','ε','Re','Pr',
               'η','h','wall','par','U','L','H','B']
for param, name in zip(params,param_names):
    melting.parameters[name] = param

melting.substitutions['q'] = 'dz(u) - dx(w)'
# Equations
melting.add_equation("dx(u) + dz(w) = 0",condition='(nx != 0) or (nz != 0)')
melting.add_equation("p = 0",condition='(nx == 0) and (nz == 0)')
melting.add_equation("dt(u) + dx(p) - (1/Re)*dz(q) = - w*q - (f/η)*u - (wall/η)*(u-U)")
melting.add_equation("dt(w) + dz(p) + (1/Re)*dx(q) =   u*q - (f/η)*w - (wall/η)*w + B*par*T")
melting.add_equation("dt(T) - (1/(Pr*Re))*(dx(dx(T)) + dz(dz(T))) - S*dt(f) = - (1-f)*(u*dx(T) + w*dz(T)) + T*(u*dx(f) + w*dz(f)) - (wall/η)*(T-1)")
melting.add_equation("Re*Pr*S*(5/6)*dt(f) - (dx(dx(f)) + dz(dz(f))) = - (1/ε**2)*f*(1-f)*(1-2*f+T)")

# Build timestepper and solver
ts = de.timesteppers.RK222 # CNAB bad, SBDF good, RK only good for BC on acceleration
solver = melting.build_solver(ts)

# Initial conditions
u,w,p,T,f = variables = [solver.state[field] for field in melting.variables]
for field in variables: field.set_scales(domain.dealias)
if restart == 0:
    u['g'] = 0
    w['g'] = 0
    p['g'] = 0
    f['g'] = sigmoid(z-(H-h),a=2*ϵ)*sigmoid(x-(L-l)/2,a=2*ϵ)*sigmoid(-(x-(L+l)/2),a=2*ϵ)
    T['g'] = 1-f['g']
    p['g'] = 0

    import file_tools as flts
    if rank == 0: flts.save_domain('domain-{}.h5'.format(sim_name),domain)

else:
    from glob import glob
    import re
    pattern = 'data-{n}-{r:0>2d}'.format(n=sim_name,r=restart-1)
    save_files = glob('{p}/{p}_s*.h5'.format(p=pattern))
    nums = [int(re.search('.*_s(\d+).h5',f).group(1)) for f in save_files]
    last = np.argmax(nums)
    write,_ = solver.load_state(save_files[last],-1)

# Save configurations
solver.stop_iteration = steps
solver.stop_wall_time = wall_time
solver.stop_sim_time = np.inf

# CFL # Breaks quickly
# CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=2, safety=.5,
#                      max_change=1.5, min_change=0.5, max_dt=η/2,
# threshold=0.1)
# CFL.add_velocities(('u', 'w'))

# # Flow properties # Doesnt work
# flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
# flow.add_property("sqrt(u*u + w*w)", name='Re')

# Save state variables
analysis = solver.evaluator.add_file_handler(join(save_dir,'data-{}-{:0>2d}'.format(sim_name,restart)), iter=save_freq, max_writes=save_max,mode='overwrite')
for task in melting.variables: analysis.add_task(task)
analysis.add_task("integ(T - S*f,'x','z')",name='energy')
analysis.add_task("q")

# Save parameters
parameters = solver.evaluator.add_file_handler(join(save_dir,'parameters-{}-{:0>2d}'.format(sim_name,restart)), iter=np.inf, max_writes=100,mode='overwrite')
for task in melting.variables: parameters.add_task(task)
for name in param_names: parameters.add_task(name)
parameters.add_task("q")

start_time = time.time()
while solver.ok:
    if solver.iteration % print_freq == 0:
        max_speed = u['g'].max()
        logger.info('{:0>6d}, u max {:f}, dt {:.5f}, time {:.2f}'.format(solver.iteration,max_speed,dt,(time.time()-start_time)/60))
        if np.isnan(max_speed): sys.exit(1)
        if f.integrate('x','z')['g'][0,0]<1e-3: break

    solver.step(dt)
solver.step(dt)
