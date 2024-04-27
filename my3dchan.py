import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def solve_uvw( dt,u,v,w,ut,vt,wt ):

    # x-momentum (final terms land in ucv-centers)
    for k in range(1, ncv_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, nno_x-1):

                # pre-compute useful velocities
                # compute neighbouring CV quantities
                ucv  = u[k,j,i]
                ucvE = u[k,j,i+1]
                ucvW = u[k,j,i-1]
                ucvN = u[k,j+1,i]
                ucvS = u[k,j-1,i]
                ucvF = u[k+1,j,i]
                ucvB = u[k-1,j,i]
                # compute CV corner quantities
                vfNW = v[k,j,i]
                vfSW = v[k,j-1,i]
                vfNE = v[k,j,i+1]
                vfSE = v[k,j-1,i+1]

                wfFW = w[k,j,i]
                wfBW = w[k-1,j,i]
                wfFE = w[k,j,i+1]
                wfBE = w[k-1,j,i+1]

                # convective terms:
                # d(uu)/dx +d(vu)/dy +d(wu)/dz
                conv = 0.0

                # d(uu)/dx
                # interp u in x so it is at E and W faces
                ufE = 0.5*( ucv +ucvE )
                ufW = 0.5*( ucvW +ucv )

                conv += ( ufE*ufE -ufW*ufW ) / dx

                # d(vu)/dy
                # interp v in x so it is at N and S faces
                vfN = 0.5*( vfNW +vfNE )
                vfS = 0.5*( vfSW +vfSE )
                # compute required ucv lengths
                dy = y[j+1] -y[j]
                dyN = y[j+2] -y[j+1]
                dyS = y[j] -y[j-1]
                # interp u in y so it is at N and S faces
                ufN = ( (dyN/2)*ucv +(dy/2)*ucvN ) / (dyN/2 +dy/2)
                ufS = ( (dy/2)*ucvS +(dyS/2)*ucv ) / (dy/2 +dyS/2)

                conv += ( vfN*ufN -vfS*ufS ) / dy

                # d(wu)/dz
                # interp w in x so it is at F and B faces
                wfF = 0.5*( wfFW +wfFE )
                wfB = 0.5*( wfBW +wfBE )
                # interp u in z so it is at F and B faces
                ufF = 0.5*( ucv +ucvF )
                ufB = 0.5*( ucvF +ucv )
                conv += ( wfF*ufF -wfB*ufB ) / dz

                # diffusive terms:
                # d2(u)/dx2 +d2(u)/dy2 +d2(u)/dz2
                diff = 0.0

                # d2(u)dx2 = d/dx(du/dx)
                # d2(u)/dx2
                # compute dudx (lands on E/W faces)
                dudxE = ( ucvE -ucv ) / dx
                dudxW = ( ucv -ucvW ) / dx
                # compute d2udx2
                diff += NU*( dudxE -dudxW ) / dx

                # d2(u)dy2
                # compute dudy (lands on N/S faces)
                PFN = (dyN/2)/(dy/2)
                # see Sundqvist and Veronis (1969) eq 1.3
                dudyN = ( ucvN-(PFN**2)*ucv -(1 -(PFN**2))*ufN ) /\
                        (dyN/2*(1+PFN))
                PFS = (dy/2)/(dyS/2)
                dudyS = (ucv-(PFS**2)*ucvS -(1 -(PFS**2))*ufS) /\
                        (dy/2*(1+PFS))

                diff += NU*( dudyN -dudyS ) / dy

                # d2(u)dz2
                # compute dudz (lands on F/B faces)
                dudzF = ( ucvF -ucv ) / dz
                dudzB = ( ucv -ucvB ) / dz
                # compute d2udz2
                diff += NU*( dudzF -dudzB ) / dz

                ut[k,j,i] = u[k,j,i] +dt*( -conv +diff +1.0 )

    # y-momentum (final terms land in vcv-centers)
    for k in range(1, ncv_z-1):
        for j in range(1, nno_y-1):
            for i in range(1, ncv_x-1):

                # compute neighbouring CV quantities
                vcv  = v[k,j,i]
                vcvE = v[k,j,i+1]
                vcvW = v[k,j,i-1]
                vcvN = v[k,j+1,i]
                vcvS = v[k,j-1,i]
                vcvF = v[k+1,j,i]
                vcvB = v[k-1,j,i]
                # compute CV corner quantities
                



def apply_BCs(u,v,w,p):
    # x
    u[:,:,0] = u[:,:,-2]
    u[:,:,-1] = u[:,:,1]
    # y
    u[:,0,:] = -u[:,1,:]
    u[:,-1,:] = -u[:,-2,:]
    # z
    u[0,:,:] = u[-3,:,:]
    u[-2,:,:] = u[1,:,:]
    u[-1,:,:] = u[2,:,:]
    

nno_x = 4
nno_y = 5
nno_z = 5

NITER = 50
NU = 0.1
dt = 0.01
#    dt_vis_u = CFL*dxmin**2d0/nu
#    dt_vis_v = CFL*dymin**2d0/nu
#    dt_vis_w = CFL*dzmin**2d0/nu
#
#    dt_vis = Minval( (/dt_vis_u,dt_vis_v,dt_vis_w/) )
timesteps = 10000 
CFL = 0.9

ncv_x = nno_x +1
ncv_y = nno_y +1
ncv_z = nno_z +1

u = np.zeros((ncv_z,ncv_y,nno_x))
v = np.zeros((ncv_z,nno_y,ncv_x))
w = np.zeros((nno_z,ncv_y,ncv_x))
p = np.zeros((ncv_z,ncv_y,ncv_x))
ut = np.zeros_like(u)
vt = np.zeros_like(v)
wt = np.zeros_like(w)

# grid points (nodes)
x = np.linspace(0,2*np.pi,nno_x)
y = np.linspace(-1.0, 1.0, nno_y)  
y = np.tanh(2.2 * y) / np.tanh(2.2) 
# remember that y and v are offset by 1 in indexing because of below:
y = np.insert(y, 0, y[0]-(y[1]-y[0]))
y = np.append(y, y[-1]+(y[-1]-y[-2]))
z = np.linspace(0.0,np.pi,nno_z)

# equal spacing
dx = x[1]-x[0]
dz = z[1]-z[0]

tim = 0.0

for istep in range(timesteps):

    solve_uvw( dt,u,v,w,ut,vt,wt )
    apply_BCs( ut, vt, wt, p )
    #solve_p( dt,p,b,ut,vt,wt )
    #apply_BCs( ut, vt, wt, p )
    #correct_uvw( dt,u,v,w,p,ut,vt,wt )
    #apply_all_BCs( ut, vt, wt, p )
    u = ut

    u_max = np.max(u)
    v_max = np.max(v)
    w_max = np.max(w)

    vel_max = abs(max(u_max, v_max, w_max))
    print(f'Max velocity: {u_max:.4f} {v_max:.4f} {w_max:.4f}')

    print(f'Time step size: {dt:.8f}')
    print(f'Simulation time: {tim:.8f}')

    tim = tim + dt

np.save(f'u.npy', u)
np.save(f'v.npy', v)
np.save(f'w.npy', w)
np.save(f'p.npy', p)
np.save(f'y.npy', y)
np.save(f'x.npy', x)
np.save(f'z.npy', z)
