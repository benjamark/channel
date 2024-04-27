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
                ufNW = u[k,j+1,i-1]
                ufSW = u[k,j,i-1]
                ufNE = u[k,j+1,i]
                ufSE = u[k,j,i]

                wfNB = w[k-1,j+1,i] 
                wfSB = w[k-1,j,i]
                wfNF = w[k,j+1,i]
                wfSF = w[k,j,i]

                # convective terms:
                # d(uv)/dx +d(vv)/dy +d(wv)/dz
                conv = 0.0

                # d(uv)/dx
                # interp v in x 
                vfE = 0.5*( vcv +vcvE )
                vfW = 0.5*( vcvW +vcv )
                # interp u in y
                # compute required vcv half-lengths
                dyNh = 0.5*( y[j+2] -y[j+1] )  # length of north half of vcv
                dySh = 0.5*( y[j+1] -y[j] ) 
                ufW = ( ufNW*dySh +ufSW*dyNh ) / ( dySh +dyNh ) 
                ufE = ( ufNE*dySh +ufSE*dyNh ) / ( dySh +dyNh )

                conv += ( ufE*vfE -ufW*vfW ) / dx

                # d(v2)/dy
                # compute vcv length
                dy = 0.5*( y[j+2] +y[j+1] ) -0.5*( y[j+1] +y[j] )
                # interp v in y (F to C)
                vfN = 0.5*( vcv +vcvN )
                vfS = 0.5*( vcvS +vcv )

                conv += ( vfN**2 -vfS**2 ) / dy

                # d(wv)/dz
                # interp w in y
                wfB = ( wfNB*dySh +wfSB*dyNh ) / ( dySh +dyNh )
                wfF = ( wfNF*dySh +wfSF*dyNh ) / ( dySh +dyNh )
                # interp v in z
                vfB = 0.5*( vcv +vcvB )
                vfF = 0.5*( vcv +vcvF )

                conv += ( wfF*vfF -wfB*vfB ) / dz

                # diffusive terms:
                # d2(v)/dx2 +d2(v)/dy2 +d2(v)/dz2
                diff = 0.0

                # d2(v)/dx2
                dvdxE = ( vcvE -vcv ) / dx
                dvdxW = ( vcv -vcvW ) / dx
                diff += NU*( dvdxE -dvdxW ) / dx

                # d2(v)/dy2
                # compute required distances to N and S vcvs
                dyN = y[j+2] -y[j+1]
                dyS = y[j+1] -y[j]
                dvdyN = ( vcvN -vcv ) / dyN
                dvdyS = ( vcv -vcvS ) / dyS
                diff += NU*( dvdyN -dvdyS ) / dy

                # d2(v)/dz2
                dvdzF = ( vcvF -vcv ) / dz
                dvdzB = ( vcv -vcvB ) / dz
                diff += NU*( dvdzF -dvdzB ) / dz

                vt[k,j,i] = v[k,j,i] +dt*( -conv +diff )


                
    # z-momentum (final terms land in wcv-centers)
    for k in range(1, nno_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, ncv_x-1):

                # pre-compute useful velocities
                # compute neighbouring CV quantities
                wcv  = w[k,j,i]
                wcvE = w[k,j,i+1]
                wcvW = w[k,j,i-1]
                wcvN = w[k,j+1,i]
                wcvS = w[k,j-1,i]
                wcvF = w[k+1,j,i]
                wcvB = w[k-1,j,i]
                # compute CV corner quantities
                ufEF = u[k+1,j,i]
                ufWF = u[k+1,j,i-1]
                ufEB = u[k,j,i]
                ufWB = u[k,j,i-1]

                vfNF = v[k+1,j,i]
                vfSF = v[k+1,j-1,i]
                vfNB = v[k,j,i]
                vfSB = v[k,j-1,i]

                # convective terms:
                # d(uw)/dx +d(vw)/dy +d(w2)/dz
                conv = 0.0

                # d(uw)/dx
                # interp u in x so it is at E and W faces
                wfE = 0.5*( wcv +wcvE )
                wfW = 0.5*( wcvW +wcv )
                # interp u in x
                ufE = 0.5*( ufEF +ufEB )
                ufW = 0.5*( ufWF +ufWB )

                conv += ( ufE*wfE -ufW*wfW ) / dx

                # d(vw)/dy
                # interp v in z
                vfN = 0.5*( vfNB +vfNF )
                vfS = 0.5*( vfSB +vfSF )
                # interp w in y
                # compute required wcv lengths
                dy = y[j+1] -y[j]
                dyN = y[j+2] -y[j+1]
                dyS = y[j] -y[j-1]
                # interp w in y so it is at N and S faces
                wfN = ( (dyN/2)*wcv +(dy/2)*wcvN ) / (dyN/2 +dy/2)
                wfS = ( (dy/2)*wcvS +(dyS/2)*wcv ) / (dy/2 +dyS/2)

                conv += ( vfN*wfN -vfS*wfS ) / dy

                # d(w2)/dz
                # interp w in z
                wfF = 0.5*( wcv +wcvF )
                wfB = 0.5*( wcvB +wcv )

                conv += ( wfF*wfF -wfB*wfB ) / dz


                # diffusive terms:
                # d2(w)/dx2 +d2(w)/dy2 +d2(w)/dz2
                diff = 0.0

                # d2(w)/dx2
                # compute dwdx (lands on E/W faces)
                dwdxE = ( wcvE -wcv ) / dx
                dwdxW = ( wcv -wcvW ) / dx
                # compute d2wdx2
                diff += NU*( dwdxE -dwdxW ) / dx

                # d2(w)dy2
                # compute dwdy (lands on N/S faces)
                PFN = (dyN/2)/(dy/2)
                # see Sundqvist and Veronis (1969) eq 1.3
                dwdyN = ( wcvN-(PFN**2)*wcv -(1 -(PFN**2))*wfN ) /\
                        (dyN/2*(1+PFN))
                PFS = (dy/2)/(dyS/2)
                dwdyS = (wcv-(PFS**2)*wcvS -(1 -(PFS**2))*wfS) /\
                        (dy/2*(1+PFS))

                diff += NU*( dwdyN -dwdyS ) / dy

                # d2(w)dz2
                # compute dwdz (lands on F/B faces)
                dwdzF = ( wcvF -wcv ) / dz
                dwdzB = ( wcv -wcvB ) / dz
                # compute d2wdz2
                diff += NU*( dwdzF -dwdzB ) / dz

                wt[k,j,i] = w[k,j,i] +dt*( -conv +diff )

def jacobi_update_rbgs(p, a_east, a_west, a_north, a_south, a_front, a_back, a_p, b, color_phase):
    for k in range(1, ncv_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, ncv_x-1):
                if (i + j + k) % 2 == color_phase:
                    # assumes ghosts have been properly updated
                    tmp = (b[k, j, i] - \
                           a_east[k, j, i] * p[k, j, i+1] - \
                           a_west[k, j, i] * p[k, j, i-1] - \
                           a_north[k, j, i] * p[k, j+1, i] - \
                           a_south[k, j, i] * p[k, j-1, i] - \
                           a_front[k, j, i] * p[k+1, j, i] - \
                           a_back[k, j, i] * p[k-1, j, i]) / a_p[k, j, i]
                    p[k,j,i] = tmp


def solve_p( dt,p,b,ut,vt,wt ):

    # populate RHS b
    # b = (1/dt)*div(u)

    for k in range(1, ncv_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, ncv_x-1):
                dy = y[j+1]-y[j]  
                # F to C 
                b[k,j,i] = (1/dt)*( (ut[k,j,i] -ut[k,j,i-1])/dx +\
                                    (vt[k,j,i] -vt[k,j-1,i])/dy +\
                                    (wt[k,j,i] -wt[k-1,j,i])/dz )

    # solve Poisson system in-place, iteratively
    for _ in range(NITER):
        # update red nodes
        jacobi_update_rbgs(p, a_east, a_west, a_north, a_south, a_front, a_back, a_p, b, 0)
        # update black nodes
        jacobi_update_rbgs(p, a_east, a_west, a_north, a_south, a_front, a_back, a_p, b, 1)


def correct_uvw( dt,u,v,w,p,ut,vt,wt ):
    # correct u
    for k in range(1, ncv_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, nno_x-1):
                u[k, j, i] = ut[k, j, i] - (dt/dx) * (p[k, j, i+1] - p[k, j, i])
    # correct v
    for k in range(1, ncv_z-1):
        for j in range(1, nno_y-1):
            for i in range(1, ncv_x-1):
                # need pressure at y-faces
                dyN = 0.5*(y[j+2]-y[j+1])
                dyS = 0.5*(y[j+1]-y[j])
                pf = ( dyS*p[k,j+1,i] + dyN*p[k,j,i] ) / (dyS+dyN)
                v[k, j, i] = vt[k, j, i] - dt * ( (p[k,j+1,i] -pf)/dyN**2 +\
                             (pf -p[k,j,i])/dyS**2 ) * (dyS*dyN)/(dyS+dyN)
    # correct w
    for k in range(1, nno_z-1):
        for j in range(1, ncv_y-1):
            for i in range(1, ncv_x-1):
                w[k, j, i] = wt[k, j, i] - (dt/dz) * (p[k+1, j, i] - p[k, j, i])


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

    # for v
    # x
    v[:,:,0] = v[:,:,-3]
    v[:,:,-2] = v[:,:,1]
    v[:,:,-1] = v[:,:,2]
    # y
    v[:,0,:] = 0.0
    v[:,-1,:] = 0.0
    # z
    v[0,:,:] = v[-3,:,:]
    v[-2,:,:] = v[1,:,:]
    v[-1,:,:] = v[2,:,:]

    # for w
    # x
    w[:,:,0] = w[:,:,-3]
    w[:,:,-2] = w[:,:,1]
    w[:,:,-1] = w[:,:,2]
    # y
    w[:,0,:] = -w[:,1,:]
    w[:,-1,:] = -w[:,-2,:]
    # z
    w[0,:,:] = w[-2,:,:]
    w[-1,:,:] = w[1,:,:]

    # for p
    # x
    p[:,:,0] = p[:,:,-3]
    p[:,:,-2] = p[:,:,1]
    p[:,:,-1] = p[:,:,2]
    # y
    p[:,0,:] = p[:,1,:]
    p[:,-1,:] = p[:,-2,:]
    # z
    p[0,:,:] = p[-3,:,:]
    p[-2,:,:] = p[1,:,:]
    p[-1,:,:] = p[2,:,:]
    

nno_x = 4
nno_y = 5
nno_z = 5

NITER = 50
NU = 1/10
dt = 0.01
#    dt_vis_u = CFL*dxmin**2d0/nu
#    dt_vis_v = CFL*dymin**2d0/nu
#    dt_vis_w = CFL*dzmin**2d0/nu
#
#    dt_vis = Minval( (/dt_vis_u,dt_vis_v,dt_vis_w/) )
timesteps = 10000
CFL = 0.4

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

# initialize the matrix of coefficient
a_east  = np.zeros((ncv_z,ncv_y,ncv_x))
a_west  = np.zeros((ncv_z,ncv_y,ncv_x))
a_north = np.zeros((ncv_z,ncv_y,ncv_x))
a_south = np.zeros((ncv_z,ncv_y,ncv_x))
a_front = np.zeros((ncv_z,ncv_y,ncv_x))
a_back = np.zeros((ncv_z,ncv_y,ncv_x))
a_p     = np.zeros((ncv_z,ncv_y,ncv_x))
b       = np.zeros((ncv_z,ncv_y,ncv_x))

# build the matrix for the pressure equation
for k in range(1,ncv_z-1):
    for j in range(1,ncv_y-1):
        for i in range(1,ncv_x-1):
            a_west[k,j,i] = 1/dx**2
            a_east[k,j,i] = 1/dx**2

            dyS = 0.5*(y[j+1]+y[j]) -0.5*(y[j]+y[j-1])
            dyN = 0.5*(y[j+2]+y[j+1]) -0.5*(y[j+1]+y[j]) 
            a_south[k,j,i] = 2/(dyS*(dyN +dyS))
            a_north[k,j,i] = 2/(dyN*(dyN +dyS))

            a_back[k,j,i] = 1/dz**2
            a_front[k,j,i] = 1/dz**2

a_p = -( a_north +a_south +a_east +a_west +a_front +a_back )

tim = 0.0

# IC
u = np.random.random(u.shape)
v = np.random.random(v.shape)
w = np.random.random(w.shape)

for istep in range(timesteps):

    solve_uvw( dt,u,v,w,ut,vt,wt )
    apply_BCs( ut, vt, wt, p )
    solve_p( dt,p,b,ut,vt,wt )
    apply_BCs( ut, vt, wt, p )
    correct_uvw( dt,u,v,w,p,ut,vt,wt )
    apply_BCs( u, v, w, p )
    u = ut
    v = vt
    w = wt

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
