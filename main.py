import numpy as np 
from numba import jit
from Initialize import *
from Animation import *
from contour import *
import copy
from ruamel.yaml import YAML
import sys 

def timer(t1):
    # print(time.time()-t1)
    return time.time()


@jit(nopython=True)
def transport(T,u,v,dt,dx,dy,alpha):
    nx = T.shape[0]-1
    ny = T.shape[1]-1

    T_ = np.zeros(T.shape)
    T_[:,:] = T[:,:]
    def RHS(alpha,T,dx,dy,i,j):
        rx_top=dx[(i,j)]-dx[(i-1,j)]
        rx_bot=dx[(i,j)]+dx[(i-1,j)]
        rx = rx_top/rx_bot
        
        ry_top=dy[(i,j)]-dy[(i,j-1)]
        ry_bot=dy[(i,j)]+dy[(i,j-1)]
        ry = ry_top/ry_bot

        T2_x=(((1-rx)*T[(i+1,j)])-(2*T[(i,j)])+((1+rx)*T[(i-1,j)]))/((dx[(i,j)]**2+dx[(i-1,j)]**2)/2)
        T2_y=(((1-ry)*T[(i,j+1)])-(2*T[(i,j)])+((1+ry)*T[(i,j-1)]))/((dy[(i,j)]**2+dy[(i,j-1)]**2)/2)

        rhs=alpha*(T2_x + T2_y)

        return rhs

    # def Der_1(u,v,T,dx,dy,i,j):
    #     # duTdx=((u[(i,j)]*(T[(i+1,j)]+T[(i,j)]))-(u[(i-1,j)]*(T[(i,j)]+T[(i-1,j)])))/dx[(i,j)]/2
    #     # dvTdy=((v[(i,j)]*(T[(i,j+1)]+T[(i,j)]))-(v[(i,j-1)]*(T[(i,j)]+T[(i,j-1)])))/dy[(i,j)]/2

    #     duTdx = 0.
    #     dvTdy = 0.

    #     return duTdx, dvTdy

    # def Der_1(u,v,T,dx,dy,i,j):

    #     x_1 = dx[(i,j)] 
    #     x_0 = dx[(i-1,j)]

    #     y_1 = dy[(i,j)] 
    #     y_0 = dy[(i,j-1)]

    #     duTdx = (x_0 + x_1)**(-1)*(0.5*(u[i+1,j] + u[i,j]))*(T[i+1,j]-T[i-1,j])
    #     dvTdy = (y_0 + y_1)**(-1)*(0.5*(v[i,j+1] + v[i,j]))*(T[i,j+1]-T[i,j-1])

    #     return duTdx, dvTdy

    # for i in range(1,nx):
    #     for j in range(1,ny):
    #         rhs = RHS(alpha,T,dx,dy,i,j)
    #         duTdx,dvTdy=Der_1(u,v,T,dx,dy,i,j)
    #         T_[(i,j)]=T[(i,j)]+(dt*(rhs-duTdx-dvTdy))
    # return T_

    def Der_1(u,v,T,dx,dy,i,j):

        x_1 = dx[(i,j)] 
        x_0 = dx[(i-1,j)]

        y_1 = dy[(i,j)] 
        y_0 = dy[(i,j-1)]

        U = (0.5*(u[i,j] + u[i-1,j]))
        V = (0.5*(v[i,j] + v[i,j-1]))

        if U > 0:
            duTdx = (x_0)**(-1)*U*(T[i,j]-T[i-1,j])
        else:
            duTdx = (x_1)**(-1)*U*(T[i+1,j]-T[i,j])

        if V > 0:
            dvTdy = (y_0)**(-1)*V*(T[i,j]-T[i,j-1])
        else:
            dvTdy = (y_1)**(-1)*V*(T[i,j+1]-T[i,j])

        #duTdx = 0
        #dvTdy = 0

        return duTdx, dvTdy

    # def Der_1(u,v,T,dx,dy,i,j):

    #     x_1 = dx[(i,j)] 
    #     x_0 = dx[(i-1,j)]

    #     y_1 = dy[(i,j)] 
    #     y_0 = dy[(i,j-1)]

    #     duTdx = (x_1)**(-1)*((u[i,j])*(T[i+1,j]-T[i,j]))
    #     #print(duTdx)
    #     dvTdy = (y_1)**(-1)*((v[i,j])*(T[i,j+1]-T[i,j]))

    #     #duTdx = 0
    #     #dvTdy = 0

    #     return duTdx, dvTdy

    for i in range(1,nx):
        for j in range(1,ny):
            if i == 10 and j == 10:
                Q = 0
            else:
                Q = 0
            rhs = RHS(alpha,T,dx,dy,i,j)
            duTdx,dvTdy=Der_1(u,v,T,dx,dy,i,j)
            T_[(i,j)]=T[(i,j)]+(dt*(rhs-duTdx-dvTdy + Q))
    return T_

#*************************************************************
#         if k<2:
#             err = np.max((P_-temp))
#             err1 = err
#             print(err)
#         if k > 2:
#             err = np.max(np.abs(P_-temp))/err1
            
#         P = P_
#     print(k)
#     return P_

@jit(nopython=True)
def poisson(P,u,v,dt,dx,dy,rho):
    # u and v are from the "predictor step"
    # P comes from the previous time step
    nx = P.shape[0]-1
    ny = P.shape[1]-1
    def Frac(dx,dy,i,j):
        rx_top=dx[(i,j)]-dx[(i-1,j)]
        rx_bot=dx[(i,j)]+dx[(i-1,j)]
        rx = rx_top/rx_bot
        
        ry_top=dy[(i,j)]-dy[(i,j-1)]
        ry_bot=dy[(i,j)]+dy[(i,j-1)]
        ry = ry_top/ry_bot

        frac_x=4/(dx[(i,j)]**2+dx[(i-1,j)]**2)
        frac_y=4/(dy[(i,j)]**2+dy[(i,j-1)]**2)
        Rx_p=2*(1-rx)/(dx[(i,j)]**2+dx[(i-1,j)]**2)
        Rx_n=2*(1+rx)/(dx[(i,j)]**2+dx[(i-1,j)]**2)
        Ry_p=2*(1-ry)/(dy[(i,j)]**2+dy[(i,j-1)]**2)
        Ry_n=2*(1+ry)/(dy[(i,j)]**2+dy[(i,j-1)]**2)
        return frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n

    def RHS(u,v,dx,dy,i,j,dt,rho):
        U_=(u[(i,j)]-u[(i-1,j)])/dx[(i-1,j)]
        V_=(v[(i,j)]-v[(i,j-1)])/dy[(i,j-1)]
        
        rhs=rho/dt*(U_+V_)
        return rhs

    Con = 1e-2
    err = 1
    k = 0
    temp=np.zeros(P.shape)
    while (err>Con):
        temp[:,:] = P[:,:]#copy.deepcopy(P)
        #temp=P
        k+=1
        for i in range(1,nx):
            for j in range(1,ny):
                frac_x,frac_y,Rx_p,Rx_n,Ry_p,Ry_n=Frac(dx,dy,i,j)
                rhs = RHS(u,v,dx,dy,i,j,dt,rho)
                
                P[(i,j)]=(frac_x+frac_y)**(-1)*((Rx_p*P[(i+1,j)]+Rx_n*P[(i-1,j)])+(Ry_p*P[(i,j+1)]+Ry_n*P[(i,j-1)])-rhs)
        if k == 100000:# Look into this
            print('not converged', err)
            break
        err = np.max(np.abs(P-temp))
        # print(err)
    #print(k)
    return P

@jit(nopython=True)
def Adv(u, v, x, y, i, j, flag): 
    x_1 = x[(i,j)] 
    x_0 = x[(i-1,j)]

    y_1 = y[(i,j)] 
    y_0 = y[(i,j-1)]

    # u_1 = np.zeros((u.shape[0], u.shape[1]))
    # u_2 = np.zeros((u.shape[0], u.shape[1]))

    # v_1 = np.zeros((v.shape[0], v.shape[1]))
    # v_2 = np.zeros((v.shape[0], v.shape[1]))

    if flag == 0:
        U = u[i][j]
        V = 0.25*(v[i-1][j]+v[i][j]+v[i-1][j+1]+ v[i][j+1])

        if U > 0:
            u_1 = U*(u[i][j] - u[i-1][j])/(x_0)
        else:
            u_1 = U*(u[i+1][j] - u[i][j])/(x_1)

        if V > 0:
            u_2 = V*(u[i][j] - u[i][j-1])/(y_0)
        else:
            u_2 = V*(u[i][j+1] - u[i][j])/(y_1)

        return u_1 + u_2

    if flag == 1:
        U = 0.25*(u[i][j-1]+u[i][j]+u[i+1][j-1]+ u[i+1][j])
        V = v[i][j]

        if V > 0:
            v_1 = V*(v[i][j] - v[i][j-1])/(y_0)
        else:
            v_1 = V*(v[i][j+1] - v[i][j])/(y_1)

        if U > 0:    
            v_2 = U*(v[i][j] - v[i-1][j])/(x_0)
        else:
            v_2 = U*(v[i+1][j] - v[i][j])/(x_1)

        return v_1 + v_2

@jit(nopython=True)
def Diff(u, x, y, i, j):

    # u_1 = np.zeros((u.shape[0], u.shape[1]))
    # u_2 = np.zeros((u.shape[0], u.shape[1]))

    x_1 = x[(i,j)] 
    x_0 = x[(i-1,j)]

    y_1 = y[(i,j)] 
    y_0 = y[(i,j-1)]

    r_x = (x_1 - x_0)/(x_0 + x_1)
    r_y = (y_1 - y_0)/(y_0 + y_1)

    u_1 = 2*((1+r_x)*u[i-1][j] - 2*u[i][j] + (1-r_x)*u[i+1][j])/(x_0**2 + x_1**2)#x[(i,j)]**2)
    u_2 = 2*((1+r_y)*u[i][j-1] - 2*u[i][j] + (1-r_y)*u[i][j+1])/(y_0**2 + y_1**2)#(y[(i,j)]**2)

    return u_1 + u_2

@jit(nopython=True)
def predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta):
    # Main Predictor Loop
    nx = T.shape[0]
    ny = T.shape[1]
    Cs = 0.2
    delta_g = 0.00125

    #print(v.shape[0], v.shape[1])

    u_ = np.zeros((u.shape[0], u.shape[1]))
    v_ = np.zeros((v.shape[0], v.shape[1]))

    # u_ = copy.deepcopy(u)
    # v_ = copy.deepcopy(v)

    u_[:,:]= u[:,:]
    v_[:,:]= v[:,:]
    #print('*******')
    #print(nx, ny)


    for i in range(1, nx-1):
        for j in range(1, ny-1):
            #print(i,j)
            S = 0.5*((u[i][j+1]-u[i][j])/y[i,j]) + ((v[i+1][j] - v[i][j])/x[i,j])
            nut = ((Cs*delta_g)**2)*S
            nut = min(1e-4, abs(nut))
            #print(nut)
            #nut = 0
            Nu = nu + nut
            if nut > 1e-3:

                print(nut)

            u_[i][j] = u[i][j] + dt*(Nu*(Diff(u, x, y, i, j)) - Adv(u, v, x, y, i, j, 0))

            v_[i][j] = v[i][j] + dt*(Nu*(Diff(v, x, y, i, j)) - Adv(u, v, x, y, i, j, 1)+ rho*g*beta*(0.5*(T[i][j] + T[i][j+1])-T_ref)) #Add Bousinessq Terms  
    
    return u_, v_

@jit(nopython=True)
def corrector(x, y, u, v, p, dt, rho):

    nx = p.shape[0]
    ny = p.shape[1]


    u_ = np.zeros((u.shape[0], u.shape[1]))
    v_ = np.zeros((v.shape[0], v.shape[1]))

    # u_ = copy.deepcopy(u)
    # v_ = copy.deepcopy(v)
    u_[:,:]= u[:,:]
    v_[:,:]= v[:,:]

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            x_1 = x[(i,j)] 
            x_0 = x[(i-1,j)]
            y_1 = y[(i,j)] 
            y_0 = y[(i,j-1)]

            u_[i][j] = u[i][j] - (dt/rho)*(p[i+1][j] - p[i][j])/(x_1)#(x_1 + x_0)
            v_[i][j] = v[i][j] - (dt/rho)*(p[i][j+1] - p[i][j])/(y_1)#(y_1 + y_0)
    
    return u_, v_

@jit(nopython=True)
def BC_update(u, v, p, T, phi):
    nx = p.shape[0]-1
    ny = p.shape[1]-1
    #inlet
    #v[0,:] = v[1,:]
    v[0,:] = -v[1,:]

    p[0,:] = p[1,:]
    T[0,:] = T[1,:]
    phi[0,:] = phi[1,:]

    #bottom
    u[:,0] =  -u[:,1]

    v[:,0] = 0.

    p[:,0] = p[:,1]
    T[:,0] = T[:,1]
    phi[:,0] = phi[:,1]

    #outlet
    u[nx-1,:] = u[nx-2,:]
    u[nx,:] = u[nx-1,:]

    v[nx,:] = v[nx-1,:]
    p[nx-1,:] = p[nx,:]
    T[nx,:] = T[nx-1,:]
    phi[nx,:] = phi[nx-1,:]

    #top
    u[:,ny] =  u[:,ny-1]
    # u[:,ny] = 2.
    # u[:,ny-1] = 2.
    # u[:,ny] = -u[:,ny-1]

    # v[:,ny-1] =  v[:,ny-2]
    # v[:,ny] =  v[:,ny-1]
    v[:,ny] = 0.
    v[:,ny-1] = 0.


    p[:,ny] = p[:,ny-1]
    T[:,ny] = T[:,ny-1]
    phi[:,ny] = phi[:,ny-1]

    
    T[150:152,100:102] = 375
    phi[150:152,100:102] = 40

    return u, v, p, T, phi

@jit(nopython=True)
def Building_BC(u, v, p, T, phi, Dim):
    k, k_, r = Dim
    nx = p.shape[0]-1
    ny = p.shape[1]-1
    #left
    u[k,:r] = 0.
    v[k+1,:r] = -v[k,:r]
    p[k+1,:r] = p[k,:r]
    T[k+1,:r] = T[k,:r]
    phi[k+1,:r] = phi[k,:r]

   
    #top
    u[k+1:k_,r-1] = -u[k+1:k_,r]
    v[k+1:k_,r-1] = 0.
    p[k+1:k_,r-1] = p[k+1:k_,r]
    T[k+1:k_,r-1] = T[k+1:k_,r]
    phi[k+1:k_,r-1] = phi[k+1:k_,r]
    
    #right
    u[k_-1,:r] = 0.
    v[k_-1,:r] = -v[k_,:r]
    p[k_-1,:r] = p[k_,:r]
    T[k_-1,:r] = T[k_,:r]
    phi[k_-1,:r] = phi[k_,:r]
    
    #inside
    u[k+1:k_,:r-1] = 0.
    v[k+2:k_-1,:r-1] = 0.
    p[k+2:k_-1,:r-1] = 0.
    T[k+2:k_-1,:r-1] = 325.
    phi[k+2:k_-1,:r-1] = 10.
    
    return u, v, p, T, phi


#@numba.jit(nopython=True, parallel=True)
def main():
    yaml = YAML()
    with open(sys.argv[1]) as file: 
        v = yaml.load(file)  
        constant_dict=v["constants"]

        rho = constant_dict["rho"]
        T_ref = constant_dict["T_ref"]
        beta = constant_dict["beta"]
        nu = constant_dict["nu"]
        alpha_T = constant_dict["alpha_T"]
        alpha_pollutant = constant_dict["alpha_pollutant"]
        total_t = constant_dict["total_t"]
        dt = constant_dict["dt"]
        dx = constant_dict["dx"]
        g = constant_dict["g"]
        cfl = constant_dict["cfl"]
    
        b_1=v['building_1']
        b_2=v['building_2']
        b_3=v['building_3']

    initialize(sys.argv[1])

    x,y = read_delta(0)
    P, T, u, v, phi = read_all_scalar(0)
    print(P.shape) 
    #print(T.shape[0], T.shape[1])

    running = True
    t = 0.
    ite = 0.
    while running:
        ite += 1
        if int(ite)%80==0:
            print('\ntime=%.3f'%t)
            
            if t!=0:
                print('V',V)
                write_all_scalar(P, T, u, v, phi, t)
            # sys.exit()
        t1=time.time()
        
        u_, v_ = predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta)
        t1=timer(t1)
        #print(u_)
        p_new = poisson(P,u_,v_,dt,x,y,rho)
        t1=timer(t1)
        #p_new = copy.deepcopy(P)
        #print(p_new)
        u_new, v_new = corrector(x, y, u_, v_, p_new, dt, rho)
        t1=timer(t1)
        #print(u_new)
        T_new = transport(T,u,v,dt,x,y,alpha_T)
        #T_new = np.zeros(T.shape)
        #print(T_new)
        phi_new = transport(phi,u,v,dt,x,y,alpha_pollutant) #Pollutant Transport

        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(BC_update(u_new, v_new, p_new, T_new, phi_new))
        
        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new, v_new, p_new, T_new, phi_new,[b_1["x_0"],b_1["x_1"],b_1["y_1"]]))
        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new, v_new, p_new, T_new, phi_new,[b_2["x_0"],b_2["x_1"],b_2["y_1"]]))
        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new, v_new, p_new, T_new, phi_new,[b_3["x_0"],b_3["x_1"],b_3["y_1"]]))

        u = copy.deepcopy(u_new)
        v = copy.deepcopy(v_new)

        P = copy.deepcopy(p_new)
        T = copy.deepcopy(T_new)
        phi = copy.deepcopy(phi_new)

        u_max =np.max(abs(u))
        v_max = np.max(abs(v))
        V = np.sqrt(u_max**2 + v_max**2)
        # print('V',V)
        dt = (cfl*dx)/V
        # print(dt)
        t += dt

        #sys.exit()
        #write_all_scalar(P, T, u_new, v_new, t)
        
        if t >= total_t:
            write_all_scalar(P, T, u_new, v_new, phi, t)
            running = False
    Contour('U',grid='yes')
    Contour('V',grid='yes')
    Contour('P',grid='yes')

    #Streamlines('U','V',grid='yes')

    Contour('T',grid='yes')
    Contour('phi',grid='yes')

    # print(u)


if __name__ == '__main__':
    main()