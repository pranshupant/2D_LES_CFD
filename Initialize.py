import numpy as np,sys,time,os,shutil
import numba 
from numba import jit 
import inspect
import json
from functions import *
#blah blah blah
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
# Hello its Me!
# Hello this is papa and mama
def initialize():
    
    try:
        print('Removing 0 time')
        files=os.listdir()
        for i in files:
            if isfloat(i):
                shutil.rmtree(i)
        
        shutil.rmtree('Constant')
        print('Removed Constant')
        
        shutil.rmtree('Results')
        print('Removed Results')
        
    except:
        print('Results/Constants/0 does not exist')
    try:
        os.makedirs('Constant/')
        print('Created Constants')
        
        os.makedirs('0/')
        print('Created 0')
        
    except:
        print('Running Code with existing Constants/0')

    Cons='Constant/'
    Rho=1.225
    N=800
    M=200
    nx=N+1
    ny=M+1
    xmax=2
    ymax=0.5
    atm=1
    Temperature=300

    x=np.linspace(0,xmax,nx)
    y=np.linspace(0,ymax,ny)

    X,Y=np.meshgrid(x,y)
    t1=time.time()

    Points=np.zeros([2,nx,ny])
    for j in range(ny):
        for i in range(nx):
            Points[0,i,j]=X[j,i]
            Points[1,i,j]=Y[j,i]

    write_points(Cons+'Points.txt',Points)

    # Points_read=read_points(Cons+'Points.txt')

    dx,dy=deltas_write(Points)
    # print(dx)
    # print(dy)
    #print(dx.shape)
    # dx=np.ones([N+1,M+1])*xmax/N
    # dy=np.ones([N+1,M+1])*ymax/M
    
    write_scalar(Cons+'Dx.txt',dx)
    write_scalar(Cons+'Dy.txt',dy)

    P=np.zeros([nx+1,ny+1])*atm
    write_scalar('0/P.txt',P)

    U_in=np.linspace(0,1,ny+1)
    U=np.zeros([nx+1,ny+1])
    U[0,:]= 2.*U_in
    # print(U)
    write_scalar('0/U.txt',U)

    V=np.zeros([nx+1,ny+1])
    write_scalar('0/V.txt',V)

    T=np.ones([nx+1,ny+1])*Temperature

    T[20,5] = 325

    # print(T)
    write_scalar('0/T.txt',T)
# initialize()
