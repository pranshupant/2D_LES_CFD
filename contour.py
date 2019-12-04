import numpy as np,os,copy,sys
from functions import *
import matplotlib.pyplot as plt

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def averaging(Scalar,axis=0):
    if axis==0: # x axis averaging
        nx=Scalar.shape[0]-2
        ny=Scalar.shape[1]-2
        Scalar1=np.zeros([nx,ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i,j]=(Scalar[i+1,j]+Scalar[i+1,j+1])/2
    if axis==1: # y axis averaging
        nx=Scalar.shape[0]-2
        ny=Scalar.shape[1]-2
        Scalar1=np.zeros([nx,ny])    
        for i in range(nx):
            for j in range(ny):
                Scalar1[i,j]=(Scalar[i,j+1]+Scalar[i+1,j+1])/2                
    return Scalar1

def Point_average(Points):
    nx=Points.shape[1]-1
    ny=Points.shape[2]-1
    Points1=np.zeros([2,nx,ny])    
    for k in range(2):
        for i in range(nx):
            for j in range(ny):
                Points1[k,i,j]=(Points[k,i,j]+Points[k,i+1,j]+Points[k,i,j+1]+Points[k,i+1,j+1])/4
    return Points1


def plotting(Points,Scalar,Scalar_name,show='yes',P='no'):
    plt.contourf(Points[0],Points[1],Scalar,20)
    plt.title('Contour of '+Scalar_name)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    xmin,xmax=np.min(Points[0]),np.max(Points[0])
    ymin,ymax=np.min(Points[1]),np.max(Points[1])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])    
    if P=='yes':
        print(Scalar)

    if show=='yes':
        plt.show()
    else:
        plt.close()
    
def Contour(Scalar_name,time=-1,show='yes',P='no'): #scalar must be string of what you want to print
    #find all times, hence end time
    times=[]
    files=os.listdir('Results')
    for i in files:
        if isfloat(i):
            times.append('Results/'+i)
    if time==-1:
        Scalar=read_scalar(times[time]+'/'+Scalar_name+'.txt')
    elif time=='0':
        Scalar=read_scalar(str(time)+'/'+Scalar_name+'.txt')
    else: 
        Scalar=read_scalar('Results/'+str(time)+'/'+Scalar_name+'.txt')
    Points=read_points('Constant/Points.txt')
    Points=copy.deepcopy(Point_average(Points))
    if Scalar_name=='U':
        Scalar=copy.deepcopy(averaging(Scalar,1))
    elif Scalar_name=='V':
        Scalar=copy.deepcopy(averaging(Scalar,0))
    else:
        Scalar = copy.deepcopy(Scalar[1:-1,1:-1])
    
    plotting(Points,Scalar,Scalar_name,show,P)


def Streamlines(U_name,V_name,time=-1,show='yes'):
    times=[]
    files=os.listdir('Results')
    for i in files:
        if isfloat(i):
            times.append('Results/'+i)
    if time==-1:
        U=read_scalar(times[time]+'/'+U_name+'.txt')
        V=read_scalar(times[time]+'/'+V_name+'.txt')    
    elif time=='0': 
        U=read_scalar(str(time)+'/'+U_name+'.txt')
        V=read_scalar(str(time)+'/'+V_name+'.txt')
    else:
        U=read_scalar('Results/'+str(time)+'/'+U_name+'.txt')
        V=read_scalar('Results/'+str(time)+'/'+V_name+'.txt')
            
    Points=read_points('Constant/Points.txt')
    Points=copy.deepcopy(Point_average(Points))
    U=copy.deepcopy(averaging(U,1))
    V=copy.deepcopy(averaging(V,0)) 
    
    speed=np.sqrt(U**2 + V**2).T
    lw=3*speed**0.3/np.max(speed)
    plt.title('Streamlines')
    plt.streamplot(Points[0][:,0],Points[1][0,:], U.T, V.T, color=speed,linewidth=lw, cmap='Spectral',density=4)#,minlength=dx/10)
    plt.colorbar()
    xmin,xmax=min(Points[0][:,0]),max(Points[0][:,0])
    ymin,ymax=min(Points[1][0,:]),max(Points[1][0,:])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    
    plt.show()    

# Contour('U','0',P='yes')
# Contour('V','0',P='yes')

# Streamlines('U','V','0')

