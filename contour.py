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

def averaging_grid(Scalar,axis=0):
    if axis==0: # x axis averaging
        nx=Scalar.shape[0]-1
        ny=Scalar.shape[1]-1
        Scalar1=np.zeros([nx,ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i,j]=(Scalar[i,j]+Scalar[i+1,j])/2
    if axis==1: # y axis averaging
        nx=Scalar.shape[0]-1
        ny=Scalar.shape[1]-1
        Scalar1=np.zeros([nx,ny])    
        for i in range(nx):
            for j in range(ny):
                Scalar1[i,j]=(Scalar[i,j]+Scalar[i,j+1])/2                
    return Scalar1

def average_scalar(Scalar):
    nx=Scalar.shape[0]-1
    ny=Scalar.shape[1]-1
    Scalar1=np.zeros([nx,ny])    
    for i in range(nx):
        for j in range(ny):
            Scalar1[i,j]=(Scalar[i,j]+Scalar[i+1,j]+Scalar[i,j+1]+Scalar[i+1,j+1])/4
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
    if Scalar_name=='T':
        plt.contourf(Points[0],Points[1],Scalar,200,cmap='coolwarm',vmin=300,vmax=350)
    else:
        plt.contourf(Points[0],Points[1],Scalar,250,cmap='coolwarm')
    
    plt.title('Contour of '+Scalar_name)
    plt.colorbar(orientation='horizontal')
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
    
def Contour(Scalar_name,time=-1,show='yes',P='no',grid='no'): #scalar must be string of what you want to print
    #find all times, hence end time
    times=[]
    try:
        files=os.listdir('Results')
        for i in files:
            if isfloat(i):
                times.append('Results/'+i)
    except:
        print('Result file not there')
        return 0
    if time==-1:
        Scalar=read_scalar(times[time]+'/'+Scalar_name+'.txt')
    elif time=='0':
        Scalar=read_scalar(str(time)+'/'+Scalar_name+'.txt')
    else: 
        Scalar=read_scalar('Results/'+str(time)+'/'+Scalar_name+'.txt')
    Points=read_points('Constant/Points.txt')
    if grid=='no':
        Points=copy.deepcopy(Point_average(Points))
        if Scalar_name=='U':
            Scalar=copy.deepcopy(averaging(Scalar,1))
        elif Scalar_name=='V':
            Scalar=copy.deepcopy(averaging(Scalar,0))
        else:
            Scalar = copy.deepcopy(Scalar[1:-1,1:-1])
    else:      
        if Scalar_name=='U':
            Scalar=copy.deepcopy(averaging_grid(Scalar,1))
        elif Scalar_name=='V':
            Scalar=copy.deepcopy(averaging_grid(Scalar,0))
        else:
            Scalar = copy.deepcopy(average_scalar(Scalar))

    plotting(Points,Scalar,Scalar_name,show,P)


def Streamlines(U_name,V_name,time=-1,show='yes',grid='no'):
    times=[]
    try:
        files=os.listdir('Results')
        for i in files:
            if isfloat(i):
                times.append('Results/'+i)
    except:
        print('Result file not there')
        return 0
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
    if grid=='no':
        Points=copy.deepcopy(Point_average(Points))
        U=copy.deepcopy(averaging(U,1))
        V=copy.deepcopy(averaging(V,0)) 
    else:
        U=copy.deepcopy(averaging_grid(U,1))
        V=copy.deepcopy(averaging_grid(V,0)) 
    #speed=np.sqrt(U**2 + V**2).T
    speed=np.sqrt(U**2 + V**2).T
    #speed=1    
    lw=1*speed**0.3/np.max(speed)
    
    plt.title('Streamlines')
    #plt.gca().set_aspect('equal') 
    print(Points[0][:,0],Points[1][0,:])
    print(Points.shape)
    print(U.shape)
    print(V.shape)
    np.savetxt('temp.txt',U,fmt='%0.3f')
    stpoints=np.array([[1,1,1,1,1,1,1,1],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]])#np.array([Points[0,0,:],Points[1,0,:]])
    # plt.streamplot(Points[0][:,0],Points[1][0,:], U, V, color=speed,linewidth=lw, cmap='Spectral',density=4)#,minlength=dx/10)
    plt.streamplot(Points[0][:,0],Points[1][0,:], U.T, V.T, color=speed,linewidth=lw, cmap='coolwarm',density=5)#,minlength=dx/10)
    # plt.colorbar(orientation='horizontal')
    xmin,xmax=min(Points[0][:,0]),max(Points[0][:,0])
    ymin,ymax=np.min(Points[1][0,:]),np.max(Points[1][0,:])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.show()    

def Quiver(U_name,V_name,time=-1,show='yes',grid='no'):
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
    if grid=='no':
        Points=copy.deepcopy(Point_average(Points))
        U=copy.deepcopy(averaging(U,1))
        V=copy.deepcopy(averaging(V,0)) 
    else:
        U=copy.deepcopy(averaging_grid(U,1))
        V=copy.deepcopy(averaging_grid(V,0)) 
    #speed=np.sqrt(U**2 + V**2).T
    speed=np.sqrt(U**2 + V**2).T
    #speed=1    
    lw=1*speed**0.3/np.max(speed)
    
    plt.title('Streamlines')
    plt.gca().set_aspect('equal') 

    # print(Points[0][:,0].shape)
    # print(Points[1][0,:].shape)
    # sys.exit()

    #plt.quiver(Points[0][:,0],Points[1][0,:], U.T, V.T, color=speed,linewidth=lw, cmap='coolwarm',density=4)#,minlength=dx/10)
    plt.quiver([Points[0][:,0],Points[1][0,:]], U.T, V.T)#,minlength=dx/10)
    
    plt.colorbar(orientation='horizontal')
    xmin,xmax=min(Points[0][:,0]),max(Points[0][:,0])
    ymin,ymax=min(Points[1][0,:]),max(Points[1][0,:])
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.show()    


def Grid_plot(show='no'):
    dx=read_scalar('Constant/Dx.txt')
    dy=read_scalar('Constant/Dy.txt')
    nx=dx.shape[0]
    ny=dx.shape[1]
    
    Points=np.zeros([2,nx,ny])
    for i in range(1,nx):
        for j in range(ny):
            Points[0,i,j]=Points[0,i-1,j]+dx[i-1,j]
    for i in range(nx):
        for j in range(1,ny):
            Points[1,i,j]=Points[1,i,j-1]+dy[i,j-1]
    # sh=Points.shape
    for i in range(nx):
        for j in range(ny-1):
            x=[Points[0,i,j],Points[0,i,j+1]]
            y=[Points[1,i,j],Points[1,i,j+1]]
            plt.plot(x,y,c='k',lw=5.0/ny,zorder=6)
    
    for j in range(nx-1):
        for i in range(ny):
            x=[Points[0,j,i],Points[0,j+1,i]]
            y=[Points[1,j,i],Points[1,j+1,i]]
            plt.plot(x,y,c='k',lw=5.0/ny,zorder=6)
    

    if show=='no':
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.savefig('grid.png',dpi=1200,bbox_inches='tight')
        print('Grid saved !')   
    elif show=='yes':
        plt.gca().set_aspect('equal')
        plt.show()
        
    return 0

# Grid_plot()
# Contour('U','0',P='yes')

#Contour('V','0.026888',grid='yes')
# Contour('U',grid='yes')

# Contour('V',grid='yes')
# Contour('phi',grid='yes')
# Contour('T',grid='yes')
# Contour('phi',grid='yes')

# Contour('V',P='yes')

# Streamlines('U','V',grid='yes')
#Streamlines('U','V',grid='yes')

# Quiver('U','V','0',grid='yes')