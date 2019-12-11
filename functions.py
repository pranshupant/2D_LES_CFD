import numpy as np,os,copy,sys
import sympy as sp

def write_points(fileout,Points):
    d=6
    nx=len(Points[0,:,0])
    ny=len(Points[0,0,:])
    file=open(fileout,'w')
    file.write('[%i,%i]\n'%(nx,ny))
    for j in range(ny):
        for i in range(nx):
            file.write('[%i,%i]:%s,%s'%(i,j,round(Points[0,i,j],d),round(Points[1,i,j],d)))
            if j<ny and i<nx:
                file.write('\n')

def read_points(filein):
    
    with open(filein) as tsvfile:
        iterant=tsvfile.read().strip().split('\n')
        n=iterant[0].strip('[').strip(']').split(',')
        nx=int(n[0])
        ny=int(n[1])
        Points_read=np.zeros([2,nx,ny])
        for row in iterant[1:]:
            row=row.split(':')
            ind=row[0].strip('[').strip(']').split(',')
            xy=row[1].strip('(').strip(')').split(',')
            i=int(ind[0])
            j=int(ind[1])
            Points_read[0,i,j]=float(xy[0])
            Points_read[1,i,j]=float(xy[1])   
    return Points_read    

def deltas_write(Points_read):
    nx=len(Points_read[0,:,0])
    ny=len(Points_read[0,0,:])
    dx=np.zeros([nx,ny])
    dy=np.zeros([nx,ny])
    for i in range(nx-1):
        for j in range(ny-1):
            dx[i,j]=Points_read[0,i+1,j]-Points_read[0,i,j]
            dy[i,j]=Points_read[1,i,j+1]-Points_read[1,i,j]
    dx[0,:]=copy.deepcopy(dx[1,:])
    dx[-1,:]=copy.deepcopy(dx[-2,:])
    dx[:,0]=copy.deepcopy(dx[:,1])
    dx[:,-1]=copy.deepcopy(dx[:,-2])
    dy[:,0]=copy.deepcopy(dy[:,1])
    dy[:,-1]=copy.deepcopy(dy[:,-2]) 
    dy[0,:]=copy.deepcopy(dy[1,:])
    dy[-1,:]=copy.deepcopy(dy[-2,:])
    return dx,dy

def deltas_write_nonuni(nx,ny,xmax,ymax,dyf):
    # To find expansion ratio in the y direction from ymax,first cell;dyf and number of cells ny. 
    e=sp.Symbol('e')
    y=dyf*e**ny-ymax*e+(ymax-dyf)
    y_=sp.diff(y)
    x1=10
    err=1e-6
    i=1
    while 1:
        x2=x1-y.subs(e,x1)/y_.subs(e,x1)
        if np.abs(x2-x1)<err or i>100:
            break
        else:
            x1=x2
    e=x2
    dx=np.ones([nx,ny])*xmax/(nx-1)
    dy=np.zeros([nx,ny])
    for i in range(nx):
        for j in range(ny):
            dy[i,j]=dyf*e**j
    print(e)
    # print(dy)
    # print(dx)
    return dx,dy



def write_scalar(fileout,Scalar):
    d=6
    nx=len(Scalar[:,0])
    ny=len(Scalar[0,:])
    file=open(fileout,'w')
    file.write('[%i,%i]\n'%(nx,ny))
    for j in range(ny):
        for i in range(nx):
            file.write('[%i,%i]:%s'%(i,j,round(Scalar[i,j],d)))
            if j<ny and i<nx:
                file.write('\n')

def read_scalar(filein,ch=0):
    with open(filein) as tsvfile:
        iterant=tsvfile.read().strip().split('\n')
        n=iterant[0].strip('[').strip(']').split(',')
        nx=int(n[0])
        ny=int(n[1])
        Scalar=np.zeros([nx,ny])    
        for row in iterant[1:]:
            row=row.split(':')
            ind=row[0].strip('[').strip(']').split(',')
            xy=row[1].strip('(').strip(')').split(',')
            i=int(ind[0])
            j=int(ind[1])
            Scalar[i,j]=float(xy[0])   
    if ch==0:
        return Scalar
    elif ch==1:
        Dict={}
        for i in range(nx):
            for j in range(ny):
                Dict[(i,j)]=Scalar[i,j]
        return Dict


def write_all_scalar(P,T,U,V,phi,t='trash'):
    try:
        path='Results/%.6f/'%t
        os.makedirs(path)
        write_scalar(path+'P.txt',P)
        write_scalar(path+'U.txt',U)
        write_scalar(path+'V.txt',V)
        write_scalar(path+'T.txt',T)
        write_scalar(path+'phi.txt',phi)

        np.savetxt(path+'P_.txt',P, delimiter='\t',fmt='%.3f')
        np.savetxt(path+'U_.txt',U, delimiter='\t',fmt='%.3f')
        np.savetxt(path+'V_.txt',V, delimiter='\t',fmt='%.3f')
        np.savetxt(path+'T_.txt',T, delimiter='\t',fmt='%.3f')
        np.savetxt(path+'phi_.txt',phi, delimiter='\t',fmt='%.3f')
        print('## WRITTEN : time=%.6f'%t)
    except:
        print("Failed to write !")

def read_all_scalar(ch=0):
    path='0/'
    files=os.listdir(path)
    files.sort()
    # print(files)
    X=[]
    for paths in files:
        try:
            X.append(read_scalar(path+paths,ch))
        except:
            continue
    return X

def read_delta(ch=0):
    path='Constant/'
    Dx=read_scalar(path+'Dx.txt',ch)
    Dy=read_scalar(path+'Dy.txt',ch)
    return Dx,Dy



# read_all_scalar()