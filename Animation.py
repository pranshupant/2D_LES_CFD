import numpy as np,os,copy,sys
from functions import *
from contour import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

# def isfloat(value):
#   try:
#     float(value)
#     return True
#   except ValueError:
#     return False


def Animation(Scalar_name,grid='no'):

    def gridder(Scalar,Points,grid='no'):
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
        return Scalar,Points


    times=[]
    times.append('0/')
    files=os.listdir('Results')
    counts=[]
    for i in files:
        if isfloat(i):
            counts.append(float(i)) 
    counts.sort()
    k=0
    for i in counts:
        if k%10==0:
            times.append('Results/'+'%.6f'%i)
        
            # break
        k+=1
    # times.append('Results/'+i)
        

    Scalars=[None]*len(times)
    # print(times)
    for t in range(len(times)):
        Scalars[t]=read_scalar(times[t]+'/'+Scalar_name+'.txt')



    # sys.exit()




    # ax = plt.axes(xlim=(-15, 40), ylim=(-20,40)) 
    # ax = plt.axes() 


    # ax.set(ylim=(0, 1), xlim=(0, 10))

    contour_opts = {'levels': np.linspace(-9, 9, 10),
                    'cmap':'RdBu', 'lw': 2}

    # cax = ax.contour([], [], [], **contour_opts)


    # initialization function 
    def init(): 
        # creating an empty plot/frame 
        ax.collections = []
        # plt.ylabel(r'$y  \longrightarrow$') 
        # plt.xlabel(r'$x  \longrightarrow$')
        ax.contour(Points[0], Points[1],np.zeros(Scalars[0].shape))

        # return ax 

    # lists to store x and y axis points 
    # animation function 
    Points=read_points('Constant/Points.txt')

    # i=5
    # Scalar,Point=gridder(Scalars[i],Points,grid)
    # print(Scalar,Point)
    # print(Scalar.shape,Point.shape)

    # fig=plt.figure()
    # plt.contourf(Point[0],Point[1],Scalar)
    # plt.show()
    # plt.close()
    # sys.exit()

    fig = plt.figure() 
    xmin,xmax=min(Points[0][:,0]),max(Points[0][:,0])
    ymin,ymax=np.min(Points[1][0,:]),np.max(Points[1][0,:])
    
    ax = plt.axes(xlim=(xmin,xmax), ylim=(ymin,ymax)) 

    #contour_opts = {'levels': np.linspace(-9, 9, 10),
                    # 'cmap':'RdBu', 'lw': 2}

    Scalar,Point=gridder(Scalars[0],Points,grid)
    # cax=ax.contourf(Point[0],Point[1],Scalar,cmap='coolwarm',vmin=np.min(Scalars),vmax=np.max(Scalars))
    cmapp='gist_yarg'
    cmapp='hot'

    cax=ax.contourf(Point[0],Point[1],Scalar,100,cmap=cmapp,vmin=np.min(Scalars),vmax=np.max(Scalars))

    def animate(i): 
        ax.collections=[]
        #cbar=[],cax=[]
        Scalar,Point=gridder(Scalars[i+1],Points,grid)
        plt.gca().set_aspect('equal')
        print(i)
        cax=ax.contourf(Point[0], Point[1],Scalar,100,cmap=cmapp,vmin=np.min(Scalars),vmax=np.max(Scalars))#, **contour_opts)


    anim = animation.FuncAnimation(fig, animate,frames=len(times)-1, interval=1) 

    # cbar   = fig.colorbar(cax,orientation='horizontal')


    plt.draw()
    plt.show()

    save={'bbox_inches=':'tight'}

    # anim.save('%s.gif'%Scalar_name,savefig_kwargs=save,dpi=600) 
    #anim.save('%s.mp4'%Scalar_name,fps=10) 

    plt.close()



# plt.figure(figsize=(10,10))
#Animation('T','yes')
# Animation('U','yes')
Animation('phi','yes')

# Animation('T','yes')
