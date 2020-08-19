import numpy as np
import matplotlib.pyplot as plt
class Atom:
    
    def __init__(self,x,z,vx,vz,a,m):
        self.x = x 
        self.z = z 
        self.vx = vx
        self.vz = vz
        self.a = a
        self.m = m
            
class Layer:
    
    def __init__(self,xinit,z,count,dist,m,v):
        self.v = v
        self.m = m
        self.dist = dist
        self.z = z
        self.xinit = xinit
        self.count = count

        self.atoms = []
        for c in range(self.count):
            self.atoms.append(Atom(self.xinit+c*self.dist,self.z,self.v,0,0,self.m)) # layer vy dusun 

        self.x = []
        for s in range(self.count):
            self.x.append(self.atoms[s].x)

    def update(self):
        self.x.clear()
        for s in range(self.count):
            self.x.append(self.atoms[s].x)
        
class Agent:
    def __init__(self,x,v):
        self.v=v
        self.x=x

def LF(dist,atom,layer,sigma,epsilon):
    term1 = (4)*epsilon*(-12*(sigma**12/(dist**13))+6*(sigma**6/(dist**7)))
    term11 = (atom.z-np.array(layer.z))/dist
    ay = sum(term1*term11)/atom.m

    return ay

def distl(layer,atom):#layerdaki her atomun object ile arasındaki mesafeyi ölçüyor
    d = []
    for i in range(len(layer.x)):
        dista = np.sqrt((layer.atoms[i].x-atom.x)**2+(layer.atoms[i].z-atom.z)**2)
        d.append(dista)    
    return np.array(d)

def LA(dist,atom,layer,epsilon,sigma,k,k2):
    term1 = (-4)*epsilon*(-12*(sigma**12/(dist**13))+6*(sigma**6/(dist**7)))
    term11 = (np.array(layer.x)-atom.x)/dist
    #son termü ekle
    lx = layer.x
    al = []
    
    for i in range(len(layer.x)):
        term2 = k*(layer.x[i]-lx[i])
        term3 = 2*k2*layer.x[i]
        
        if i == 0:
            term4 = 0
            
        elif i == len(layer.x)-1:
            term4 = 0
            
        else:
            term4 = k2*(layer.x[i+1]-layer.x[i-1])
        al.append(-term2-term3+term4)
        
    al = np.array(al)
    al = -term1*term11+al
    return al/atom.m 

def dista(atom,layer):#objectin tüm atomlarla arasındaki mesafeyi ölçüyor
    dist =[]
    for p in range(len(layer.x)):
        dist.append(np.sqrt((atom.x-layer.x[p])**2+(atom.z-layer.z)**2))
    return np.array(dist) 
def AA(atom,agent,layer,sigma,epsilon,k,dist):
    term1 = (-4)*epsilon*(-12*(sigma**12/(dist**13))+6*(sigma**6/(dist**7)))
    term11 = (atom.x-np.array(layer.x))/dist
    term2 = k*(atom.x-agent.x)

    a = -(sum(term1*term11)+term2)/atom.m
    return a

def motion(time,dt,atom,layer,agent,sigma,epsilon,k,k2):
    layer_x = []
    atom_x = []
    atom_z = []
    
    for t in range(int(time/dt)):
        distat = dista(atom,layer)
        distla = distl(layer,atom)

        aa = AA(atom,agent,layer,sigma,epsilon,k,distat)
        la = LA(distla,atom,layer,epsilon,sigma,k,k2)
        lf = LF(distat,atom,layer,sigma,epsilon)

        atom.vx = atom.vx +  aa*dt
        atom.x = atom.x + atom.vx

        atom.vz = atom.vz +  lf*dt
        atom.z = atom.z + atom.vz
        
        atom_x.append(atom.x)
        atom_z.append(atom.z)
        agent.x = agent.x + agent.v*dt
        tx = []
        for atms in range(len(layer.atoms)):
            layer.atoms[atms].vx = layer.atoms[atms].vx + la[atms]*dt
            layer.atoms[atms].x = layer.atoms[atms].x + layer.atoms[atms].vx*dt
            tx.append(layer.atoms[atms].x)

        layer_x.append(tx)

    return atom_x,atom_z,layer_x

time = 2
dt = .01
k = 10
k2 = 10
sigma = 7.17868959
epsilon = 1e-12
atom = Atom(50,100,.00005,.05,1,0.1)
layer = Layer(1,0,1000,1,1,1)
agent = Agent(50.001,.001)
dl = distl(layer,atom)
da = dista(atom,layer)

# lf = LF(da,atom,layer,sigma,epsilon)

# la = LA(dl,atom,layer,epsilon,sigma,k,k2)
# aa = AA(atom,agent,layer,sigma,epsilon,k,da)
m = motion(time,dt,atom,layer,agent,sigma,epsilon,k,k2)

# print(aa)
# print(lfk)
# print(aa)

# print(m[1])
# print(m[0])

mt = np.rot90(m[2])
timea = np.arange(0,time,dt)
plt.plot(timea,m[0])

#for g in range(len(mt)):
 #   plt.plot(timea,mt[g])
plt.show()

