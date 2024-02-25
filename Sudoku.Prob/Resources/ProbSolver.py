import matplotlib.pyplot as plt
from timeit import default_timer
import numpy as np
from sinkhorn_knopp import sinkhorn_knopp as skp


class Constraints:
    c = np.array([]) # [3N][N]
    m = np.array([]) # [N^2][3]
    p = np.array([]) # [N^2][N]
    r = np.array([]) #P(C(m)|s(n)=x) [3N][N^2][N]
    q = np.array([]) #P(Sn = x|all the constraints except Cm involving Sn are satisfied)
    cq = np.array([])
    s =  np.array([])
    size = 9
    CellDomain = list(range(1, size+1))
    CellIndices = list(range(size**2))
    sk=[]

    def __init__(self, size = 9):
        c = []
        self.size = size
        self.CellDomain = list(range(1, size+1))
        self.CellIndices = list(range(size**2))
        self.sk = skp.SinkhornKnopp(max_iter=1) 

        for i in range(size):
            c.append(list(range(i * size, (i + 1) * size)))
        c = np.array(c)
        c = np.vstack((c, c.T))
        for i in range(size):
                tmp =[]
                for j in range(int(np.sqrt(size))):
                        tmp.append(list(range(j*size+i%int(np.sqrt(size))*int(np.sqrt(size))+size*int(np.sqrt(size))*int(i/np.sqrt(size)),j*size+i%int(np.sqrt(size))*int(np.sqrt(size))+size*int(np.sqrt(size))*int(i/np.sqrt(size))+int(np.sqrt(size)))))
                c = np.vstack((c,np.ravel(tmp))) 
        self.c=c
        m=np.empty((0,int(np.sqrt(size))))   

        for i in range(size**2):
              tmp=np.array([])
              for j in range(len(self.c)):
                    if i in c[j]:
                          tmp=np.append(tmp,j)
              m=np.vstack((m,tmp))
        self.m= m.astype(int)

        self.p=np.zeros((self.size**2,self.size))
        self.r=np.zeros((self.size*3,self.size**2,self.size))
        self.q=np.zeros((self.size*3,self.size**2,self.size))
        self.cq=np.zeros((self.size*3,self.size,self.size))

    def Nh(self,S):
        nh=np.empty((0,int(self.size)))   
        
        for i in self.m[S]:
              nh=np.vstack((nh,self.c[i]))
        return np.unique(nh[nh!=S]).astype(int)
    
    def read(self,s):
         self.s=s

    def innit_p(self):
          for i in range(len(self.s)):
             tmp=[]
             if self.s[i]== 0:
                tmp = np.ones(self.size)
             else :   
                tmp = np.ones(self.size)  
                mask = np.ones_like(tmp)
                mask[np.array([ self.s[i]])-1] = 0
                tmp[mask.astype(bool)] = 0
             self.p[i]=tmp/sum(tmp)
             
    def grid_p(self):
          for i in range(len(self.p)):
                for j in range(t.size):
                    self.p[i][j]=self.get_p(i,j)

    def get_p(self,cell,x):
         
         newp=self.p[cell][x]
         for i in self.m[cell]:
               newp*=self.cq[i][self.c[i]==cell][0][x]
         return newp
         
          
    
    def solve(self):
          return self.s
    
    def get_grid_p_c(self):
          for i in range(len(self.c)):
                self.get_p_c(i)
    def get_p_c(self,c):
          imp= []
          for i in self.c[c]:
                if self.s[i] != 0:
                      imp.append(self.s[i])
          p = []
          for i in range(len(self.c[c])):
                tmp=[]
                if self.s[self.c[c][i]]== 0:
                    tmp = np.zeros(self.size)
                    mask = np.ones_like(tmp)
                    mask[np.array(imp)-1] = 0
                    tmp[mask.astype(bool)] = 1
                else :
                      
                      tmp = np.ones(self.size)  
                      mask = np.ones_like(tmp)
                      mask[np.array([ self.s[self.c[c][i]]])-1] = 0
                      tmp[mask.astype(bool)] = 0
                self.cq[c][i]=tmp/sum(tmp)


    def guess(self):
          for i in range(len(self.p)):
            max_value = np.max(self.p[i])
            indices = np.where(self.p[i] == max_value)[0]
            if len(indices)==1:
                self.s[i]=indices[0]+1

    def sss(self):
          #while True:
                for i in range(len(self.cq)): 
                      self.cq[i]=self.sk.fit(self.cq[i])     
                
          

    def print_s(self):
          for i in range(self.size):
                print(self.s[i*self.size:(1+i)*self.size])
    
# instance = np.array([0,0,0,5,9,2,8,1,0,2,0,4,0,7,3,0,0,0,0,5,0,0,1,0,0,0,3,0,3,2,1,0,0,0,9,0,0,4,0,9,0,7,0,3,0,0,6,0,0,0,5,1,4,0,1,0,0,0,4,0,0,2,0,0,0,0,3,5,0,9,0,7,0,9,5,7,2,8,0,0,0])
# conversion de la variable d'entrée dans le format désiré                
sudoku = asNumpyArray(instance)

#résolution                
t=Constraints()
t.read(sudoku)
t.innit_p()
# t.print_s()
for i in range(10):
    t.get_grid_p_c()
    t.grid_p()
    t.guess()
# print()
# t.print_s()


# Redimensionner t.s pour qu'il soit une matrice size x size
solution_2d = t.s.reshape((9, 9))
# Convertir la matrice numpy en tableau .NET
r = [asNetArray(solution_2d[i]) for i in range(9)]
