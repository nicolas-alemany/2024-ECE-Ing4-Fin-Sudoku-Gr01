# import matplotlib.pyplot as plt
# from timeit import default_timer
import numpy as np
# from sinkhorn_knopp import sinkhorn_knopp as skp
import math

class Constraints:
    c = np.array([]) # [3N][N]
    m = np.array([]) # [N^2][3]
    p = np.array([]) # [N^2][N]
    pq = np.array([]) 
    r = np.array([]) #P(C(m)|s(n)=x) [3N][N^2][N]
    q = np.array([]) #P(Sn = x|all the constraints except Cm involving Sn are satisfied)
    cq = np.array([])
    s =  np.array([])
    n_c = np.array([])
    size = 9
    CellDomain = list(range(1, size+1))
    CellIndices = list(range(size**2))
    sk=[]
    memo = {}

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
        self.pq=np.zeros((self.size**2,self.size))
        self.n_c=np.ones_like(c)*-1

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
                for j in self.m[i]:
                     self.cq[self.c==j]=self.s[i]
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
        maxP = float("-inf")
        maxIndex = 0

        for n in range(len(self.s)):
            if self.s[n] == 0:
                for v in range(self.size):
                    if 0 < self.p[n][v] < 1:
                        if self.p[n][v] > maxP:
                            maxP = self.p[n][v]
                            maxIndex = n

        if self.s[maxIndex] != 0:
            return -1

        maxRP = float("-inf")
        value = 0
        relatedCells = [-1] * (2*(self.size-1)+((int(np.sqrt(self.size)))-1)**2)
        r_index = 0

        for C in range(len(self.m[0])):
            m = self.m[maxIndex][C]
            for i in range(self.size):
                if self.s[self.c[m][i]] == 0:
                    flag = True
                    for ii in range(r_index):
                        if relatedCells[ii] == self.c[m][i]:
                            flag = False
                            break
                    if flag:
                        relatedCells[r_index] = self.c[m][i]
                        r_index += 1

        for v in range(self.size):
            if self.p[maxIndex][v] > 0:
                sum_val = float("-inf")
                i = 0
                while relatedCells[i] != -1:
                    sum_val = self.sum_log(sum_val, self.p[relatedCells[i]][v])
                    i += 1
                self.pq[maxIndex][v] = self.p[maxIndex][v] - sum_val
                if self.pq[maxIndex][v] > maxRP:
                    maxRP = self.pq[maxIndex][v]
                    value = v + 1

        if value == 0:
            return -1

        self.s[maxIndex] = value
        for v in range(self.size):
            self.p[maxIndex][v] = float("-inf")

        relatedCells = [-1] * 20
        r_index = 0

        for C in range(len(self.m[0])):
            m = self.m[maxIndex][C]
            for i in range(9):
                if self.s[self.c[m][i]] == 0 and self.p[self.c[m][i]][value - 1] > 0 and self.c[m][i] != maxIndex:
                    flag = True
                    for ii in range(r_index):
                        if relatedCells[ii] == self.c[m][i]:
                            flag = False
                            break
                    if flag:
                        relatedCells[r_index] = self.c[m][i]
                        r_index += 1

        i = 0
        while relatedCells[i] != -1:
            num = sum(1 for v in range(9) if self.p[relatedCells[i]][v] > 0)
            distribute = self.p[relatedCells[i]][value - 1] - math.log10(num - 1)
            self.p[relatedCells[i]][value - 1] = float("-inf")
            for v in range(9):
                if self.p[relatedCells[i]][v] > 0:
                    self.sum_log(self.p[relatedCells[i]][v], distribute)
            i += 1

        return 1



    def message_passing(self):
        # R[m][n][x]=* # (1 - q[m][n'][value]) other 0 cell related in the same C
        
        for n in range(81):
            if self.s[n] == 0:
                for x in range(9):
                    if self.p[n][x] != -float('inf'):  # possible values!
                        for C in range(3):
                            m = self.m[n][C]
                            self.r[m][n][x] = 0  # Initialize R[m][n][x] to 1
                            
                            vp = self.get_possible_vp_r(m, n, x)
                            
                            possible_vp = vp.split(",")
                            v = possible_vp[0]
                            p = possible_vp[1].strip().split(" ")
                            sum_val = self.permutate(n, "", v, p, -float('inf'), m)#
                            self.r[m][n][x] = sum_val  # Update R[m][n][x] with the calculated sum

        # Q[m][n][x]= P(n=x) * # R[m'][n][x], m'= other two
        for n in range(81):
            if self.s[n] == 0:
                for x in range(9):
                    if self.p[n][x] != -float('inf'):  # possible values!
                        for C in range(3):
                            m = self.m[n][C]
                            self.q[m][n][x] = self.p[n][x]  # P(n=x)
                            for C_other in range(3):
                                if C_other != C:
                                    m_2 = self.m[n][C_other]
                                    self.q[m][n][x] += self.r[m_2][n][x]

        # P[n][x]
        for n in range(81):
            if self.s[n] == 0:
                for x in range(9):
                    if self.p[n][x] != -float('inf'):  # possible values!
                        for C in range(3):
                            self.p[n][x] += self.r[self.m[n][C]][n][x]
    

     

    def permutate(self,n, pre, last, position, sum_val, m):
        if len(last) == 0:
            if pre in self.memo:
                return self.memo[pre]
            product = 0
            for i in range(len(pre)):
                value = int(pre[i])
                indexS = int(position[i])
                product += self.q[m][indexS][value]
            sum_val = self.sum_log(sum_val, product)
            self.memo[pre] = sum_val
            return sum_val
        for i in range(len(last)):
            sum_val = self.permutate(n, pre + last[i], last[:i] + last[i+1:], position, sum_val, m)
        return sum_val


    
    def get_possible_vp_r(self,m, indexS, value):
        result = ""
        for i in range(9):
            flag = True
            index_NC = 0
            while self.n_c[m][index_NC] != -1 and flag:
                if (i + 1) == self.s[self.n_c[m][index_NC]]:
                    flag = False
                index_NC += 1
                
            if flag and i != value:
                result += str(i)
        result += ","
        for i in range(9):
            flag = True
            index_NC = 0
            while self.n_c[m][index_NC] != -1 and flag:
                if self.c[m][i] == self.n_c[m][index_NC]:
                    flag = False
                index_NC += 1
            if flag and self.c[m][i] != indexS:
                result += str(self.c[m][i]) + " "
        return result.strip()


    def sum_log(a, b):
        return math.log10(math.pow(10, a) + math.pow(10, b))


    def sum_log(self,a, b):
        if a == float("-inf"):
            return b
        elif b == float("-inf"):
            return a
        else:
            if a > b:
                x = a
                y = b
            else:
                x = b
                y = a
            decide = 10 ** (x - y)
            if (decide + 1) == float("inf"):
                c = x
            else:
                decide += 1
                c = y + math.log10(decide)
            return c



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
for i in range(100):
    #t.get_grid_p_c()
    #t.grid_p()
    t.message_passing()
t.guess()
# print()
# t.print_s()


# Redimensionner t.s pour qu'il soit une matrice size x size
solution_2d = t.s.reshape((9, 9))
# Convertir la matrice numpy en tableau .NET
r = [asNetArray(solution_2d[i]) for i in range(9)]
