
import numpy as np


class SolnData():
   "Data for the Solution"

   def __init__(self, pdat):

      self.u = np.zeros(pdat.dim, dtype=np.float64)
      
      self.U = np.zeros(pdat.dimslow, dtype=np.float64)
      self.Uout = np.zeros((pdat.dim,int(pdat.Tend/pdat.dtsubs)), dtype=np.float64)
      self.xu = np.zeros(pdat.dimslow, dtype=np.float64)
      
      

    #---------------------
    #Dam Break Initial Condition
      for kk in range(pdat.dim):
         xx = pdat.L/pdat.dim * kk
         #self.u[kk] = 0
         self.u[kk]= np.sin(2*np.pi*xx/pdat.L)+np.sin(4*np.pi*xx/pdat.L)
         #self.h[kk] = 10
         #self.h[kk]=10+0.01*np.sin(2*np.pi*xx/pdat.L)+0.01*np.sin(4*np.pi*xx/pdat.L)
         #if (kk <= (pdat.dim/2)):
         #    self.h[kk]=1
         #else:
         #    self.h[kk]=0.5
         #self.hu[kk]=self.h[kk]*self.u[kk]


        
      self.u = np.random.normal(0,1,pdat.dim)
      for i in range(int(pdat.dimslow/2)):
          summ = 0
          for j in range(2*pdat.averwindow-1):
              summ = summ + self.u[2*i*pdat.averwindow + j]
           
          self.u[2*(i+1)*pdat.averwindow - 1] = - summ

      ener_initial = np.sum(self.u**2)/pdat.dim*pdat.L 
      print(ener_initial)
      self.u = np.sqrt(1.716/ener_initial)*self.u

    
      for k in range(pdat.dimslow):
          averu = 0.0
          for j in range(pdat.averwindow):
              indx = j + k*pdat.averwindow
              averu += self.u[indx]
          self.xu[k] = averu/pdat.averwindow
      
      self.U=self.xu

      
      print("IC for u = ", self.u, "\n\n")
      
      print("IC for U, xu = ", self.U, "\n\n")
      
      self.fftu    = np.zeros(pdat.dim21, dtype=np.float64)
      self.fftxu    = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.fftU    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.fftu  = np.fft.rfft(self.u) / pdat.dim
      self.fftxu  = np.fft.rfft(self.xu) / pdat.dimslow
      self.fftU  = np.fft.rfft(self.U) / pdat.dimslow
      
      self.netfluxu    = np.zeros(pdat.dim, dtype=np.float64)
      self.netfluxxu    = np.zeros(pdat.dimslow, dtype=np.float64)
      self.netfluxU    = np.zeros(pdat.dimslow, dtype=np.float64)
      
      self.force    = np.zeros(pdat.dim, dtype=np.float64)
      self.alpha = np.zeros(3, dtype=np.float64)
      self.phi = np.zeros(3, dtype=np.float64)
     
      self.xforce    = np.zeros(pdat.dimslow, dtype=np.float64)
      self.yu = np.zeros(pdat.dim, dtype=np.float64)
      #self.yuout = np.zeros((int(2*pdat.dimslow),int(pdat.Tend/pdat.dtsubs)), dtype=np.float64)

      
          

      
# -----------------------------------
# End class SolnData
# -----------------------------------
