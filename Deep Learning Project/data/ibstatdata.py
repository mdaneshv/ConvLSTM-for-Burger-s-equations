import numpy as np
import math
#import time as walltime
import pickle
import ast
#import sys


class StatData:
   "Data for Statistics"

   def __init__(self, pdat):
      "Allocate Arrays for computing Stat"
       # one-point stat

      self.meanu = np.zeros(pdat.dim, dtype=np.float64)
      self.varu  = np.zeros(pdat.dim, dtype=np.float64)
      self.ex3u = np.zeros(pdat.dim, dtype=np.float64)
      self.ex4u = np.zeros(pdat.dim, dtype=np.float64)
      
      self.meanU = np.zeros(pdat.dimslow, dtype=np.float64)
      self.varU  = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex3U = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex4U = np.zeros(pdat.dimslow, dtype=np.float64)
      
    
      self.meanxu = np.zeros(pdat.dimslow, dtype=np.float64)
      self.varxu  = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex3xu = np.zeros(pdat.dimslow, dtype=np.float64)
      self.ex4xu = np.zeros(pdat.dimslow, dtype=np.float64)
      
      self.meanBu = np.zeros(pdat.dimslow, dtype=np.float64)
      self.varBu  = np.zeros(pdat.dimslow, dtype=np.float64)
      
      self.stat_counter = 0.0
      self.stat_counter_param = 0.0

      self.cfu     = np.zeros((pdat.cfdim, pdat.cflen), dtype=np.float64)
      self.cfsolnu = np.zeros((pdat.cfdim, pdat.cflen), dtype=np.float64)
      self.cfxu     = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      self.cfsolnxu = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      self.cfU     = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      self.cfsolnU = np.zeros((pdat.cfdimslow, pdat.cflen), dtype=np.float64)
      
      self.cfu_soln_filled = False
      self.stat_counter_cfu = 0.0
      self.cfu_index = 0
      self.timecfu = 0.0

      
      self.cfU_soln_filled = False
      self.stat_counter_cfU = 0.0
      self.cfU_index = 0
      self.timecfU = 0.0

      self.cfxu_soln_filled = False
      self.stat_counter_cfxu = 0.0
      self.cfxu_index = 0
      self.timecfxu = 0.0
      
      self.meanfftu = np.zeros(pdat.dim21, dtype=np.float64)
      self.specu    = np.zeros(pdat.dim21, dtype=np.float64)
      
      self.meanfftU = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.specU    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.meanfftxu = np.zeros(pdat.dimslow21, dtype=np.float64)
      self.specxu    = np.zeros(pdat.dimslow21, dtype=np.float64)
      
      self.spec_counter = 0.0
      
      self.pdfpos  = np.zeros((pdat.pdfdim, pdat.pdflen), dtype=np.float64)
      self.pdfneg  = np.zeros((pdat.pdfdim, pdat.pdflen), dtype=np.float64)

      

   # -----------------------------------
   def compute_specall(self, fftu, fftU, fftxu):
      "Compute Averaged Spectra"

      self.spec_counter += 1.0
  
      self.meanfftu = self.meanfftu + fftu
      self.specu    = self.specu + np.absolute(fftu)**2
      self.meanfftU = self.meanfftU + fftU
      self.specU    = self.specU + np.absolute(fftU)**2   
      self.meanfftxu = self.meanfftxu + fftxu      
      self.specxu    = self.specxu + np.absolute(fftxu)**2   


   # -----------------------------------
   def output_spec(self, filenameu, filenameu2):
      "Output Averaged Spectra U"

      with open(filenameu, 'w') as handle:
         for k in self.specu:
            tmp3 = k / self.spec_counter
            handle.write(str(tmp3))
            handle.write(" ")

      with open(filenameu2, 'w') as handle:
         for k in self.meanfftu:
            tmp4 = k / self.spec_counter
            handle.write(str(tmp4))
            handle.write(" ")

   # -----------------------------------
   def output_specslow(self, filenameu, filenameu2):
      "Output Averaged Spectra xu"


      with open(filenameu, 'w') as handle:
         for k in self.specxu:
            tmp3 = k / self.spec_counter
            handle.write(str(tmp3))
            handle.write(" ")

      with open(filenameu2, 'w') as handle:
         for k in self.meanfftxu:
            tmp4 = k / self.spec_counter
            handle.write(str(tmp4))
            handle.write(" ")

   # -----------------------------------
   def output_specaverage(self, filenameu, filenameu2):
      "Output Averaged Spectra U"

        
      with open(filenameu, 'w') as handle:
         for k in self.specU:
            tmp3 = k / self.spec_counter
            handle.write(str(tmp3))
            handle.write(" ")

      with open(filenameu2, 'w') as handle:
         for k in self.meanfftU:
            tmp4 = k / self.spec_counter
            handle.write(str(tmp4))
            handle.write(" ")


   # -----------------------------------
   # One Point Stat = mean and var
   # -----------------------------------
   def compute_onep_stat(self, u):
      "Compute One-Point Stat"

      self.meanu = self.meanu + u
      self.varu = self.varu + np.square(u)
      self.ex3u = self.ex3u + np.power(u,3)
      self.ex4u = self.ex4u + np.power(u,4)
      
      self.stat_counter += 1.0

   def print_onep_stat(self, filenameu, pdat):
      "Print One Point Stat"
      print("-- Stat Counter    = ", self.stat_counter)   

      print("Mean(u) = ", self.meanu/self.stat_counter)
      print("Var(u)  = ", self.varu/self.stat_counter - (self.meanu/self.stat_counter)**2)
      print("E[u^3] = ", self.ex3u/self.stat_counter)
      print("E[u^4] = ", self.ex4u/self.stat_counter)
      
      with open(filenameu, 'w') as handle:
         for idim in range(pdat.dim):
            handle.write(str(idim))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dim):
            tmp = self.meanu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dim):
            tmp = self.varu[idim]/self.stat_counter - (self.meanu[idim]/self.stat_counter)**2
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dim):
            tmp = self.ex3u[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dim):
            tmp = self.ex4u[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
 
       
   # -----------------------------------
   def compute_all_onep_stat(self, u, pdat):
      "WRAPPER: Compute all onep stat"

      self.compute_onep_stat(u)
      
      
    # -----------------------------------
   def output_all_stat(self, pdat):
      "WRAPPER: Output all stat"

      self.print_onep_stat("statu.dat", pdat)
      self.print_cft("cftu.dat", pdat)

   # -----------------------------------
   # One Point Stat Slow = mean and var of slow variables
   # -----------------------------------
   def compute_onep_statslow(self, xu):
      "Compute One-Point Stat"

      self.meanxu = self.meanxu + xu
      self.varxu = self.varxu + np.square(xu)
      self.ex3xu = self.ex3xu + np.power(xu,3)
      self.ex4xu = self.ex4xu + np.power(xu,4)
      
      #self.stat_counter += 1.0  #uncomment if only slow is needed to be output

   def print_onep_statslow(self, filenameu, pdat):
      "Print One Point Stat"
      print("-- Stat Counter    = ", self.stat_counter)   

      
      print("Mean(xu) = ", self.meanxu/self.stat_counter)
      print("Var(xu)  = ", self.varxu/self.stat_counter - (self.meanxu/self.stat_counter)**2)
      print("E[xu^3] = ", self.ex3xu/self.stat_counter)
      print("E[xu^4] = ", self.ex4xu/self.stat_counter)
          
      with open(filenameu, 'w') as handle:
         for idim in range(pdat.dimslow):
            handle.write(str(idim))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.meanxu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.varxu[idim]/self.stat_counter - (self.meanxu[idim]/self.stat_counter)**2
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex3xu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex4xu[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
 
       
   # -----------------------------------
   def compute_all_onep_statslow(self, xu, pdat):
      "WRAPPER: Compute all onep stat"

      self.compute_onep_statslow(xu)
      
      
    # -----------------------------------
   def output_all_statslow(self, pdat):
      "WRAPPER: Output all stat"

      self.print_onep_statslow("statxu.dat", pdat)
      self.print_cftslow("cftxu.dat", pdat)

   # -----------------------------------
   # One Point Stat Average = mean and var of local averages
   # -----------------------------------
   def compute_onep_stataverage(self, U):
      "Compute One-Point Stat"
  
      self.meanU = self.meanU + U
      self.varU = self.varU + np.square(U)
      self.ex3U = self.ex3U + np.power(U,3)
      self.ex4U = self.ex4U + np.power(U,4)
      
      #self.stat_counter += 1.0
   
   def print_onep_stataverage(self, filenameu, pdat):
      "Print One Point Stat"
      print("-- Stat Counter    = ", self.stat_counter)   
  
      print("Mean(U) = ", self.meanU/self.stat_counter)
      print("Var(U)  = ", self.varU/self.stat_counter - (self.meanU/self.stat_counter)**2)
      print("E[U^3] = ", self.ex3U/self.stat_counter)
      print("E[U^4] = ", self.ex4U/self.stat_counter)
      

         
      with open(filenameu, 'w') as handle:
         for idim in range(pdat.dimslow):
            handle.write(str(idim))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.meanU[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")

         for idim in range(pdat.dimslow):
            tmp = self.varU[idim]/self.stat_counter - (self.meanU[idim]/self.stat_counter)**2
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex3U[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
         
         for idim in range(pdat.dimslow):
            tmp = self.ex4U[idim]/self.stat_counter
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")
 
       
   # -----------------------------------
   def compute_all_onep_stataverage(self, U, pdat):
      "WRAPPER: Compute all onep stat"

      self.compute_onep_stataverage(U)
      
      
    # -----------------------------------
   def output_all_stataverage(self, pdat):
      "WRAPPER: Output all stat"

      self.print_onep_stataverage( "statU2.dat", pdat)
      self.print_cftaverage("cftU2.dat", pdat)



   # -----------------------------------
   # CF Time
   # -----------------------------------
   def record_cfsoln(self, u, pdat):
      "Record soln into cf_soln"

      self.cfsolnu[:,self.cfu_index] = u[0:pdat.cfdim]
      self.cfu_index += 1
      self.timecfu = 0.0


   # -----------------------------------
   def update_timecfu(self, dt):
      "Update timecf"
      self.timecfu += dt

   # -----------------------------------
   def compute_cft(self, u, pdat):
      "Compute CF Time"

      if self.cfu_soln_filled:
         if (self.timecfu >= pdat.cflag - pdat.dt05):
            #print("Computing CF, time = ", pdat.time)

            # compute CF
            for ilen in range(pdat.cflen):
               self.cfu[:,ilen]  = self.cfu[:,ilen]  + self.cfsolnu[:,0] * self.cfsolnu[:,ilen]
            self.stat_counter_cfu += 1.0
            self.timecfu = 0.0

            # shift soln
            for ilen in range(pdat.cflen-1):
               self.cfsolnu[:,ilen] = self.cfsolnu[:,ilen+1]

            # add one more value of v to cfsoln
            self.cfsolnu[:,pdat.cflen-1] = u[0:pdat.cfdim]
      else:
         if (self.timecfu >= pdat.cflag - pdat.dt05)and(self.cfu_index < pdat.cflen):
            self.record_cfsoln(u, pdat)
            if (self.cfu_index == pdat.cflen):
               self.cfu_soln_filled = True

      
               
   # -----------------------------------
   def print_cft(self, filenamecfu, pdat):
      "Write CF Time into File"

      print("-- Stat Counter CF =", self.stat_counter_cfu)

      with open(filenamecfu, 'w') as handle:
         for ilen in range(pdat.cflen):
            handle.write(str(ilen*pdat.cflag))
            handle.write(" ")         
      
         handle.write("\n")


         for idim in range(pdat.cfdim):
            for ilen in range(pdat.cflen):
               tmp = self.cfu[idim,ilen]/self.stat_counter_cfu - (self.meanu[idim]/self.stat_counter)**2
               handle.write(str(tmp))
               handle.write(" ")
            handle.write("\n")


   # -----------------------------------
   # CF Time Slow
   # -----------------------------------
   def record_cfsolnslow(self, xu, pdat):
      "Record soln into cf_soln"
      
      self.cfsolnxu[:,self.cfxu_index] = xu[0:pdat.cfdimslow]
      self.cfxu_index += 1
      self.timecfxu = 0.0

      
   # -----------------------------------
   def update_timecfxu(self, dt):
      "Update timecf"
      self.timecfxu += dt


   # -----------------------------------
   def compute_cftslow(self, xu, pdat):
      "Compute CF Time"

      if self.cfxu_soln_filled:
         if (self.timecfxu >= pdat.cflag - pdat.dt05):
            #print("Computing CF, time = ", pdat.time)

            # compute CF
            for ilen in range(pdat.cflen):
               self.cfxu[:,ilen]  = self.cfxu[:,ilen]  + self.cfsolnxu[:,0] * self.cfsolnxu[:,ilen]
            self.stat_counter_cfxu += 1.0
            self.timecfxu = 0.0

            # shift soln
            for ilen in range(pdat.cflen-1):
               self.cfsolnxu[:,ilen] = self.cfsolnxu[:,ilen+1]

            # add one more value of v to cfsoln
            self.cfsolnxu[:,pdat.cflen-1] = xu[0:pdat.cfdimslow]
      else:
         if (self.timecfxu >= pdat.cflag - pdat.dt05)and(self.cfxu_index < pdat.cflen):
            self.record_cfsolnslow(xu, pdat)
            if (self.cfxu_index == pdat.cflen):
               self.cfxu_soln_filled = True

      
               
   # -----------------------------------
   def print_cftslow(self, filenamecfu, pdat):
      "Write CF Time into File"

      print("-- Stat Counter CF =", self.stat_counter_cfxu)

      with open(filenamecfu, 'w') as handle:
         for ilen in range(pdat.cflen):
            handle.write(str(ilen*pdat.cflag))
            handle.write(" ")         
      
         handle.write("\n")


         for idim in range(pdat.cfdimslow):
            for ilen in range(pdat.cflen):
               tmp = self.cfxu[idim,ilen]/self.stat_counter_cfxu - (self.meanxu[idim]/self.stat_counter)**2
               handle.write(str(tmp))
               handle.write(" ")
            handle.write("\n")
     

   # -----------------------------------
   # CF Time Local Average
   # -----------------------------------
   def record_cfsolnaverage(self, U, pdat):
      "Record soln into cf_soln"
      
      self.cfsolnU[:,self.cfU_index] = U[0:pdat.cfdimslow]
      self.cfU_index += 1
      self.timecfU = 0.0


   # -----------------------------------
   def update_timecfU(self, dt):
      "Update timecf"
      self.timecfU += dt

   # -----------------------------------
   def compute_cftaverage(self, U, pdat):
      "Compute CF Time"

      if self.cfU_soln_filled:
         if (self.timecfU >= pdat.cflag - pdat.dt05):
            #print("Computing CF, time = ", pdat.time)

            # compute CF
            for ilen in range(pdat.cflen):
               self.cfU[:,ilen]  = self.cfU[:,ilen]  + self.cfsolnU[:,0] * self.cfsolnU[:,ilen]
            self.stat_counter_cfU += 1.0
            self.timecfU = 0.0

            # shift soln
            for ilen in range(pdat.cflen-1):
               self.cfsolnU[:,ilen] = self.cfsolnU[:,ilen+1]

            # add one more value of v to cfsoln
            self.cfsolnU[:,pdat.cflen-1] = U[0:pdat.cfdimslow]
      else:
         if (self.timecfU >= pdat.cflag - pdat.dt05)and(self.cfU_index < pdat.cflen):
            self.record_cfsolnaverage(U, pdat)
            if (self.cfU_index == pdat.cflen):
               self.cfU_soln_filled = True

      
               
   # -----------------------------------
   def print_cftaverage(self, filenamecfu, pdat):
      "Write CF Time into File"

      print("-- Stat Counter CF =", self.stat_counter_cfU)

      with open(filenamecfu, 'w') as handle:
         for ilen in range(pdat.cflen):
            handle.write(str(ilen*pdat.cflag))
            handle.write(" ")         
      
         handle.write("\n")


         for idim in range(pdat.cfdimslow):
            for ilen in range(pdat.cflen):
               tmp = self.cfU[idim,ilen]/self.stat_counter_cfU - (self.meanU[idim]/self.stat_counter)**2
               handle.write(str(tmp))
               handle.write(" ")
            handle.write("\n")
     
   # -----------------------------------
   # Marginal PDF
   # -----------------------------------
   def compute_single_pdf(self, v_tmp, ipdf, pdat):
      "Compute Single PDF given by the index ipdf"

      # adjust the mean
      v  = v_tmp - pdat.pdfmean

      if v >= 0.0:
         indx = int(v/pdat.pdfstep)
         if indx < pdat.pdflen:
            self.pdfpos[ipdf, indx] += 1.0
      else:
         indx = int(-v/pdat.pdfstep)
         if indx < pdat.pdflen:
            self.pdfneg[ipdf, indx] += 1.0
      
   # -----------------------------------
   def compute_pdf(self, v, pdat):
      "Compute PDF for v[0:dimp]"

      for k in range(pdat.pdfdim):
         self.compute_single_pdf(v[k], k, pdat)
         
   # -----------------------------------
   def print_pdf(self, filename, pdat):
      "Output PDF for v[0:dimp]"

      with open(filename, 'w') as handle:
         for k in range(pdat.pdflen):
            tmp = (0.5 + k - pdat.pdflen)*pdat.pdfstep + pdat.pdfmean
            handle.write(str(tmp))
            handle.write(" ")
         for k in range(pdat.pdflen):
            tmp = (0.5 + k)*pdat.pdfstep + pdat.pdfmean
            handle.write(str(tmp))
            handle.write(" ")
         handle.write("\n")

         for jj in range(pdat.pdfdim):
            for k in range(pdat.pdflen):
               tmp = self.pdfneg[jj, pdat.pdflen - k - 1]/self.stat_counter/pdat.pdfstep
               handle.write(str(tmp))
               handle.write(" ")

            for k in range(pdat.pdflen):
               tmp = self.pdfpos[jj, k]/self.stat_counter/pdat.pdfstep
               handle.write(str(tmp))
               handle.write(" ")
            handle.write("\n")




# -----------------------------------
# End class StatData
# -----------------------------------
