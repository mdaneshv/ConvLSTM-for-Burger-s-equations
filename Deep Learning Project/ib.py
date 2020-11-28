
#Forced Burgers Equations
#using Flux-difference + (Euler or RK4)
#fb.py

import numpy as np
import math
import time as walltime
import ast
import sys

from ibpdata import *
from ibsolndata import *
from ibstatdata import *

#from IPython import get_ipython
#get_ipython().magic('reset -sf') 


# -----------------------------------
def PrintHelp():
   "Print Help"

   print("-------------------------------")
   print("==> Forced Burgers equation Simulation")
   print("Long Equilibrium Simulation \n")
   print("Version August 2018")
   print("-------------------------------")

   print("Parameters in the file fb_par.txt \n")

   print("L          = Domain\n")

   
   print("dim        = dimension of the problem")
   print("Tend       = final time of simulation")
   print("dt         = time-step of integration")
   print("seed       = seed for the random number generator")   
   print("Tskip      = skip to schieve stationarity")
   print("dtsubs     = time-step for computing stat (int multiple of dt) \n")


   print("dtmes      = How often to print message about progress")
   print("MesFile    = True - print into file mes.out (for long simulations)")
   print("             False - print to terminal \n")

   print("TstartOut  = Time to Start Soln Output")
   print("TstopOut   = Time to Stop Soln Output")
   print("dtout      = time-step to Soln Output (int multiple of dt) \n") 
   
   print("cflen      = length of array for correlation (<= 100)")
   print("cflag      = CF lag (int multiple of sub-sampling step dtsubs)")
   print("cfdim      = compute CF for u[0:cfdim]")

   print("pdflen     = length for Marginal PDF")
   print("pdfstep    = size of the bin for copmuting PDF using bin-counting")
   print("pdfmean    = pdf will be centered to this value")
   print("pdfdim     = compute PDF for u[0:pdfdim]\n")


   print("-------------------------------")
   print("-- Params for Slow Variables")
   print("averwindow = Aver Window for Fast")
   print("dimslow    = Dimension for the slow variables/averages")

   print("-------------------------------")
   print("Output Files \n")

   print("fbsolnu.dat   = Solution of u")
   print("fbsolnU2.dat  = Local averages")
   print("fbsolnxu.dat  = Slow variables \n")
   print("statu.dat     = Mean, Variance, 3rd and 4th Moment of u[k]")
   print("statU2.dat    = Mean, Variance, 3rd and 4th Moment of U[k]")
   print("statxu.dat    = Mean, Variance, 3rd and 4th Moment of xu[k] \n")
   print("cftu.dat      = CF in time <u[k](t) u[k](t+s)> k=0:cfdim")
   print("cftU2.dat     = CF in time <U[k](t) U[k](t+s)> k=0:cfdim")
   print("cftxu.dat     = CF in time <xu[k](t) xu[k](t+s)> k=0:cfdim \n")
   print("specu.dat     = Energy spectra for u")
   print("specU2.dat    = Energy spectra for U")
   print("specxu.dat    = Energy spectra for xu \n")
   print("pdfu.dat      = PDF of u[k]=0:pdfdim \n")
   print("solnfile1u.dat   = Time series for u[k] k=0:n")
   print("solnfile1U2.dat  = Time series for U[k] k=0:n")
   print("solnfile1xu.dat  = Time series for xu[k] k=0:n")
   print("solnfile1yu.dat  = Time series for yu[k] k=0:n")
   print("-------------------------------")



# -----------------------------------
def printfile_soln(u,pdat):
   "Output Soln into a file"
   
   if pdat.output_condition():
      
      solnfileu.write(str(pdat.time))
      
      for k in u:
         solnfileu.write(" ")         
         solnfileu.write(str(k))
      solnfileu.write("\n")
      


# -----------------------------------
def printfile_soln_average(U,pdat):
   "Output Soln into a file"
   
   if pdat.output_condition():
      
      solnfileU.write(str(pdat.time))
      
      for k in U:
         solnfileU.write(" ")         
         solnfileU.write(str(k))
      solnfileU.write("\n")
      

# -----------------------------------
def printfile_soln_slow(xu,pdat):
   "Output Soln into a file"
   
   if pdat.output_condition():
      
      solnfilexu.write(str(pdat.time))
      
      for k in xu:
         solnfilexu.write(" ")         
         solnfilexu.write(str(k))
      solnfilexu.write("\n")
      

# -----------------------------------
def printfile_force(f, pdat):
   "Output Soln into a file"
   
   if pdat.output_condition():
       
      solnfileforce.write(str(pdat.time))
      
      for k in f:
         solnfileforce.write(" ")         
         solnfileforce.write(str(k))
      solnfileforce.write("\n")
      
# -----------------------------------
def printfile_soln_one_dim(u,U,xu,yu,n,pdat):
   "Output Soln at one specific x into a file"
   
      #print("writing solution, time=", pdat.time)
   
   if 1==1:
       solnfile1u.write(str(pdat.time))
       solnfile1U.write(str(pdat.time))
       solnfile1xu.write(str(pdat.time))
       solnfile1yu.write(str(pdat.time))
      
       for i in range(n):
          solnfile1u.write(" ")         
          solnfile1u.write(str(u[i]))
       solnfile1u.write("\n")
       
       for i in range(n):
          solnfile1U.write(" ")         
          solnfile1U.write(str(U[i]))
       solnfile1U.write("\n")
      
       for i in range(n):
          solnfile1xu.write(" ")         
          solnfile1xu.write(str(xu[i]))
       solnfile1xu.write("\n")

       for i in range(n):
          solnfile1yu.write(" ")         
          solnfile1yu.write(str(yu[i]))
       solnfile1yu.write("\n")

#-------------
def PrintSlowFast(filename1, filename2,pdat):
    
    with open(filename1, 'w') as handle:
       for i in range(pdat.dimslow):
          for j in range(int(pdat.Tend/pdat.dtsubs)):
             handle.write(str(solndat.Uout[i][j]))
             handle.write(" ")
          handle.write("\n")

    with open(filename2, 'w') as handle:
       for i in range(int(2*pdat.dimslow)):
          for j in range(int(pdat.Tend/pdat.dtsubs)):
             handle.write(str(solndat.yuout[i][j]))
             handle.write(" ")
          handle.write("\n")

#-------------
def PrintSlow(filename1, pdat):
    
    with open(filename1, 'w') as handle:
       for i in range(pdat.nstart, pdat.nstop):
          for j in range(int(pdat.Tend/pdat.dtsubs)):
             handle.write(str(solndat.Uout[i][j]))
             handle.write(" ")
          handle.write("\n")


#----------
def compute_rhs(u):
 
    ujm1=u[pdat.dim-1]
    ujp1=u[0]
  
    for j in range(1,pdat.dim-1):
       solndat.netfluxu[j] = -(u[j+1]**2+u[j+1]*u[j]-u[j]*u[j-1]-u[j-1]**2)/pdat.dx6
                            
    j=0
    solndat.netfluxu[j] = -(u[j+1]**2+u[j+1]*u[j]-u[j]*ujm1-ujm1**2)/pdat.dx6
                         
    j=pdat.dim-1
    solndat.netfluxu[j] = -(ujp1**2+ujp1*u[j]-u[j]*u[j-1]-u[j-1]**2)/pdat.dx6


#----------
def compute_rhs_slow(xu):
    
    xujm1=xu[pdat.dimslow-1]
    xujp1=xu[0]
  
    for j in range(1,pdat.dimslow-1):
       solndat.netfluxxu[j] = (xu[j+1]**2+xu[j+1]*xu[j]-xu[j]*xu[j-1]-xu[j-1]**2)/pdat.ndx6-pdat.vndx2*(xu[j+1]-2*xu[j]+xu[j-1])
                            
    j=0
    solndat.netfluxxu[j] = (xu[j+1]**2+xu[j+1]*xu[j]-xu[j]*xujm1-xujm1**2)/pdat.ndx6-pdat.vndx2*(xu[j+1]-2*xu[j]+xujm1)
                         
    j=pdat.dimslow-1
    solndat.netfluxxu[j] = (xujp1**2+xujp1*xu[j]-xu[j]*xu[j-1]-xu[j-1]**2)/pdat.ndx6-pdat.vndx2*(xujp1-2*xu[j]+xu[j-1])
    
    solndat.netfluxxu=-solndat.netfluxxu+solndat.xforce
    

#----------
def compute_rhs_average(U):
    
    Ujm1=U[pdat.dimslow-1]
    Ujp1=U[0]
  
    for j in range(1,pdat.dimslow-1):
       solndat.netfluxU[j] = (U[j+1]**2+U[j+1]*U[j]-U[j]*U[j-1]-U[j-1]**2)/pdat.ndx6-pdat.vndx22*(U[j+1]-2*U[j]+U[j-1])
                            
    j=0
    solndat.netfluxU[j] = (U[j+1]**2+U[j+1]*U[j]-U[j]*Ujm1-Ujm1**2)/pdat.ndx6-pdat.vndx22*(U[j+1]-2*U[j]+Ujm1)
                         
    j=pdat.dimslow-1
    solndat.netfluxU[j] = (Ujp1**2+Ujp1*U[j]-U[j]*U[j-1]-U[j-1]**2)/pdat.ndx6-pdat.vndx22*(Ujp1-2*U[j]+U[j-1])
    
    solndat.netfluxU=-solndat.netfluxU+solndat.xforce


def force_term():
    
    global ran_counter
    
    for i in range(0,3):
        solndat.alpha[i]=ran[ran_counter + 2*i]
        solndat.phi[i]=ran[ran_counter + 2*i + 1]
    
    ran_counter +=  6
    
    for j in range(0,pdat.dimslow):
       solndat.xforce[j]=pdat.Adt*(solndat.alpha[0]*np.cos(pdat.tpi*(j*pdat.dxl + solndat.phi[0]))+\
                   solndat.alpha[1]/np.sqrt(2)*np.cos(pdat.tpi*(2*j*pdat.dxl + solndat.phi[1]))+\
                   solndat.alpha[2]/np.sqrt(3)*np.cos(pdat.tpi*(3*j*pdat.dxl + solndat.phi[2])))
       for i in range(pdat.averwindow):
          indx = i + j*pdat.averwindow
          solndat.force[indx] = solndat.xforce[j]

#----------
def make_one_step():

   compute_rhs(solndat.u)
   
   solndat.paramu = solndat.netfluxu

   solndat.u = solndat.u+pdat.dt*solndat.netfluxu
       
   solndat.fftu = np.fft.rfft(solndat.u)/pdat.dim


#----------
def make_one_step_average():

   compute_rhs_average(solndat.xu)

   solndat.xu = solndat.xu+pdat.dt*solndat.netfluxxu
       
   solndat.fftxu = np.fft.rfft(solndat.xu)/pdat.dimslow

#----------
def make_one_step_RK3():

   compute_rhs(solndat.u)
   
   k1=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u+0.5*k1)
   
   k2=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u+0.75*k2)
   
   k3=pdat.dt*solndat.netfluxu
    
   solndat.u = solndat.u+(2.0*k1+3.0*k2+4.0*k3)/9.0
       
   solndat.fftu = np.fft.rfft(solndat.u)/pdat.dim


#----------
def make_one_step_RK3_1():

   compute_rhs(solndat.u)
   
   k1=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u+0.5*k1)
   
   k2=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u-k1+2.0*k2)
   
   k3=pdat.dt*solndat.netfluxu
    
   solndat.u = solndat.u+(k1+4.0*k2+k3)/6.0
       
   solndat.fftu = np.fft.rfft(solndat.u)/pdat.dim


#----------
def make_one_step_RK3_ec():

   compute_rhs(solndat.u)
   
   k1=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u+k1/3)
   
   k2=pdat.dt*solndat.netfluxu
   
   compute_rhs(solndat.u + a1*k1 + a2*k2)
   
   k3=pdat.dt*solndat.netfluxu
    
   solndat.u = solndat.u+(1.0*k1+5.0*k2+4.0*k3)/10.0
       
   solndat.fftu = np.fft.rfft(solndat.u)/pdat.dim

   

#----------
def make_one_step_RK3_average():

   compute_rhs_average(solndat.xu)
   
   k1=pdat.dt*solndat.netfluxxu
   
   compute_rhs_average(solndat.xu+0.5*k1)
   
   k2=pdat.dt*solndat.netfluxxu
   
   compute_rhs_average(solndat.xu+0.75*k2)
   
   k3=pdat.dt*solndat.netfluxxu
    
   solndat.xu = solndat.xu+(2.0*k1+3.0*k2+4.0*k3)/9.0
       
   solndat.fftxu = np.fft.rfft(solndat.xu)/pdat.dimslow


# -----------------------------------
def ComputeCellAverages(u, pdat):
   "Compute Cell Avergaes = Slow and Fast Variables"

   for k in range(pdat.dimslow):
      averu = 0.0
      for j in range(pdat.averwindow):
         indx = j + k*pdat.averwindow
         averu += u[indx]
      solndat.U[k] = averu/pdat.averwindow

      #for j in range(pdat.averwindow):
      #   indx = j + k*pdat.averwindow
      #   solndat.yu[indx] = u[indx] - solndat.U[k]
    
   #solndat.fftU = np.fft.rfft(solndat.U)/pdat.dimslow

def ComputeAverages(u, pdat):
   "Compute Cell Avergaes = Slow and Fast Variables"

   X = np.zeros(pdat.dimslow, dtype = np.float64)
   y = np.zeros(pdat.dim, dtype = np.float64)
   
   for k in range(pdat.dimslow):
      averu = 0.0
      for j in range(pdat.averwindow):
         indx = j + k*pdat.averwindow
         averu += u[indx]
      X[k] = averu/pdat.averwindow

      for j in range(pdat.averwindow):
         indx = j + k*pdat.averwindow
         y[indx] = u[indx] - X[k]
    
   return X,y
    


# -----------------------------------
# MAIN PROGRAM
# -----------------------------------

a1 = -5.0/48
a2 = 15.0/16



if len(sys.argv) >= 2: 
   if (sys.argv[1] == '-h')or(sys.argv[1]=='-help'):
      PrintHelp()
      sys.exit()

pdat = PData()

pdat.update_from_file('ib_par.txt')
pdat.init_params()

np.random.seed(pdat.seed)


# init arrays for solution and integration
solndat = SolnData(pdat)
statdat = StatData(pdat)


solnfileu = open('fbsolnu.dat', 'w')
solnfileU = open('fbsolnU2.dat', 'w')
solnfilexu = open('fbsolnxhu.dat', 'w')

#Time series for dim = 1: pdat.n
solnfile1u = open('solnfile1u.dat', 'w')
solnfile1U = open('solnfile1U2.dat', 'w')
solnfile1xu = open('solnfile1xu.dat', 'w')
solnfile1yu = open('solnfile1yu.dat', 'w')

cl1 = walltime.clock()
pdat.clockstart = cl1
print("-------------------")
print("Start Computations")
print("Flux-difference scheme + RK4 time-stepping")
print("-------------------")

t=0
# Skip up to time Tskip
for j in range(pdat.nums_skip):
   for k in range(pdat.nums_subs):
      #force_term()
      make_one_step_RK3_ec()
      pdat.advance_time() 
   pdat.PrintMesProgress(solndat.u, solndat.U, solndat.yu, pdat.dtsubs)
   

   #ComputeCellAverages(solndat.u,pdat)
   #for i in range(pdat.n):
   #   solndat.Uout[i][t] = solndat.U[i]
   #for i in range(int(pdat.n*pdat.averwindow)):
   #   solndat.yuout[i][t] = solndat.yu[i]
   t=t+1
   #printfile_soln_one_dim(solndat.u,solndat.U,solndat.xu,solndat.yu,pdat.n, pdat)
   printfile_soln(solndat.u, pdat)
   printfile_soln_average(solndat.U, pdat)
   printfile_soln_slow(solndat.xu,pdat)

   pdat.reset_timeout

print("Time After Skip = ", pdat.time)

compute_stat = False

if compute_stat:

    statdat.compute_specall(solndat.fftu, solndat.fftU, solndat.fftxu)
    
    statdat.compute_all_onep_stat(solndat.u, pdat)
    statdat.compute_all_onep_statslow(solndat.xu,pdat)
    statdat.compute_all_onep_stataverage(solndat.U, pdat)
        
    statdat.record_cfsoln(solndat.yu, pdat)
    statdat.record_cfsolnslow(solndat.xu, pdat)
    statdat.record_cfsolnaverage(solndat.U, pdat)

# ---------------
# Main Loop
# After Tskip
# ---------------
for j in range(pdat.nums):
   for k in range(pdat.nums_subs):
      #force_term()
      make_one_step_RK3_ec()
      pdat.advance_time()      
      
   #ComputeCellAverages(solndat.u,pdat)
   for i in range(pdat.nstart, pdat.nstop):
      solndat.Uout[i][t] = solndat.u[i]
   #for i in range(pdat.dimslow):
   #   solndat.yuout[2*i][t] = solndat.yu[int(i*pdat.averwindow)]
   #   solndat.yuout[2*i+1][t] = solndat.yu[int((i+1)*pdat.averwindow - 1)]
   t=t+1
   #printfile_soln_one_dim(solndat.u,solndat.U,solndat.xu,solndat.yu,pdat.n, pdat)
   printfile_soln(solndat.u, pdat)
   printfile_soln_average(solndat.U, pdat)
   printfile_soln_slow(solndat.xu, pdat)
   pdat.reset_timeout 
      
   pdat.PrintMesProgress(solndat.u, solndat.U, solndat.yu, pdat.dtsubs)
   
   if compute_stat:

      # compute spectra 
      statdat.compute_specall(solndat.fftu, solndat.fftU, solndat.fftxu)
      
      # compute one parameter statistics
      statdat.compute_all_onep_stat(solndat.yu, pdat)
      statdat.compute_all_onep_statslow(solndat.xu, pdat)
      statdat.compute_all_onep_stataverage(solndat.U,pdat)
      
      # compute CF
      statdat.compute_cft(solndat.yu, pdat)
      statdat.compute_cftslow(solndat.xu,pdat)
      statdat.compute_cftaverage(solndat.U, pdat)

      # Update timecf
      statdat.update_timecfu(pdat.dt * pdat.nums_subs)
      statdat.update_timecfxu(pdat.dt * pdat.nums_subs)
      statdat.update_timecfU(pdat.dt * pdat.nums_subs)
      
      # compute marginal PDF
      statdat.compute_pdf(solndat.u,pdat)
      
   

print("-------------------")
print("End Computations")
print("-------------------")


# ---------------
# End of Main Loop
# ---------------

solnfileu.close()
solnfileU.close()
solnfilexu.close()
solnfile1u.close()
solnfile1U.close()
solnfile1xu.close()
solnfile1yu.close()

pdat.enerfile.close()

if compute_stat:
   
   statdat.output_spec("specu.dat", "meanfftu.dat")
   statdat.output_specslow("specxu.dat", "meanfftxu.dat")
   statdat.output_specaverage("specU2.dat", "meanfftU2.dat")
   
   print("-------------------")
   print("Output Stat for Full Dynamics")
   print("-------------------")
   statdat.output_all_stat(pdat)
   statdat.output_all_statslow(pdat)
   statdat.output_all_stataverage(pdat)
   statdat.print_pdf('pdfu.dat',pdat)

cl2 = walltime.clock()
print("-------------------")
print("End of Computations, walltime = ", (cl2-cl1), " sec")
days = int((cl2-cl1)/60/60/24)
hrs  = int((cl2-cl1)/60/60 - days*24)
mnts = int((cl2-cl1)/60 - hrs*60.0 - days*60.0*24.0)
print("    days =", days, "hrs =", hrs, "min =", mnts)
print("-------------------")

#PrintSlowFast('Useries.dat','yuseries.dat',pdat)
PrintSlow('Useries.dat',pdat)

#Print last u
with open('lastu.dat', 'w') as handle:
   for i in range(pdat.dim):
      handle.write(str(solndat.u[i]))
      handle.write(" ")
   handle.write("\n")

   
