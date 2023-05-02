# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 21:24:39 2022

@author: nikol
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fsolve
import os
import warnings

warnings.filterwarnings('ignore', 'The iteration is not making good progress')
def sustav(initial_design, plot = False, plot_title = '4_stapni_mehanizam', animate = False):      
        

    res_dir = 'results'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    """
    Jednadžbe kretanja mehanizma
    """
    # fi = fi3, fi4
    
    # a, b - pozicija ploče za tracer point
    # r1, r2, r3, r4 - štapovi mehanizma
    r1, r2, r3, r4, a, b  = initial_design
    fi1 = 0

    mehanizam = lambda theta, fi: np.array([r2*np.cos(theta) +  r3*np.cos(fi[0]) -  r4*np.cos(fi[1]) -  r1*np.cos(fi1)
                                          , r2*np.sin(theta) +  r3*np.sin(fi[0]) -  r4*np.sin(fi[1]) -  r1*np.sin(fi1)])

    D = np.zeros([len(theta), 4]) # fi1, fi2/theta, fi3, fi4
    d = np.array([0, 360*(np.pi/180)])
    for i in range(len(theta)):
        d = fsolve(lambda fi: mehanizam(theta[i],fi), d)
        D[i, 0] = fi1
        D[i, 1] = theta[i]
        D[i, 2] = d[0]
        D[i, 3] = d[1]         
    
    # Raspakiravanje vrijednosti za pojedine kuteve štapnog mehanizma
    fi1, fi2, fi3, fi4 = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    
    """
    Računanje kordinata štapova mehanizma
    """
    # Postava ploče
    ra = a * r3 # % duljine štapa r3
    rb = b * r3 # visina ploce prema duljini štapa r3
    rc = np.sqrt((ra**2 + rb**2)) # udaljenost izmedu tocke A i P
    alfa = np.arccos(ra/rc) # kut izmedu vektora AP i AB
    

    Matrica1 = np.ones(np.size(fi2))
    Matrica0 = np.zeros(np.size(fi2))
    
    # točka B_0 x = r1, y = 0
    RB0 = r1*np.vstack([Matrica1,Matrica0])

    # točka A_0 x = 0 y = 0
    RA0 = np.vstack([Matrica0,Matrica0])
    
    # točka A x = r2 * cos(fi2) y = r2 * sin(fi2)
    RA = RA0 + r2*np.vstack([np.cos(fi2),np.sin(fi2)]) 
    
    # točka B x = RAx + r3 * cos(fi3) y = RAy + r3 * sin(fi3)
    RB = RA + r3*np.vstack([np.cos(fi3),np.sin(fi3)])

    # točka P 
    RP = RA + ra*np.vstack([np.cos(fi3),np.sin(fi3)]) + rb*np.vstack([-np.sin(fi3),np.cos(fi3)])
    
    xkomp = np.zeros([len(theta), len(xPi)])
    ykomp = np.zeros([len(theta), len(yPi)])   
    
    # Spajanje svih vrijednosti u 2 varijable; xkomp, ykomp
    for i in range(len(theta)):
        xkomp[i, 0] = RA0[0, i]
        xkomp[i, 1] = RA[0, i]
        xkomp[i, 2] = RP[0, i]
        xkomp[i, 3] = RB[0, i]
        xkomp[i, 4] = RB0[0, i]
        ykomp[i, 0] = RA0[1, i]
        ykomp[i, 1] = RA[1, i]
        ykomp[i, 2] = RP[1, i]
        ykomp[i, 3] = RB[1, i]
        ykomp[i, 4] = RB0[1, i]
    
    # Vrh mehanizma - tracer point
    xP = xkomp[:, 2]
    yP = ykomp[:, 2]
    
    """
    Constraints
    """
    design = r1, r2, r3, r4
    hmax = max(design)
    hmin = min(design)
    hrest = np.sum(design) - hmax - hmin

    # Grashofov uvjet
    if  hmax + hmin <= hrest:
        constraint_1 = 0
    else:
        constraint_1 = ((hmax + hmin) - hrest)
    
    # Vector loop uvjet
    if abs((r2 + r3) - (r4 + r1)) == 0:  
        constraint_2 = 0
    else:
        constraint_2 = abs((r2 + r3) - (r4 + r1))

    """
    Penalizacija
    """
    penali = np.zeros([len(theta), len(xPi)])
    penali_opt  = np.zeros([len(xPi)])
    xPi_M, yPi_M = np.ones([len(xPi),len(xP)]), np.ones([len(xPi),len(xP)])
    
    # Broj penala = broj točaka krivulje 
    theta_opt, theta_check = [], []
    for i in range(len(xPi)):
        for j in range(len(theta)):
            xPi_M[i,j] = xPi[i] 
            yPi_M[i,j] = yPi[i] 
            penali[j, i] = (xPi_M[i,j]-xP[j])**2 + (yPi_M[i,j]-yP[j])**2 # Vrijednost penala
    
        # Traženje minimalnih kuteva od len(theta)
        loc = np.where(penali[:, i] == np.min(penali[:, i]))[0].tolist()
        theta_check.append(loc)
        theta_opt.append(theta[loc])
        penali_opt[i] = np.min(penali[:,i])
    # Penalizacija za N optimalnih kuteva
    Lp = np.sum(penali_opt)
    
    """
    Plotanje i animacija
    """
    # Raspakiravanje varijabli ==> za plotanje
    xA0, xA, xP, xB, xB0 = xkomp[:, 0], xkomp[:, 1], xkomp[:, 2], xkomp[:, 3], xkomp[:, 4]
    yA0, yA, yP, yB, yB0 = ykomp[:, 0], ykomp[:, 1], ykomp[:, 2], ykomp[:, 3], ykomp[:, 4]
    
    # Pozicije mehanizama za najbliže kuteve točkama
    XA0, XA, XP, XB, XB0 = [], [], [], [], []
    YA0, YA, YP, YB, YB0 = [], [], [], [], []
    for i in range(len(xPi)):
        XA0.append(xA0[theta_check[i]])
        XA.append(xA[theta_check[i]])
        XP.append(xP[theta_check[i]])
        XB.append(xB[theta_check[i]])
        XB0.append(xB0[theta_check[i]])
        YA0.append(yA0[theta_check[i]])
        YA.append(yA[theta_check[i]])
        YP.append(yP[theta_check[i]])
        YB.append(yB[theta_check[i]])
        YB0.append(yB0[theta_check[i]])
        
    if plot:
        fig = plt.figure(figsize = (10, 8), constrained_layout = True)
        # Skaliranje grafa
        maximumi_x = max(np.max(xkomp), np.max(xPi))
        maximumi_y = max(np.max(ykomp), np.max(yPi))
        minimumi_x = min(np.min(xkomp), np.min(xPi))
        minimumi_y = min(np.min(ykomp), np.min(yPi))
        min_, max_ = min(minimumi_x, minimumi_y), max(maximumi_x, maximumi_y)
        scale = 1.15
        ax = fig.add_subplot(111, autoscale_on=False, xlim=[min_*scale, max_*scale], ylim=[min_*scale, max_*scale])
        
        # Tracer point
        ax.plot(xP,yP, c = 'blue', lw = 1)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        
        # Najbliže točke
        plt.plot(XP, YP, 'ro', label = 'Nearest points')
        
        # Krivulja - tracer point
        plt.plot(xPi,yPi,lw = 1, marker='o' , label = 'Target points')
        # Krivulja - štapa A
        plt.plot(xA, yA, lw = 0.5, ls = '--', c = 'red', label = 'Linkage A')
        # Krivulja - štapa B
        plt.plot(xB, yB, lw = 0.5, ls = '--', c = 'orange', label = 'Linkage B')
        

        kut_template = 'Angle = %.f °'
        kut_text = ax.text(0.08, 0.95, '', transform=ax.transAxes)
        bar = f'r1 = {r1:.5f} m, r2 = {r2:.5f} m, r3 = {r3:.5f} m, r4 = {r4:.5f} m'
        ax.text(0.1, 0.02, bar, transform=ax.transAxes)

        ax.text(0.08, 0.90, f'Fitness = {Lp:.5f}', transform=ax.transAxes)
        # Grashofov uvijet tekst
        ax.text(0.08, 0.85, 'Grashof condition = ' , transform=ax.transAxes)
        if constraint_1 <= 0.001:
            ax.text(0.24, 0.85, 'Satisfied', color = 'green', transform=ax.transAxes)
        else:
            ax.text(0.24, 0.85, f'Not satisfied; {constraint_1:.3f}', color = 'red', transform=ax.transAxes)
            
        # Vector loop condition
        ax.text(0.08, 0.80, 'Vector loop condition = ' , transform=ax.transAxes)
        if constraint_2 <= 0.001:
            ax.text(0.26, 0.80, 'Satisfied', color = 'green', transform=ax.transAxes)
        else:
            ax.text(0.26, 0.80, f'Not satisfied; {constraint_2:.3f}', color = 'red', transform=ax.transAxes)
            
        # Sadržaj za animaciju
        line1, = ax.plot([], [], marker = 'o',c = 'black',lw = 3,ms = 8)
        line2, = ax.plot([], [], marker = 'o',c = 'black',lw = 3,ms = 8)
        line3, = ax.plot([], [], marker = 'p',c = 'red',lw = 3,ms = 14, label = 'Tracer point')
        plt.legend(loc = 'upper right')

        plt.title(plot_title)
        plt.grid()
        plt.savefig(f'{res_dir}\{plot_title}.png')
        if animate:
            
            def init():
                
                line1.set_data([],[])
                line2.set_data([],[])
                line3.set_data([],[])
                return line1,line2,line3,
                
            def animate(i):
                stapovi1x = [xA0[i], xA[i], xB[i], xB0[i]]
                stapovi1y = [yA0[i], yA[i], yB[i], yB0[i]]
                stapovi2x = [xA[i], xP[i], xB[i]]
                stapovi2y = [yA[i], yP[i], yB[i]]
                markerx = [xP[i]]
                markery = [yP[i]]
                
                line1.set_data(stapovi1x, stapovi1y)
                line2.set_data(stapovi2x, stapovi2y)
                line3.set_data(markerx, markery)    
                kut_text.set_text(kut_template%(i*(360/n)))      
                return line1, line2, line3,

    
            global anim
            anim = animation.FuncAnimation(fig, animate, frames = np.arange(0, len(xA)) , interval= 100, blit=False, init_func=init)
            # anim.save(f'{res_dir}\{plot_title}.avi', fps = 15, dpi = 300)
       
        plt.show()

    return Lp, constraint_1, constraint_2
            
def fitness_penalty(initial_design):
    objective, constraint_1, constraint_2 = sustav(initial_design)
    f = objective
    if constraint_1 > 0:
        f += 1 + constraint_1
    if constraint_2 > 0:
        f += 1 + constraint_2
    return f
    

if __name__ =='__main__':

    """
    Raspon kuteva fi2, lower bound i upper bound, izgled krivulje
    """
    theta_min, theta_max, n = 0 * (np.pi/180), 360 * (np.pi/180), 180
    theta = np.linspace(theta_min, theta_max, n)
    
    LB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    UB = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.3])
    
    # Krivulja 1
    xPi = [-0.1,0,0.15,0.3,0.35]
    yPi = [0.2,0.25,0.28,0.2,0.10]
    
    # Krivulja 2
    # xPi = [-0.05,0,0.1,0.2,0.25]
    # yPi = [0.25,0.2,0.15,0.2,0.25]
    
    """
    Test design
    """
    test_design = [0.150, 0.100 , 0.250, 0.200, 0.8, 0.1]   # r1, r2, r3, r4
    p_title = 'four_bar_mechanism'
    sustav(test_design, plot = True, plot_title= p_title, animate = True)
    

    """
    Scipy Nelder-Mead, Indago Nelder-Mead, Scipy L-BFGS-B, Indago MSGS
    """
    import scipy.optimize as spo
    import indago

    n_optim = 3 # Broj optimizacija - neparan broj
    NelderMead_spo = np.zeros([n_optim, 8])
    NelderMead_indago = np.zeros([n_optim, 8])
    L_BFSG_B_spo = np.zeros([n_optim, 8])
    MSGS_indago = np.zeros([n_optim, 8])
    # [0:6] - optimalni dizajn
    # [6] - fitness penalty sa constraint vrijednost
    # [7] - fitness penalty bez constraint vrijednosti
    
    for i in range(n_optim):
        """
        Scipy Nelder-Mead
        """
        bounds = np.array([LB, UB]).T
        initial_design = np.random.uniform(LB, UB)
        r_NM_spo =  spo.minimize(fitness_penalty, 
                          initial_design,
                          method='Nelder-Mead',
                          bounds=zip(LB, UB),
                          options={'maxiter':2000,
                                    'maxfev':2000,
                                    'xatol': 1e-9,
                                    'fatol': 1e-9,
                                    'adaptive': True})
    
        print(f'Total evaluations spo: {r_NM_spo.nfev}')
        print(f'Total iterations spo: {r_NM_spo.nit}')
        NelderMead_spo[i, 0:6] = r_NM_spo.x
        NelderMead_spo[i, 6] = sustav(r_NM_spo.x, plot = False, plot_title= False, animate = False)[0]
        NelderMead_spo[i, 7] = r_NM_spo.fun
        
        """
        Indago Nelder-Mead
        """
        r_NM_indago = indago.NelderMead()
        r_NM_indago.dimensions = len(initial_design)
        r_NM_indago.lb = LB
        r_NM_indago.ub = UB
        r_NM_indago.evaluation_function = fitness_penalty
    
        r_NM_indago.objective_labels = ['Fitness']
        r_NM_indago.max_iterations = 2000
        r_NM_indago.max_evaluations = 2000
        r_NM_indago.target_fitness = 1e-9
        r_NM_indago.monitoring = 'None' # 'none', 'basic', 'dashboard'
        r_NM_i = r_NM_indago.optimize()
        
        NelderMead_indago[i, 0:6] = r_NM_i.X
        NelderMead_indago[i, 6] = r_NM_i.f
        NelderMead_indago[i, 7] = fitness_penalty(r_NM_i.X)
    
        """
        Scipy L-BFSG-B
        """
        r_L_BFSG_B_spo =  spo.minimize(fitness_penalty, 
                          initial_design,
                          method='L-BFGS-B',
                          jac='3-point',
                          bounds = bounds,
                          options={'maxiter':2000,
                                    'maxfun':2000,
                                    'eps':1e-9,
                                    'ftol': 1e-9})
        

        print(f'Total evaluations spo: {r_L_BFSG_B_spo.nfev}')
        print(f'Total iterations spo: {r_L_BFSG_B_spo.nit}')
        
        L_BFSG_B_spo[i, 0:6] = r_L_BFSG_B_spo.x
        L_BFSG_B_spo[i, 6] = sustav(r_L_BFSG_B_spo.x, plot = False, plot_title= False, animate = False)[0]
        L_BFSG_B_spo[i, 7] = r_L_BFSG_B_spo.fun
        
        """
        Indago MSGS
        """
        r_MSGS_indago = indago.MSGS()
        r_MSGS_indago.dimensions = len(initial_design)
        r_MSGS_indago.lb = LB
        r_MSGS_indago.ub = UB
        r_MSGS_indago.constraints = 2
        r_MSGS_indago.evaluation_function = sustav
    
        r_MSGS_indago.objective_labels = ['Fitness']
        r_MSGS_indago.constraint_labels = ['Constraint_1', 'Constraint_2']
        r_MSGS_indago.max_iterations = 2000
        r_MSGS_indago.max_evaluations = 2000
        r_MSGS_indago._stalled_eval = 100
        r_MSGS_indago._stalled_it = 100
        r_MSGS_indago.target_fitness = 1e-9
        r_MSGS_indago.monitoring = 'none' # 'none', 'basic', 'dashboard'
        r_MSGS_i = r_MSGS_indago.optimize()
        
        MSGS_indago[i, 0:6] = r_MSGS_i.X
        MSGS_indago[i, 6] = r_MSGS_i.f
        MSGS_indago[i, 7] = fitness_penalty(r_MSGS_i.X)
        
        print(f'{((i + 1)/n_optim) * 100:.2f} %')

    """
    Soritranje rezultata prema fitness value
    """
    NelderMead_spo = NelderMead_spo[NelderMead_spo[:, 6].argsort()]
    NelderMead_indago = NelderMead_indago[NelderMead_indago[:, 6].argsort()]
    L_BFSG_B_spo = L_BFSG_B_spo[L_BFSG_B_spo[:, 6].argsort()]
    MSGS_indago = MSGS_indago[MSGS_indago[:, 6].argsort()]
    
   
    """
    Spremanje rezultata i videa
    """
    rezultati = [NelderMead_spo, NelderMead_indago, L_BFSG_B_spo, MSGS_indago]
    save_title = ['Nelder_Mead_scipy', 'Nelder_Mead_indago', 'L_BFGS_B_scipy', 'MSGS_indago']
    plot_title_average = ['Nelder_Mead_scipy_average', 'Nelder_Mead_indago_average', 'L_BFGS_B_scipy_average', 'MSGS_indago_average']
    plot_title_best = ['Nelder_Mead_scipy_best', 'Nelder_Mead_indago_best', 'L_BFGS_B_scipy_best', 'MSGS_indago_best']

    for i in range(len(rezultati)):
        print(f'Fitness = {rezultati[i][:,6]}')
        np.savetxt(f'results\{save_title[i]}.txt', rezultati[i], delimiter = ',')
        average_design = rezultati[i][n_optim//2, 0:6]
        best_design = rezultati[i][0, 0:6]
        sustav(average_design, plot = True, animate = True, plot_title = plot_title_average[i])
        plt.close()
        sustav(best_design, plot = True, animate = True, plot_title = plot_title_best[i])
        plt.close()