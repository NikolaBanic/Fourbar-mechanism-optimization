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

def system(initial_design, plot=False, plot_title='4-bar_mechanism', animate=False):

    res_dir = 'results'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    """
    Equations of motion for the mechanism
    """
    # fi = fi3, fi4

    # a, b - position of the tracer point on the plate
    # r1, r2, r3, r4 - lengths of the mechanism links
    r1, r2, r3, r4, a, b  = initial_design
    fi1 = 0

    mechanism = lambda theta, fi: np.array([r2 * np.cos(theta) + r3 * np.cos(fi[0]) - r4 * np.cos(fi[1]) - r1 * np.cos(fi1),
                                           r2 * np.sin(theta) + r3 * np.sin(fi[0]) - r4 * np.sin(fi[1]) - r1 * np.sin(fi1)])

    D = np.zeros([len(theta), 4])  # fi1, fi2/theta, fi3, fi4
    d = np.array([0, 360 * (np.pi / 180)])
    for i in range(len(theta)):
        d = fsolve(lambda fi: mechanism(theta[i], fi), d)
        D[i, 0] = fi1
        D[i, 1] = theta[i]
        D[i, 2] = d[0]
        D[i, 3] = d[1]

    # Unpacking values for individual angles of the four-bar mechanism
    fi1, fi2, fi3, fi4 = D[:, 0], D[:, 1], D[:, 2], D[:, 3]

    """
    Calculating the coordinates of the mechanism links
    """
    # Plate positioning
    ra = a * r3  # % position along link r3
    rb = b * r3  # height of the plate relative to link r3
    rc = np.sqrt((ra**2 + rb**2))  # distance between points A and P
    alfa = np.arccos(ra / rc)  # angle between vectors AP and AB

    Matrix1 = np.ones(np.size(fi2))
    Matrix0 = np.zeros(np.size(fi2))

    # Point B_0 x = r1, y = 0
    RB0 = r1 * np.vstack([Matrix1, Matrix0])

    # Point A_0 x = 0, y = 0
    RA0 = np.vstack([Matrix0, Matrix0])

    # Point A x = r2 * cos(fi2), y = r2 * sin(fi2)
    RA = RA0 + r2 * np.vstack([np.cos(fi2), np.sin(fi2)])

    # Point B x = RAx + r3 * cos(fi3), y = RAy + r3 * sin(fi3)
    RB = RA + r3 * np.vstack([np.cos(fi3), np.sin(fi3)])

    # Point P
    RP = RA + ra * np.vstack([np.cos(fi3), np.sin(fi3)]) + rb * np.vstack([-np.sin(fi3), np.cos(fi3)])

    xkomp = np.zeros([len(theta), len(xPi)])
    ykomp = np.zeros([len(theta), len(yPi)])

    # Combining all values into two variables; xkomp, ykomp
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

    # Top of the mechanism - tracer point
    xP = xkomp[:, 2]
    yP = ykomp[:, 2]

    """
    Constraints
    """
    design = r1, r2, r3, r4
    hmax = max(design)
    hmin = min(design)
    hrest = np.sum(design) - hmax - hmin

    # Grashof's condition
    if hmax + hmin <= hrest:
        constraint_1 = 0
    else:
        constraint_1 = (hmax + hmin) - hrest

    # Vector loop constraint
    if abs((r2 + r3) - (r4 + r1)) == 0:
        constraint_2 = 0
    else:
        constraint_2 = abs((r2 + r3) - (r4 + r1))

    """
    Penalization
    """
    penalties = np.zeros([len(theta), len(xPi)])
    penalties_opt = np.zeros([len(xPi)])
    xPi_M, yPi_M = np.ones([len(xPi), len(xP)]), np.ones([len(xPi), len(xP)])

    # Number of penalties = number of points on the curve
    theta_opt, theta_check = [], []
    for i in range(len(xPi)):
        for j in range(len(theta)):
            xPi_M[i, j] = xPi[i]
            yPi_M[i, j] = yPi[i]
            penalties[j, i] = (xPi_M[i, j] - xP[j])**2 + (yPi_M[i, j] - yP[j])**2  # Penalty value

        # Finding the minimum angles among len(theta)
        loc = np.where(penalties[:, i] == np.min(penalties[:, i]))[0].tolist()
        theta_check.append(loc)
        theta_opt.append(theta[loc])
        penalties_opt[i] = np.min(penalties[:, i])

    # Penalization for N optimal angles
    Lp = np.sum(penalties_opt)
    
    """
    Plotting and Animation
    """
    # Unpacking variables for plotting
    xA0, xA, xP, xB, xB0 = xkomp[:, 0], xkomp[:, 1], xkomp[:, 2], xkomp[:, 3], xkomp[:, 4]
    yA0, yA, yP, yB, yB0 = ykomp[:, 0], ykomp[:, 1], ykomp[:, 2], ykomp[:, 3], ykomp[:, 4]
    
    # Positions of the mechanism for the nearest angles to the points
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
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        # Scaling the plot
        maximumi_x = max(np.max(xkomp), np.max(xPi))
        maximumi_y = max(np.max(ykomp), np.max(yPi))
        minimumi_x = min(np.min(xkomp), np.min(xPi))
        minimumi_y = min(np.min(ykomp), np.min(yPi))
        min_, max_ = min(minimumi_x, minimumi_y), max(maximumi_x, maximumi_y)
        scale = 1.15
        ax = fig.add_subplot(111, autoscale_on=False, xlim=[min_*scale, max_*scale], ylim=[min_*scale, max_*scale])
        
        # Tracer point
        ax.plot(xP, yP, c='blue', lw=1)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        
        # Nearest points
        plt.plot(XP, YP, 'ro', label='Nearest points')
        
        # Curve - tracer point
        plt.plot(xPi, yPi, lw=1, marker='o', label='Target points')
        # Curve - Linkage A
        plt.plot(xA, yA, lw=0.5, ls='--', c='red', label='Linkage A')
        # Curve - Linkage B
        plt.plot(xB, yB, lw=0.5, ls='--', c='orange', label='Linkage B')

        kut_template = 'Angle = %.f °'
        kut_text = ax.text(0.08, 0.95, '', transform=ax.transAxes)
        bar = f'r1 = {r1:.5f} m, r2 = {r2:.5f} m, r3 = {r3:.5f} m, r4 = {r4:.5f} m'
        ax.text(0.1, 0.02, bar, transform=ax.transAxes)

        ax.text(0.08, 0.90, f'Fitness = {Lp:.5f}', transform=ax.transAxes)
        # Grashof condition text
        ax.text(0.08, 0.85, 'Grashof condition = ', transform=ax.transAxes)
        if constraint_1 <= 0.001:
            ax.text(0.24, 0.85, 'Satisfied', color='green', transform=ax.transAxes)
        else:
            ax.text(0.24, 0.85, f'Not satisfied; {constraint_1:.3f}', color='red', transform=ax.transAxes)
            
        # Vector loop condition text
        ax.text(0.08, 0.80, 'Vector loop condition = ', transform=ax.transAxes)
        if constraint_2 <= 0.001:
            ax.text(0.26, 0.80, 'Satisfied', color='green', transform=ax.transAxes)
        else:
            ax.text(0.26, 0.80, f'Not satisfied; {constraint_2:.3f}', color='red', transform=ax.transAxes)
            
        # Content for animation
        line1, = ax.plot([], [], marker='o', c='black', lw=3, ms=8)
        line2, = ax.plot([], [], marker='o', c='black', lw=3, ms=8)
        line3, = ax.plot([], [], marker='p', c='red', lw=3, ms=14, label='Tracer point')
        plt.legend(loc='upper right')

        plt.title(plot_title)
        plt.grid()
        plt.savefig(f'{res_dir}\{plot_title}.png')
        if animate:
            
            def init():
                
                line1.set_data([], [])
                line2.set_data([], [])
                line3.set_data([], [])
                return line1, line2, line3,
                
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
                kut_text.set_text(kut_template % (i * (360 / n)))
                return line1, line2, line3,

    
            global anim
            anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(xA)), interval=100, blit=False, init_func=init)
            anim.save(f'{res_dir}\{plot_title}.avi', fps=15, dpi=300)
       
        plt.show()

    return Lp, constraint_1, constraint_2
            
def fitness_penalty(initial_design):
    objective, constraint_1, constraint_2 = system(initial_design)
    f = objective
    if constraint_1 > 0:
        f += 1 + constraint_1
    if constraint_2 > 0:
        f += 1 + constraint_2
    return f
    

if __name__ == '__main__':
    # Angle range for theta, lower and upper bounds, and curve shape
    theta_min, theta_max, n = 0 * (np.pi/180), 360 * (np.pi/180), 180
    theta = np.linspace(theta_min, theta_max, n)

    LB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    UB = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.3])

    # Curve 1
    xPi = [-0.1, 0, 0.15, 0.3, 0.35]
    yPi = [0.2, 0.25, 0.28, 0.2, 0.10]

    # Test design
    test_design = [0.150, 0.100, 0.250, 0.200, 0.8, 0.1]   # r1, r2, r3, r4
    plot_title = 'four_bar_mechanism'
    system(test_design, plot=True, plot_title=plot_title, animate=True)
    
    """
    Scipy Nelder-Mead, Indago Nelder-Mead, Scipy L-BFGS-B, Indago MSGS
    """
    import scipy.optimize as spo
    import indago

    # Number of optimizations
    n_optim = 3

    NelderMead_spo = np.zeros([n_optim, 8])
    NelderMead_indago = np.zeros([n_optim, 8])
    L_BFSG_B_spo = np.zeros([n_optim, 8])
    MSGS_indago = np.zeros([n_optim, 8])
    # [0:6] - optimal design
    # [6] - fitness penalty with constraint
    # [7] - fitness penalty without constraint
    
    for i in range(n_optim):
        # Scipy Nelder-Mead
        bounds = np.array([LB, UB]).T
        initial_design = np.random.uniform(LB, UB)
        r_NM_spo = spo.minimize(fitness_penalty,
                               initial_design,
                               method='Nelder-Mead',
                               bounds=list(zip(LB, UB)),
                               options={'maxiter': 2000,
                                        'maxfev': 2000,
                                        'xatol': 1e-9,
                                        'fatol': 1e-9,
                                        'adaptive': True})

        print(f'Total evaluations spo: {r_NM_spo.nfev}')
        print(f'Total iterations spo: {r_NM_spo.nit}')
        NelderMead_spo[i, 0:6] = r_NM_spo.x
        NelderMead_spo[i, 6] = system(r_NM_spo.x, plot=False, plot_title=False, animate=False)[0]
        NelderMead_spo[i, 7] = r_NM_spo.fun

        # Indago Nelder-Mead
        r_NM_indago = indago.NelderMead()
        r_NM_indago.dimensions = len(initial_design)
        r_NM_indago.lb = LB
        r_NM_indago.ub = UB
        r_NM_indago.evaluation_function = fitness_penalty
        r_NM_indago.objective_labels = ['Fitness']
        r_NM_indago.max_iterations = 2000
        r_NM_indago.max_evaluations = 2000
        r_NM_indago.target_fitness = 1e-9
        r_NM_indago.monitoring = 'None'

        r_NM_i = r_NM_indago.optimize()
        NelderMead_indago[i, 0:6] = r_NM_i.X
        NelderMead_indago[i, 6] = r_NM_i.f
        NelderMead_indago[i, 7] = fitness_penalty(r_NM_i.X)

        # Scipy L-BFSG-B
        r_L_BFSG_B_spo = spo.minimize(fitness_penalty,
                                     initial_design,
                                     method='L-BFGS-B',
                                     jac='3-point',
                                     bounds=bounds,
                                     options={'maxiter': 2000,
                                              'maxfun': 2000,
                                              'eps': 1e-9,
                                              'ftol': 1e-9})
        

        print(f'Total evaluations spo: {r_L_BFSG_B_spo.nfev}')
        print(f'Total iterations spo: {r_L_BFSG_B_spo.nit}')
        L_BFSG_B_spo[i, 0:6] = r_L_BFSG_B_spo.x
        L_BFSG_B_spo[i, 6] = system(r_L_BFSG_B_spo.x, plot=False, plot_title=False, animate=False)[0]
        L_BFSG_B_spo[i, 7] = r_L_BFSG_B_spo.fun

        # Indago MSGS
        r_MSGS_indago = indago.MSGS()
        r_MSGS_indago.dimensions = len(initial_design)
        r_MSGS_indago.lb = LB
        r_MSGS_indago.ub = UB
        r_MSGS_indago.constraints = 2
        r_MSGS_indago.evaluation_function = system
        r_MSGS_indago.objective_labels = ['Fitness']
        r_MSGS_indago.constraint_labels = ['Constraint_1', 'Constraint_2']
        r_MSGS_indago.max_iterations = 2000
        r_MSGS_indago.max_evaluations = 2000
        r_MSGS_indago._stalled_eval = 100
        r_MSGS_indago._stalled_it = 100
        r_MSGS_indago.target_fitness = 1e-9
        r_MSGS_indago.monitoring = 'none'
        r_MSGS_i = r_MSGS_indago.optimize()
        MSGS_indago[i, 0:6] = r_MSGS_i.X
        MSGS_indago[i, 6] = r_MSGS_i.f
        MSGS_indago[i, 7] = fitness_penalty(r_MSGS_i.X)
        print(f'{((i + 1) / n_optim) * 100:.2f} %')

    # Sort results by fitness value
    NelderMead_spo = NelderMead_spo[NelderMead_spo[:, 6].argsort()]
    NelderMead_indago = NelderMead_indago[NelderMead_indago[:, 6].argsort()]
    L_BFSG_B_spo = L_BFSG_B_spo[L_BFSG_B_spo[:, 6].argsort()]
    MSGS_indago = MSGS_indago[MSGS_indago[:, 6].argsort()]
    
   
    """
    Saving results and creating videos
    """
    results = [NelderMead_spo, NelderMead_indago, L_BFSG_B_spo, MSGS_indago]
    save_title = ['Nelder_Mead_scipy', 'Nelder_Mead_indago', 'L_BFGS_B_scipy', 'MSGS_indago']
    plot_title_average = ['Nelder_Mead_scipy_average', 'Nelder_Mead_indago_average', 'L_BFGS_B_scipy_average', 'MSGS_indago_average']
    plot_title_best = ['Nelder_Mead_scipy_best', 'Nelder_Mead_indago_best', 'L_BFGS_B_scipy_best', 'MSGS_indago_best']
    
    for i in range(len(results)):
        print(f'Fitness = {results[i][:, 6]}')
        np.savetxt(f'results/{save_title[i]}.txt', results[i], delimiter=',')
        average_design = results[i][n_optim//2, 0:6]
        best_design = results[i][0, 0:6]
        system(average_design, plot=True, animate=True, plot_title=plot_title_average[i])
        plt.close()
        system(best_design, plot=True, animate=True, plot_title=plot_title_best[i])
        plt.close()