import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Paramètres
eps = 0.1
l = 1
diametre = l/10  # Diamètre du tuyau
L = 7/(2*l)
pas = 50
nb_pt_sin1 = 350
import sympy as sp

#On définit des fonctions qui vont représenter les fonctions sin , leur dérives ainsi qu'une interpolation cubique

def cubic_interpolation(x0, y0, x1, y1, dy0, dy1):
    # DEF
    x, a, b, c, d = sp.symbols('x a b c d')
    
    # f(x) = ax^3 + bx^2 + cx + d
    f = a*x**3 + b*x**2 + c*x + d
    
    # Eq
    eq1 = f.subs(x, x0) - y0  # f(x0) = y0
    eq2 = f.subs(x, x1) - y1  # f(x1) = y1
    eq3 = sp.diff(f, x).subs(x, x0) - dy0  # f'(x0) = dy0
    eq4 = sp.diff(f, x).subs(x, x1) - dy1  # f'(x1) = dy1
    
    solution = sp.solve([eq1, eq2, eq3, eq4], (a, b, c, d))
    
    # Polynome Final
    polynomial = f.subs(solution)

    f_numeric = sp.lambdify(x, polynomial)  # Fonction
    return f_numeric
def dinvf(x,y):
    i = 0
    while y > l/4:
        y -= l/2
        i += 1
    while x > l/2 :
        x -= l
    return (-1)**i/(np.pi*np.sqrt(1-(2*x/l)**2))
    
    
def f(x):
    return l*np.sin(2*np.pi*x/l)/2

def df(x):
    return np.pi*np.cos(2*np.pi*x/l)



#On crée le 1er tuyaux serpentin
x1 = np.linspace(-1/(5*l)+eps, L-1/(2*l)-eps, nb_pt_sin1 )  # Axe horizontal
y_centre = l/2* np.sin(2 * np.pi * l * x1) # Ordonnées de la courbe sinusoïdale (ligne centrale)
freq = l
dx = np.gradient(x1)   # Dérivée de x1 ET Y
dy = np.gradient(y_centre)
norme = np.sqrt(dx**2 + dy**2)

# Calcul du vecteur normal unitaire à chaque point de la sinusoïde
nx = -dy / norme
ny = dx / norme
# Calcul des points du bord supérieur et inf de la "bande" autour de la sinusoïde
x_sup1 = x1 + (diametre / 2) * nx
y_sup1 = y_centre + (diametre / 2) * ny
x_inf1 = x1 - (diametre / 2) * nx
y_inf1 = y_centre - (diametre / 2) * ny

x11 = y_inf1[::-1]
y11 = x_inf1[::-1]#Les points doivent etre dans le bonne ordre sur le fichier txt

# Tracé du tuyau
plt.plot(y_sup1,x_sup1 , 'pink')#Bord inf
plt.plot(y_inf1, x_inf1, 'b')#Bord supérieur


#On crée le 1er tuyau
x2 = np.linspace(-3*l/2,-1*l/2, pas)  # Axe horizontal
y_inf2 = (-1/(4*l)+ (diametre/2))*np.ones(pas) #coordonné y des tuyaux rectiligne
y_sup2 = (-1/(4*l)+(3*diametre/2))*np.ones(pas)
plt.plot(x2,y_sup2, 'black')#Bord supérieur
plt.plot(x2,y_inf2, 'pink')#Bord inf

x13 = x2[::-1] #Les points doivent etre dans le bonne ordre 


#LIAISON SIN ET TUYAU SIN1 via cubic interpolation
x_filtered = x_inf1
x_filtered1 = x_sup1

y_filtered = y_inf1
y_filtered1 = y_sup1

#On fait une interpolation cubique basé sur les point finaux du tuyaux et les premiers points du tuyaux
poly2 = cubic_interpolation(list(x2)[-1],list(y_sup2)[-1],list(y_filtered)[0],list(x_filtered)[0],0,dinvf(list(y_filtered)[0],list(x_filtered)[0]))
poly2bis = cubic_interpolation(list(x2)[-1],list(y_inf2)[-1],list(y_filtered1)[0],list(x_filtered1)[0],0,dinvf(list(y_filtered1)[0],list(x_filtered1)[0]))
# Axe horizontal pour les points interpolés (pente supérieure et inférieure)
list_x2_p1 = [i for i in np.linspace(list(x2)[-1],list(y_filtered)[0],pas)] 
list_x2_p2 = [i for i in np.linspace(list(x2)[-1],list(y_filtered1)[0],pas)]

# Calcul des ordonnées correspondantes via les polynômes d'interpolation
list_y2_p1 = [poly2(i) for i in np.linspace(list(x2)[-1],list(y_filtered)[0],pas)]
list_y2_p2 = [poly2bis(i) for i in np.linspace(list(x2)[-1],list(y_filtered1)[0],pas)]
#On force le fait que les points de depart et arrivé soient exactement les mêmes à cause de la marge d'ereur de la machine
list_y2_p1[0] = y_sup2[0] 
list_y2_p1[-1] = x_inf1[0]
list_y2_p2[0] = y_inf2[-1]
list_y2_p2[-1] = x_sup1[0]

plt.plot(list_x2_p1,list_y2_p1, color = 'pink')
plt.plot(list_x2_p2,list_y2_p2, color = 'b')

list_x2_p1 = list_x2_p1[::-1]
list_y2_p1 = list_y2_p1[::-1] #Les points doivent etre dans le bonne ordre sur le fichier txt


#On crée le 2nd tuyaux serpentin, meme chose que pour le 1erv(cf ligne 54/60)
L = 2/l
x3 = np.linspace(1/(3*l)+eps, L+1/(2*l), 200)  
y_centre = l/2* np.sin(2 * np.pi * l * x3)
dx = np.gradient(x3)
dy = np.gradient(y_centre)
norme = np.sqrt(dx**2 + dy**2)
nx = -dy / norme
ny = dx / norme

x_sup3 = x3 + (diametre / 2) * nx
y_sup3 = y_centre + (diametre / 2) * ny
x_inf3 = x3 - (diametre / 2) * nx
y_inf3 = y_centre - (diametre / 2) * ny

y_sup3=l+y_sup3
y_inf3=l+y_inf3 #on ajoute l a chaque points pour decaler la structure
plt.plot(y_sup3,x_sup3 , 'r')#Bord supérieur
plt.plot(y_inf3, x_inf3, 'b')#Bord inf

x3 = y_inf3[::-1]
y3 = x_inf3[::-1] #Les points doivent etre dans le bonne ordresur le fichier txt


#On crée le tuyau de fin
x4 = np.linspace(3*l/2,5*l/2, pas)  # Axe horizontal
y_inf4 = (1/(4*l)+ (diametre / 2))*np.ones(pas)#coordonné y des tuyaux rectiligne
y_sup4 = (1/(4*l)+(3*diametre/2))*np.ones(pas)

plt.plot(x4,y_sup4, 'black')#Bord supérieur
plt.plot(x4,y_inf4, 'black')#Bord inf
plt.plot([x4[-1],x4[-1]],[y_inf4[-1],y_sup4[-1]])

x7 = x4[::-1] #Les points doivent etre dans le bonne ordre


#Fin du tuyau, le out 
x6 = x4[-1]*np.ones(pas)
y6 = np.linspace(y_inf4[-1],y_sup4[-1],pas)

#LIAISON SIN ET TUYAU SIN2 via cubic interpolation
y_filtered_ = y_inf3 
y_filtered1_ = y_sup3
x_filtered_ =x_inf3
x_filtered1_ = x_sup3

#On refait une interpolation comme précedemment mais entre le 2nd serpentin et le tuyaux de fin
poly3 = cubic_interpolation(list(y_filtered_)[0],list(x_filtered_)[0],list(x4)[0],list(y_inf4)[0],dinvf(list(y_filtered_)[0],list(x_filtered_)[0]),0)
poly3bis = cubic_interpolation(list(y_filtered1_)[0],list(x_filtered1_)[0],list(x4)[0],list(y_sup4)[0],dinvf(list(y_filtered1_)[0],list(x_filtered1_)[0]),0)

list_x3_p1 = [i for i in np.linspace(list(y_filtered_)[0],list(x4)[0],pas)]
list_x3_p2 = [i for i in np.linspace(list(y_filtered1_)[0],list(x4)[0],pas)]
list_y3_p1 = [poly3(i) for i in np.linspace(list(y_filtered_)[0],list(x4)[0],pas)]
list_y3_p2 = [poly3bis(i) for i in np.linspace(list(y_filtered1_)[0],list(x4)[0],pas)]
#On force le fait que les points soient exactement les mêmes à cause de la marge d'ereur de la machine
list_y3_p1[0] = x_inf3[0]
list_y3_p1[-1] = y_inf4[0]
list_y3_p2[0] = x_sup3[0]
list_y3_p2[-1] = y_sup4[0]

plt.plot(list_x3_p1,list_y3_p1, color = 'pink')
plt.plot(list_x3_p2,list_y3_p2, color = 'b')

list_x3_p2 = list_x3_p2[::-1]
list_y3_p2 = list_y3_p2[::-1]#Les points doivent etre dans le bonne ordre


#Liaison entre les 2 serpentins via cubic interpolation

poly4 = cubic_interpolation(list(y_filtered)[-1],list(x_filtered)[-1],list(y_filtered1_)[-1],list(x_filtered1_)[-1],dinvf(list(y_filtered)[-1],list(x_filtered)[-1]),dinvf(f(21*l/8),21*l/8))
poly4bis = cubic_interpolation(list(y_filtered1)[-1],list(x_filtered1)[-1],list(y_filtered_)[-1],list(x_filtered_)[-1],dinvf(list(y_filtered)[-1],list(x_filtered)[-1]),dinvf(f(21*l/8),21*l/8))

list_x4_p1 = [i for i in np.linspace(list(y_filtered)[-1],list(y_filtered1_)[-1],3*pas)]
list_x4_p2 = [i for i in np.linspace(list(y_filtered1)[-1],list(y_filtered_)[-1],3*pas)]
list_y4_p1 = [poly4(i) for i in np.linspace(list(y_filtered)[-1],list(y_filtered1_)[-1],3*pas)]
list_y4_p2 = [poly4bis(i) for i in np.linspace(list(y_filtered1)[-1],list(y_filtered_)[-1],3*pas)]
#On force le fait que les points soient exactement les mêmes à cause de la marge d'ereur de la machine
list_y4_p1[0] = x_inf1[-1]
list_y4_p1[-1] = x_sup3[-1]
list_y4_p2[0] = x_sup1[-1]
list_y4_p2[-1] = x_inf3[-1]

# Tracé de l'interpolation fluide
plt.plot(list_x4_p1,list_y4_p1, color = 'b')
plt.plot(list_x4_p2,list_y4_p2, color = 'r')
#Les points doivent etre dans le bonne ordre
list_x4_p1 = list_x4_p1[::-1]
list_y4_p1 = list_y4_p1[::-1]


#On créer les 2 tuyaux de depart 
#Angle d'entrée et coordonée des points x et y
theta = 45 * np.pi / 180
x_pt = [-3*l/2,-3*l/2-(l/2)*np.cos(theta),-3*l/2-(l/2)*np.cos(theta),-3*l/2-(l/2)*np.cos(theta)+(eps/2+(l/2)*np.sin(theta)-eps)/np.tan(theta),-3*l/2-(l/2)*np.cos(theta),-3*l/2-(l/2)*np.cos(theta),-3*l/2-(l/2)*np.cos(theta),-3*l/2]

y_pt = [-l / 4 - eps,-l / 4 - eps - (l / 2) * np.sin(theta),-l / 4 - eps - (l / 2) * np.sin(theta) + eps,-l / 4 - eps / 2,-l / 4 - eps / 2 + (l / 2) * np.sin(theta) + eps / 2 - eps,-l / 4 + (l / 2) * np.sin(theta),-l / 4 + (l / 2) * np.sin(theta),-l / 4]

#On decale totalement la structure
for i in range(0, len(y_pt)): 
	y_pt[i]+= 3*diametre/2
	
nbpt = 50

#on creer les listes de points
x19 = np.linspace(x_pt[1],x_pt[0],nbpt)
y19 = np.linspace(y_pt[1], y_pt[0],nbpt)

x18 = np.linspace(x_pt[2],x_pt[1],nbpt)
y18 = np.linspace(y_pt[2], y_pt[1],nbpt)

x17 = np.linspace(x_pt[3],x_pt[2],nbpt)
y17 = np.linspace(y_pt[3], y_pt[2],nbpt)

x16 = np.linspace(x_pt[4],x_pt[3],nbpt)
y16 = np.linspace(y_pt[4], y_pt[3],nbpt)

x15 = np.linspace(x_pt[5],x_pt[4],nbpt)
y15 = np.linspace(y_pt[5], y_pt[4],nbpt)

x14 = np.linspace(x_pt[7],x_pt[6],nbpt)
y14 = np.linspace(y_pt[7], y_pt[6],nbpt)

plt.plot(x_pt, y_pt)

#On crée les 6 fichiers qui representerons les bordures

xwall1 = np.concatenate([
    x19[:-1], x2[:-1], list_x2_p2[:-1], y_sup1[:-1], 
    list_x4_p2[:-1], x3[:-1], list_x3_p1[:-1], x4])
ywall1 = np.concatenate([
    y19[:-1], y_inf2[:-1], list_y2_p2[:-1], x_sup1[:-1], 
    list_y4_p2[:-1], y3[:-1], list_y3_p1[:-1], y_inf4])
xy1final = np.column_stack((xwall1,ywall1))
np.savetxt("point1.txt", xy1final, delimiter=" ", header=f"{xy1final.shape[0]}", comments='')


xout = np.column_stack((x6,y6))
np.savetxt("point2.txt", xout, delimiter=" ", header=f"{xout.shape[0]}", comments='')


xwall2 = np.concatenate([
    x7[:-1], list_x3_p2[:-1], y_sup3[:-1], list_x4_p1[:-1], 
    x11[:-1], list_x2_p1[:-1], x13[:-1], x14])
ywall2 = np.concatenate([
    y_sup4[:-1], list_y3_p2[:-1], x_sup3[:-1], list_y4_p1[:-1], 
    y11[:-1], list_y2_p1[:-1], y_sup2[:-1], y14])
xy2final = np.column_stack((xwall2,ywall2))
np.savetxt("point3.txt", xy2final, delimiter=" ", header=f"{xy2final.shape[0]}", comments='')


xin1 = np.column_stack((x15,y15))
np.savetxt("point4.txt", xin1, delimiter=" ", header=f"{xin1.shape[0]}", comments='') 


xwall3 = np.concatenate([x16, x17])
ywall3 = np.concatenate([y16, y17])
xy3final = np.column_stack((xwall3,ywall3))
np.savetxt("point5.txt", xy3final, delimiter=" ", header=f"{xy3final.shape[0]}", comments='')


xin2 = np.column_stack((x18,y18))
np.savetxt("point6.txt", xin2, delimiter=" ", header=f"{xin2.shape[0]}", comments='') 

plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Tuyau sinusoïdal (Fréquence = {freq})")
plt.axis("equal")
plt.grid()
plt.show()



