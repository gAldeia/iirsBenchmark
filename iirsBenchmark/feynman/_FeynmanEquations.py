# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
internal use implementation, which works by mapping the name of an expression
with various information, such as a representation in LaTeX, a function in
python that implements the equation, among other information.

The python functions are implemented using jax numpy, which allows to
automatically differentiate the expressions. Having access to prediction
gradients --- in this particular module iirsBenchmark --- makes possible to 
apply every single explainer to the regressor.
"""

import jax.numpy as jnp

pi = jnp.pi


feynmanPyData = {
    'I.10.7' : {
        'string expression' : 'm_0/jnp.sqrt(1-v**2/c**2)',
        'latex expression'  : r'm_0/\sqrt{(1-v^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m_0,v,c:m_0/jnp.sqrt(1-v**2/c**2))(*args)
    },
    'I.11.19' : {
        'string expression' : 'x1*y1+x2*y2+x3*y3',
        'latex expression'  : r'x1*y1+x2*y2+x3*y3',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda x1,x2,x3,y1,y2,y3:x1*y1+x2*y2+x3*y3)(*args)
    },
    'I.12.1' : {
        'string expression' : 'mu*Nn',
        'latex expression'  : r'mu*Nn',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mu,Nn:mu*Nn)(*args)
    },
    'I.12.11' : {
        'string expression' : 'q*(Ef+B*v*jnp.sin(theta))',
        'latex expression'  : r'q*{(Ef+B*v*sin{(\theta)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda q,Ef,B,v,theta:q*(Ef+B*v*jnp.sin(theta)))(*args)
    },
    'I.12.2' : {
        'string expression' : 'q1*q2*r/(4*pi*epsilon*r**3)',
        'latex expression'  : r'q1*q2*r/{(4*\pi*\epsilon*r^3)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q1,q2,epsilon,r:q1*q2*r/(4*pi*epsilon*r**3))(*args)
    },
    'I.12.4' : {
        'string expression' : 'q1*r/(4*pi*epsilon*r**3)',
        'latex expression'  : r'q1*r/{(4*\pi*\epsilon*r^3)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q1,epsilon,r:q1*r/(4*pi*epsilon*r**3))(*args)
    },
    'I.12.5' : {
        'string expression' : 'q2*Ef',
        'latex expression'  : r'q2*Ef',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q2,Ef:q2*Ef)(*args)
    },
    'I.13.12' : {
        'string expression' : 'G*m1*m2*(1/r2-1/r1)',
        'latex expression'  : r'G*m1*m2*{(1/r2-1/r1)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m1,m2,r1,r2,G:G*m1*m2*(1/r2-1/r1))(*args)
    },
    'I.13.4' : {
        'string expression' : '1/2*m*(v**2+u**2+w**2)',
        'latex expression'  : r'1/2*m*{(v^2+u^2+w^2)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m,v,u,w:1/2*m*(v**2+u**2+w**2))(*args)
    },
    'I.14.3' : {
        'string expression' : 'm*g*z',
        'latex expression'  : r'm*g*z',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m,g,z:m*g*z)(*args)
    },
    'I.14.4' : {
        'string expression' : '1/2*k_spring*x**2',
        'latex expression'  : r'1/2*k_spring*x^2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda k_spring,x:1/2*k_spring*x**2)(*args)
    },
    'I.15.10' : {
        'string expression' : 'm_0*v/jnp.sqrt(1-v**2/c**2)',
        'latex expression'  : r'm_0*v/\sqrt{(1-v^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m_0,v,c:m_0*v/jnp.sqrt(1-v**2/c**2))(*args)
    },
    'I.15.3t' : {
        'string expression' : '(t-u*x/c**2)/jnp.sqrt(1-u**2/c**2)',
        'latex expression'  : r'{(t-u*x/c^2)}/\sqrt{(1-u^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x,c,u,t:(t-u*x/c**2)/jnp.sqrt(1-u**2/c**2))(*args)
    },
    'I.15.3x' : {
        'string expression' : '(x-u*t)/jnp.sqrt(1-u**2/c**2)',
        'latex expression'  : r'{(x-u*t)}/\sqrt{(1-u^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x,u,c,t:(x-u*t)/jnp.sqrt(1-u**2/c**2))(*args)
    },
    'I.16.6' : {
        'string expression' : '(u+v)/(1+u*v/c**2)',
        'latex expression'  : r'{(u+v)}/{(1+u*v/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda c,v,u:(u+v)/(1+u*v/c**2))(*args)
    },
    'I.18.12' : {
        'string expression' : 'r*F*jnp.sin(theta)',
        'latex expression'  : r'r*F*sin{(\theta)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda r,F,theta:r*F*jnp.sin(theta))(*args)
    },
    'I.18.14' : {
        'string expression' : 'm*r*v*jnp.sin(theta)',
        'latex expression'  : r'm*r*v*sin{(\theta)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m,r,v,theta:m*r*v*jnp.sin(theta))(*args)
    },
    'I.18.4' : {
        'string expression' : '(m1*r1+m2*r2)/(m1+m2)',
        'latex expression'  : r'{(m1*r1+m2*r2)}/{(m1+m2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m1,m2,r1,r2:(m1*r1+m2*r2)/(m1+m2))(*args)
    },
    'I.24.6' : {
        'string expression' : '1/2*m*(omega**2+omega_0**2)*1/2*x**2',
        'latex expression'  : r'1/2*m*{(\omega^2+\omega_0^2)}*1/2*x^2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m,omega,omega_0,x:1/2*m*(omega**2+omega_0**2)*1/2*x**2)(*args)
    },
    'I.25.13' : {
        'string expression' : 'q/C',
        'latex expression'  : r'q/C',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,C:q/C)(*args)
    },
    'I.26.2' : {
        'string expression' : 'jnp.arcsin(n*jnp.sin(theta2))',
        'latex expression'  : r'arcsin{(n*sin{(\theta_2)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n,theta2:jnp.arcsin(n*jnp.sin(theta2)))(*args)
    },
    'I.27.6' : {
        'string expression' : '1/(1/d1+n/d2)',
        'latex expression'  : r'1/{(1/d1+n/d2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda d1,d2,n:1/(1/d1+n/d2))(*args)
    },
    'I.29.16' : {
        'string expression' : 'jnp.sqrt(x1**2+x2**2-2*x1*x2*jnp.cos(theta1-theta2))',
        'latex expression'  : r'\sqrt{(x1^2+x2^2-2*x1*x2*cos{(\theta_1-\theta_2)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x1,x2,theta1,theta2:jnp.sqrt(x1**2+x2**2-2*x1*x2*jnp.cos(theta1-theta2)))(*args)
    },
    'I.29.4' : {
        'string expression' : 'omega/c',
        'latex expression'  : r'\omega/c',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda omega,c:omega/c)(*args)
    },
    'I.30.3' : {
        'string expression' : 'Int_0*jnp.sin(n*theta/2)**2/jnp.sin(theta/2)**2',
        'latex expression'  : r'Int_0*sin{(n*\theta/2)}^2/sin{(\theta/2)}^2',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda Int_0,theta,n:Int_0*jnp.sin(n*theta/2)**2/jnp.sin(theta/2)**2)(*args)
    },
    'I.30.5' : {
        'string expression' : 'jnp.arcsin(lambd/(n*d))',
        'latex expression'  : r'arcsin{(\lambda/{(n*d)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda lambd,d,n:jnp.arcsin(lambd/(n*d)))(*args)
    },
    'I.32.17' : {
        'string expression' : '(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)',
        'latex expression'  : r'{(1/2*\epsilon*c*Ef^2)}*{(8*\pi*r^2/3)}*{(\omega^4/{(\omega^2-\omega_0^2)}^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda epsilon,c,Ef,r,omega,omega_0:(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2))(*args)
    },
    'I.32.5' : {
        'string expression' : 'q**2*a**2/(6*pi*epsilon*c**3)',
        'latex expression'  : r'q^2*a^2/{(6*\pi*\epsilon*c^3)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,a,epsilon,c:q**2*a**2/(6*pi*epsilon*c**3))(*args)
    },
    'I.34.1' : {
        'string expression' : 'omega_0/(1-v/c)',
        'latex expression'  : r'\omega_0/{(1-v/c)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda c,v,omega_0:omega_0/(1-v/c))(*args)
    },
    'I.34.14' : {
        'string expression' : '(1+v/c)/jnp.sqrt(1-v**2/c**2)*omega_0',
        'latex expression'  : r'{(1+v/c)}/\sqrt{(1-v^2/c^2)}*\omega_0',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda c,v,omega_0:(1+v/c)/jnp.sqrt(1-v**2/c**2)*omega_0)(*args)
    },
    'I.34.27' : {
        'string expression' : '(h/(2*pi))*omega',
        'latex expression'  : r'{(h/{(2*\pi)})}*\omega',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda omega,h:(h/(2*pi))*omega)(*args)
    },
    'I.34.8' : {
        'string expression' : 'q*v*B/p',
        'latex expression'  : r'q*v*B/p',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,v,B,p:q*v*B/p)(*args)
    },
    'I.37.4' : {
        'string expression' : 'I1+I2+2*jnp.sqrt(I1*I2)*jnp.cos(delta)',
        'latex expression'  : r'I1+I2+2*\sqrt{(I1*I2)}*cos{(\delta)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda I1,I2,delta:I1+I2+2*jnp.sqrt(I1*I2)*jnp.cos(delta))(*args)
    },
    'I.38.12' : {
        'string expression' : '4*pi*epsilon*(h/(2*pi))**2/(m*q**2)',
        'latex expression'  : r'4*\pi*\epsilon*{(h/{(2*\pi)})}^2/{(m*q^2)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m,q,h,epsilon:4*pi*epsilon*(h/(2*pi))**2/(m*q**2))(*args)
    },
    'I.39.1' : {
        'string expression' : '3/2*pr*V',
        'latex expression'  : r'3/2*pr*V',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda pr,V:3/2*pr*V)(*args)
    },
    'I.39.11' : {
        'string expression' : '1/(gamma-1)*pr*V',
        'latex expression'  : r'1/{(\gamma-1)}*pr*V',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda gamma,pr,V:1/(gamma-1)*pr*V)(*args)
    },
    'I.39.22' : {
        'string expression' : 'n*kb*T/V',
        'latex expression'  : r'n*kb*T/V',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda n,T,V,kb:n*kb*T/V)(*args)
    },
    'I.40.1' : {
        'string expression' : 'n_0*jnp.exp(-m*g*x/(kb*T))',
        'latex expression'  : r'n_0*e^{(-m*g*x/{(kb*T)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda n_0,m,x,T,g,kb:n_0*jnp.exp(-m*g*x/(kb*T)))(*args)
    },
    'I.41.16' : {
        'string expression' : 'h/(2*pi)*omega**3/(pi**2*c**2*(jnp.exp((h/(2*pi))*omega/(kb*T))-1))',
        'latex expression'  : r'h/{(2*\pi)}*\omega^3/{(\pi^2*c^2*{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda omega,T,h,kb,c:h/(2*pi)*omega**3/(pi**2*c**2*(jnp.exp((h/(2*pi))*omega/(kb*T))-1)))(*args)
    },
    'I.43.16' : {
        'string expression' : 'mu_drift*q*Volt/d',
        'latex expression'  : r'mu_drift*q*Volt/d',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mu_drift,q,Volt,d:mu_drift*q*Volt/d)(*args)
    },
    'I.43.31' : {
        'string expression' : 'mob*kb*T',
        'latex expression'  : r'mob*kb*T',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mob,T,kb:mob*kb*T)(*args)
    },
    'I.43.43' : {
        'string expression' : '1/(gamma-1)*kb*v/A',
        'latex expression'  : r'1/{(\gamma-1)}*kb*v/A',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda gamma,kb,A,v:1/(gamma-1)*kb*v/A)(*args)
    },
    'I.44.4' : {
        'string expression' : 'n*kb*T*jnp.log(V2/V1)',
        'latex expression'  : r'n*kb*T*log{(V2/V1)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n,kb,T,V1,V2:n*kb*T*jnp.log(V2/V1))(*args)
    },
    'I.47.23' : {
        'string expression' : 'jnp.sqrt(gamma*pr/rho)',
        'latex expression'  : r'\sqrt{(\gamma*pr/\rho)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda gamma,pr,rho:jnp.sqrt(gamma*pr/rho))(*args)
    },
    'I.48.20' : {
        'string expression' : 'm*c**2/jnp.sqrt(1-v**2/c**2)',
        'latex expression'  : r'm*c^2/\sqrt{(1-v^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m,v,c:m*c**2/jnp.sqrt(1-v**2/c**2))(*args)
    },
    'I.50.26' : {
        'string expression' : 'x1*(jnp.cos(omega*t)+alpha*jnp.cos(omega*t)**2)',
        'latex expression'  : r'x1*{(cos{(\omega*t)}+\alpha*cos{(\omega*t)}^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x1,omega,t,alpha:x1*(jnp.cos(omega*t)+alpha*jnp.cos(omega*t)**2))(*args)
    },
    'I.6.2' : {
        'string expression' : 'jnp.exp(-(theta/sigma)**2/2)/(jnp.sqrt(2*pi)*sigma)',
        'latex expression'  : r'e^{(-{(\theta/\sigma)}^2/2)}/{(\sqrt{(2*\pi)}*\sigma)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda sigma,theta:jnp.exp(-(theta/sigma)**2/2)/(jnp.sqrt(2*pi)*sigma))(*args)
    },
    'I.6.2a' : {
        'string expression' : 'jnp.exp(-theta**2/2)/jnp.sqrt(2*pi)',
        'latex expression'  : r'e^{(-\theta^2/2)}/\sqrt{(2*\pi)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda theta:jnp.exp(-theta**2/2)/jnp.sqrt(2*pi))(*args)
    },
    'I.6.2b' : {
        'string expression' : 'jnp.exp(-((theta-theta1)/sigma)**2/2)/(jnp.sqrt(2*pi)*sigma)',
        'latex expression'  : r'e^{(-{({(\theta-\theta_1)}/\sigma)}^2/2)}/{(\sqrt{(2*\pi)}*\sigma)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda sigma,theta,theta1:jnp.exp(-((theta-theta1)/sigma)**2/2)/(jnp.sqrt(2*pi)*sigma))(*args)
    },
    'I.8.14' : {
        'string expression' : 'jnp.sqrt((x2-x1)**2+(y2-y1)**2)',
        'latex expression'  : r'\sqrt{({(x2-x1)}^2+{(y2-y1)}^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda x1,x2,y1,y2:jnp.sqrt((x2-x1)**2+(y2-y1)**2))(*args)
    },
    'I.9.18' : {
        'string expression' : 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)',
        'latex expression'  : r'G*m1*m2/{({(x2-x1)}^2+{(y2-y1)}^2+{(z2-z1)}^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda m1,m2,G,x1,x2,y1,y2,z1,z2:G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2))(*args)
    },
    'II.10.9' : {
        'string expression' : 'sigma_den/epsilon*1/(1+chi)',
        'latex expression'  : r'\sigma_den/\epsilon*1/{(1+chi)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda sigma_den,epsilon,chi:sigma_den/epsilon*1/(1+chi))(*args)
    },
    'II.11.17' : {
        'string expression' : 'n_0*(1+p_d*Ef*jnp.cos(theta)/(kb*T))',
        'latex expression'  : r'n_0*{(1+p_d*Ef*cos{(\theta)}/{(kb*T)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n_0,kb,T,theta,p_d,Ef:n_0*(1+p_d*Ef*jnp.cos(theta)/(kb*T)))(*args)
    },
    'II.11.20' : {
        'string expression' : 'n_rho*p_d**2*Ef/(3*kb*T)',
        'latex expression'  : r'n_\rho*p_d^2*Ef/{(3*kb*T)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda n_rho,p_d,Ef,kb,T:n_rho*p_d**2*Ef/(3*kb*T))(*args)
    },
    'II.11.27' : {
        'string expression' : 'n*alpha/(1-(n*alpha/3))*epsilon*Ef',
        'latex expression'  : r'n*\alpha/{(1-{(n*\alpha/3)})}*\epsilon*Ef',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n,alpha,epsilon,Ef:n*alpha/(1-(n*alpha/3))*epsilon*Ef)(*args)
    },
    'II.11.28' : {
        'string expression' : '1+n*alpha/(1-(n*alpha/3))',
        'latex expression'  : r'1+n*\alpha/{(1-{(n*\alpha/3)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n,alpha:1+n*alpha/(1-(n*alpha/3)))(*args)
    },
    'II.11.3' : {
        'string expression' : 'q*Ef/(m*(omega_0**2-omega**2))',
        'latex expression'  : r'q*Ef/{(m*{(\omega_0^2-\omega^2)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda q,Ef,m,omega_0,omega:q*Ef/(m*(omega_0**2-omega**2)))(*args)
    },
    'II.13.17' : {
        'string expression' : '1/(4*pi*epsilon*c**2)*2*I/r',
        'latex expression'  : r'1/{(4*\pi*\epsilon*c^2)}*2*I/r',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda epsilon,c,I,r:1/(4*pi*epsilon*c**2)*2*I/r)(*args)
    },
    'II.13.23' : {
        'string expression' : 'rho_c_0/jnp.sqrt(1-v**2/c**2)',
        'latex expression'  : r'\rho_{c_0}/\sqrt{(1-v^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda rho_c_0,v,c:rho_c_0/jnp.sqrt(1-v**2/c**2))(*args)
    },
    'II.13.34' : {
        'string expression' : 'rho_c_0*v/jnp.sqrt(1-v**2/c**2)',
        'latex expression'  : r'\rho_{c_0}*v/\sqrt{(1-v^2/c^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda rho_c_0,v,c:rho_c_0*v/jnp.sqrt(1-v**2/c**2))(*args)
    },
    'II.15.4' : {
        'string expression' : '-mom*B*jnp.cos(theta)',
        'latex expression'  : r'-mom*B*cos{(\theta)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda mom,B,theta:-mom*B*jnp.cos(theta))(*args)
    },
    'II.15.5' : {
        'string expression' : '-p_d*Ef*jnp.cos(theta)',
        'latex expression'  : r'-p_d*Ef*cos{(\theta)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda p_d,Ef,theta:-p_d*Ef*jnp.cos(theta))(*args)
    },
    'II.2.42' : {
        'string expression' : 'kappa*(T2-T1)*A/d',
        'latex expression'  : r'kappa*{(T2-T1)}*A/d',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda kappa,T1,T2,A,d:kappa*(T2-T1)*A/d)(*args)
    },
    'II.21.32' : {
        'string expression' : 'q/(4*pi*epsilon*r*(1-v/c))',
        'latex expression'  : r'q/{(4*\pi*\epsilon*r*{(1-v/c)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda q,epsilon,r,v,c:q/(4*pi*epsilon*r*(1-v/c)))(*args)
    },
    'II.24.17' : {
        'string expression' : 'jnp.sqrt(omega**2/c**2-pi**2/d**2)',
        'latex expression'  : r'\sqrt{(\omega^2/c^2-\pi^2/d^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda omega,c,d:jnp.sqrt(omega**2/c**2-pi**2/d**2))(*args)
    },
    'II.27.16' : {
        'string expression' : 'epsilon*c*Ef**2',
        'latex expression'  : r'\epsilon*c*Ef^2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda epsilon,c,Ef:epsilon*c*Ef**2)(*args)
    },
    'II.27.18' : {
        'string expression' : 'epsilon*Ef**2',
        'latex expression'  : r'\epsilon*Ef^2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda epsilon,Ef:epsilon*Ef**2)(*args)
    },
    'II.3.24' : {
        'string expression' : 'Pwr/(4*pi*r**2)',
        'latex expression'  : r'Pwr/{(4*\pi*r^2)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda Pwr,r:Pwr/(4*pi*r**2))(*args)
    },
    'II.34.11' : {
        'string expression' : 'g_*q*B/(2*m)',
        'latex expression'  : r'g_*q*B/{(2*m)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda g_,q,B,m:g_*q*B/(2*m))(*args)
    },
    'II.34.2' : {
        'string expression' : 'q*v*r/2',
        'latex expression'  : r'q*v*r/2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,v,r:q*v*r/2)(*args)
    },
    'II.34.29a' : {
        'string expression' : 'q*h/(4*pi*m)',
        'latex expression'  : r'q*h/{(4*\pi*m)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,h,m:q*h/(4*pi*m))(*args)
    },
    'II.34.29b' : {
        'string expression' : 'g_*mom*B*Jz/(h/(2*pi))',
        'latex expression'  : r'g_*mom*B*Jz/{(h/{(2*\pi)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda g_,h,Jz,mom,B:g_*mom*B*Jz/(h/(2*pi)))(*args)
    },
    'II.34.2a' : {
        'string expression' : 'q*v/(2*pi*r)',
        'latex expression'  : r'q*v/{(2*\pi*r)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,v,r:q*v/(2*pi*r))(*args)
    },
    'II.35.18' : {
        'string expression' : 'n_0/(jnp.exp(mom*B/(kb*T))+jnp.exp(-mom*B/(kb*T)))',
        'latex expression'  : r'n_0/{(e^{(mom*B/{(kb*T)})}+e^{(-mom*B/{(kb*T)})})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n_0,kb,T,mom,B:n_0/(jnp.exp(mom*B/(kb*T))+jnp.exp(-mom*B/(kb*T))))(*args)
    },
    'II.35.21' : {
        'string expression' : 'n_rho*mom*jnp.tanh(mom*B/(kb*T))',
        'latex expression'  : r'n_\rho*mom*tanh{(mom*B/{(kb*T)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda n_rho,mom,B,kb,T:n_rho*mom*jnp.tanh(mom*B/(kb*T)))(*args)
    },
    'II.36.38' : {
        'string expression' : 'mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M',
        'latex expression'  : r'mom*H/{(kb*T)}+{(mom*\alpha)}/{(\epsilon*c^2*kb*T)}*M',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mom,H,kb,T,alpha,epsilon,c,M:mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M)(*args)
    },
    'II.37.1' : {
        'string expression' : 'mom*(1+chi)*B',
        'latex expression'  : r'mom*{(1+chi)}*B',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mom,B,chi:mom*(1+chi)*B)(*args)
    },
    'II.38.14' : {
        'string expression' : 'Y/(2*(1+sigma))',
        'latex expression'  : r'Y/{(2*{(1+\sigma)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda Y,sigma:Y/(2*(1+sigma)))(*args)
    },
    'II.38.3' : {
        'string expression' : 'Y*A*x/d',
        'latex expression'  : r'Y*A*x/d',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda Y,A,d,x:Y*A*x/d)(*args)
    },
    'II.4.23' : {
        'string expression' : 'q/(4*pi*epsilon*r)',
        'latex expression'  : r'q/{(4*\pi*\epsilon*r)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,epsilon,r:q/(4*pi*epsilon*r))(*args)
    },
    'II.6.11' : {
        'string expression' : '1/(4*pi*epsilon)*p_d*jnp.cos(theta)/r**2',
        'latex expression'  : r'1/{(4*\pi*\epsilon)}*p_d*cos{(\theta)}/r^2',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda epsilon,p_d,theta,r:1/(4*pi*epsilon)*p_d*jnp.cos(theta)/r**2)(*args)
    },
    'II.6.15a' : {
        'string expression' : 'p_d/(4*pi*epsilon)*3*z/r**5*jnp.sqrt(x**2+y**2)',
        'latex expression'  : r'p_d/{(4*\pi*\epsilon)}*3*z/r^5*\sqrt{(x^2+y^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda epsilon,p_d,r,x,y,z:p_d/(4*pi*epsilon)*3*z/r**5*jnp.sqrt(x**2+y**2))(*args)
    },
    'II.6.15b' : {
        'string expression' : 'p_d/(4*pi*epsilon)*3*jnp.cos(theta)*jnp.sin(theta)/r**3',
        'latex expression'  : r'p_d/{(4*\pi*\epsilon)}*3*cos{(\theta)}*sin{(\theta)}/r^3',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda epsilon,p_d,theta,r:p_d/(4*pi*epsilon)*3*jnp.cos(theta)*jnp.sin(theta)/r**3)(*args)
    },
    'II.8.31' : {
        'string expression' : 'epsilon*Ef**2/2',
        'latex expression'  : r'\epsilon*Ef^2/2',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda epsilon,Ef:epsilon*Ef**2/2)(*args)
    },
    'II.8.7' : {
        'string expression' : '3/5*q**2/(4*pi*epsilon*d)',
        'latex expression'  : r'3/5*q^2/{(4*\pi*\epsilon*d)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda q,epsilon,d:3/5*q**2/(4*pi*epsilon*d))(*args)
    },
    'III.10.19' : {
        'string expression' : 'mom*jnp.sqrt(Bx**2+By**2+Bz**2)',
        'latex expression'  : r'mom*\sqrt{(Bx^2+By^2+Bz^2)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda mom,Bx,By,Bz:mom*jnp.sqrt(Bx**2+By**2+Bz**2))(*args)
    },
    'III.12.43' : {
        'string expression' : 'n*(h/(2*pi))',
        'latex expression'  : r'n*{(h/{(2*\pi)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda n,h:n*(h/(2*pi)))(*args)
    },
    'III.13.18' : {
        'string expression' : '2*E_n*d**2*k/(h/(2*pi))',
        'latex expression'  : r'2*E_n*d^2*k/{(h/{(2*\pi)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda E_n,d,k,h:2*E_n*d**2*k/(h/(2*pi)))(*args)
    },
    'III.14.14' : {
        'string expression' : 'I_0*(jnp.exp(q*Volt/(kb*T))-1)',
        'latex expression'  : r'I_0*{(e^{(q*Volt/{(kb*T)})}-1)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda I_0,q,Volt,kb,T:I_0*(jnp.exp(q*Volt/(kb*T))-1))(*args)
    },
    'III.15.12' : {
        'string expression' : '2*U*(1-jnp.cos(k*d))',
        'latex expression'  : r'2*U*{(1-cos{(k*d)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda U,k,d:2*U*(1-jnp.cos(k*d)))(*args)
    },
    'III.15.14' : {
        'string expression' : '(h/(2*pi))**2/(2*E_n*d**2)',
        'latex expression'  : r'{(h/{(2*\pi)})}^2/{(2*E_n*d^2)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda h,E_n,d:(h/(2*pi))**2/(2*E_n*d**2))(*args)
    },
    'III.15.27' : {
        'string expression' : '2*pi*alpha/(n*d)',
        'latex expression'  : r'2*\pi*\alpha/{(n*d)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda alpha,n,d:2*pi*alpha/(n*d))(*args)
    },
    'III.17.37' : {
        'string expression' : 'beta*(1+alpha*jnp.cos(theta))',
        'latex expression'  : r'beta*{(1+\alpha*cos{(\theta)})}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda beta,alpha,theta:beta*(1+alpha*jnp.cos(theta)))(*args)
    },
    'III.19.51' : {
        'string expression' : '-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)',
        'latex expression'  : r'-m*q^4/{(2*{(4*\pi*\epsilon)}^2*{(h/{(2*\pi)})}^2)}*{(1/n^2)}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda m,q,h,n,epsilon:-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2))(*args)
    },
    'III.21.20' : {
        'string expression' : '-rho_c_0*q*A_vec/m',
        'latex expression'  : r'-\rho_{c_0}*q*A_vec/m',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda rho_c_0,q,A_vec,m:-rho_c_0*q*A_vec/m)(*args)
    },
    'III.4.32' : {
        'string expression' : '1/(jnp.exp((h/(2*pi))*omega/(kb*T))-1)',
        'latex expression'  : r'1/{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda h,omega,kb,T:1/(jnp.exp((h/(2*pi))*omega/(kb*T))-1))(*args)
    },
    'III.4.33' : {
        'string expression' : '(h/(2*pi))*omega/(jnp.exp((h/(2*pi))*omega/(kb*T))-1)',
        'latex expression'  : r'{(h/{(2*\pi)})}*\omega/{(e^{({(h/{(2*\pi)})}*\omega/{(kb*T)})}-1)}',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda h,omega,kb,T:(h/(2*pi))*omega/(jnp.exp((h/(2*pi))*omega/(kb*T))-1))(*args)
    },
    'III.7.38' : {
        'string expression' : '2*mom*B/(h/(2*pi))',
        'latex expression'  : r'2*mom*B/{(h/{(2*\pi)})}',
        'expressible by IT' : True,
        'python function'   : lambda args : (lambda mom,B,h:2*mom*B/(h/(2*pi)))(*args)
    },
    'III.8.54' : {
        'string expression' : 'jnp.sin(E_n*t/(h/(2*pi)))**2',
        'latex expression'  : r'sin{(E_n*t/{(h/{(2*\pi)})})}^2',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda E_n,t,h:jnp.sin(E_n*t/(h/(2*pi)))**2)(*args)
    },
    'III.9.52' : {
        'string expression' : '(p_d*Ef*t/(h/(2*pi)))*jnp.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2',
        'latex expression'  : r'(p_d*Ef*t/(h/(2*\pi)))*sin((\omega-\omega_0)*t/2)^2/((\omega-\omega_0)*t/2)^2',
        'expressible by IT' : False,
        'python function'   : lambda args : (lambda p_d,Ef,t,h,omega,omega_0:(p_d*Ef*t/(h/(2*pi)))*jnp.sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2)(*args)
    }
}