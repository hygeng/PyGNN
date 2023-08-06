import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import lobpcg


def cheby_coeff(g, order, quad_points, range1, range2):
    #  sgwt_cheby_coeff : Compute Chebyshev coefficients of given function
    #  function c=sgwt_cheby_coeff(g,order,N,arange)
    # 
    #  Inputs:
    #  g - function handle, should define function on arange
    #  order - maximum order Chebyshev coefficient to compute
    #  N (quad_points)- grid order used to compute quadrature (default is order+1)
    #  arange - interval of approximation (defaults to [-1,1] )
    if quad_points==None:
        quad_points = order+1
    a1=(range2-range1)/2
    a2=(range2+range1)/2
    c = np.zeros(order+1)
    for j in range(1, order+2):
        # c(j)=sum(    g(a1* cos( (pi*((1:N)-0.5))/N) + a2)  .*   cos(pi*(j-1)*((1:N)-.5)/N)      )*2/N
        # g_term = g(a1 * np.cos(        (np.arange(1,N+1)- 0.5) * np.pi/N )  + a2) 
        g_term = g(a1 * np.cos(        (np.arange(1,quad_points+1)- 0.5) * np.pi/quad_points )  + a2) 
        cos_term =      np.cos( (j-1) *(np.arange(1,quad_points+1) -0.5) * np.pi/quad_points )
        c[j-1] = sum( g_term * cos_term ) * 2/quad_points
    return c

def ChebyshevApprox(g, order, quad_points = 500):  # assuming f : [0, pi] -> R
    c = np.zeros(order+1)
    # a = np.pi / 2
    a = 2/2
    for k in range(1, order + 2):
        Integrand = lambda x: np.cos((k - 1) * x) * g(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c

def cheby_op2(L, c, range1, range2):
    # sgwt_cheby_op : Chebyshev polynomial of Laplacian applied to vector
    #
    # Compute (possibly multiple) polynomials of laplacian (in Chebyshev
    # basis) applied to input.
    #
    # Coefficients for multiple polynomials may be passed as a cell array. This is
    # equivalent to setting
    # r{1}=sgwt_cheby_op(f,L,c{1},arange)
    # r{2}=sgwt_cheby_op(f,L,c{2},arange)
    # ...
    #
    # but is more efficient as the Chebyshev polynomials of L applied
    # to f can be computed once and shared.
    #
    # Inputs:
    # f- input vector
    # L - graph laplacian (should be sparse)
    # c - Chebyshev coefficients. If c is a plain array, then they are
    #     coefficients for a single polynomial. If c is a cell array,
    #     then it contains coefficients for multiple polynomials, such
    #     that c{j}(1+k) is k'th Chebyshev coefficient the j'th polynomial.
    # arange - interval of approximation
    #
    # Outputs:
    # r - result. If c is cell array, r will be cell array of vectors
    #     size of f. If c is a plain array, r will be a vector the size
    #     of f.
    maxM = c.shape[0] #order+1
    # M=0#zeros(size(Nscales))
    # for j=1:Nscales
    #     M(j)=numel(c{j})
    # end
    # assert (M>=2)
    
    #Twf_new = T_j(L) f,Twf_cur T_{j-1}(L) f,  TWf_old T_{j-2}(L) f
    # 
    a1=(range2-range1)/2
    a2=(range2+range1)/2
    N = L.shape[0]
    L_hat = L - a2*sp.eye(N)
    Twf_old=sp.eye(N) #j=0
    Twf_cur=L_hat/a1 # j=1
    # 
    r = .5*c[0]*Twf_old + c[1]*Twf_cur
    for k in range(2, maxM): #order = 2,3, ...., order (order = maxM-1)
        # Twf_new = (2/a1)*smvp(sparse(L_hat),Twf_cur)-Twf_old
        Twf_new = (2/a1)*L_hat*Twf_cur-Twf_old
        if k+1<=maxM:
            r=r+c[k]*Twf_new
        Twf_old=Twf_cur
        Twf_cur=Twf_new
    return r
