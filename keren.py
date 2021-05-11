# implementation of the article "Image Sequence Enhancement Using Sub Pixel Displacements" 
# implemented by Shlomi Amitai. based on the matlab implementation by LCAV.

import numpy as np
import cv2
import scipy
from numpy.fft import *
import imutils

def fspecial_py(shape=3,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = (shape, shape)
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def keren(imlist):
    for imnr in range(1, len(imlist)):
        # % construct pyramid scheme
        m, n= imlist[0].shape
        phi_est=[]
        delta_est=[]
        lp = fspecial_py(2,1)
        im0=[]
        im1=[]
        im0.append(imlist[0])
        im1.append(imlist[imnr])
        for i in range(1,3):
            m, n= im0[i-1].shape
            im0.append(cv2.resize(scipy.signal.convolve2d(im0[i-1],lp,'same'), (m//2, n//2), interpolation = cv2.INTER_CUBIC))
            im1.append(cv2.resize(scipy.signal.convolve2d(im1[i-1],lp,'same'), (m//2, n//2), interpolation = cv2.INTER_CUBIC))
    
        
        stot = np.array([0,0,0], dtype=np.float)
    #     % do actual registration, based on pyramid
        for pyrlevel in range(2,0,-1):
            f0 = im0[pyrlevel]
            f1 = im1[pyrlevel]
            
            y0,x0 = f0.shape
            xmean=x0//2
            ymean=y0//2
            x=np.kron(np.arange(-xmean,xmean),np.ones((y0,1)))
            y=np.kron(np.ones((x0, 1)),np.arange(-ymean,ymean)).transpose()
            
            sigma=1
            g1 = np.zeros((y0,x0))
            g2 = np.zeros((y0,x0))
            g3 = np.zeros((y0,x0))

            for i in range(y0):
                for j in range(x0):
                    g1[i,j] = -np.exp(-((i-ymean)**2+(j-xmean)**2)/(2*sigma**2))*(i-ymean)/2/np.pi/sigma**2 # d/dy
                    g2[i,j]=-np.exp(-((i-ymean)**2+(j-xmean)**2)/(2*sigma**2))*(j-xmean)/2/np.pi/sigma**2 # d/dx
                    g3[i,j]= np.exp(-((i-ymean)**2+(j-xmean)**2)/(2*sigma**2))/2/np.pi/sigma**2
    
            
            a=np.real(ifft2(fft2(f1)*fft2(g2))) # df1/dx, using circular convolution
            c=np.real(ifft2(fft2(f1)*fft2(g1))) # df1/dy, using circular convolution
            b=np.real(ifft2(fft2(f1)*fft2(g3)))-np.real(ifft2(fft2(f0)*fft2(g3))) # f1-f0
            R=c*x-a*y #  df1/dy*x-df1/dx*y
            
            a11 = sum(sum(a*a))
            a12 = sum(sum(a*c))
            a13 = sum(sum(R*a))
            a21 = sum(sum(a*c))
            a22 = sum(sum(c*c))
            a23 = sum(sum(R*c))
            a31 = sum(sum(R*a))
            a32 = sum(sum(R*c))
            a33 = sum(sum(R*R))
            b1 = sum(sum(a*b)) 
            b2 = sum(sum(c*b))
            b3 = sum(sum(R*b))
            Ainv = np.linalg.inv(np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]))
            s = np.dot(Ainv,np.array([b1, b2, b3]).transpose())
            st = s.copy()
            
            it=1
            num_rows, num_cols = f0.shape[:2]
            while ( (abs(s[0])+abs(s[1])+abs(s[2])*180/np.pi/20>0.1) and it<25):
    #             % first shift and then rotate, because we treat the reference image
                translation_matrix = np.float32([ [1,0,-st[0]], [0,1,-st[1]] ])
                f0 = cv2.warpAffine(f0, translation_matrix, (num_cols, num_rows))
                f0_ = imutils.rotate(f0, -st[2]*180/np.pi)
                b = np.real(ifft2(fft2(f1)*fft2(g3)))-np.real(ifft2(fft2(f0_)*fft2(g3)))
                s = np.dot(Ainv, np.array([sum(sum(a*b)), sum(sum(c*b)), sum(sum(R*b))]).transpose())
                st += s
                it+=1
  
            
            st[2]=-st[2]*180/np.pi
            # st = st';
            st[:2] = np.flip(st)[1:]
            stot[:2]=2*stot[:2]+st[:2]
            stot[2]=stot[2]+st[2]
            
            if pyrlevel>1:
                # first rotate and then shift, because this is cancelling the
                # motion on the image to be registered
                num_rows, num_cols = im1[pyrlevel-1].shape[:2]
                im1[pyrlevel-1] = imutils.rotate(im1[pyrlevel-1], -stot[2]) # degrees
                translation_matrix = np.float32([ [1,0,2*stot[1]], [0,1,2*stot[0]] ])
                im1[pyrlevel-1] = cv2.warpAffine(im1[pyrlevel-1], translation_matrix, (num_cols, num_rows)) # twice the parameters found at larger scale
   
        phi_est.append(stot[2])
        delta_est.append(stot[:2])
        return phi_est, delte_est

    

