import numpy as np
import matplotlib.pyplot as plt
import cv2
  
       
def reals(array):
    return np.invert(np.isnan(array))

def extremes(left, right):
    mask_l, mask_r = reals(left), reals(right)
    min_x, max_x = np.amin(left[mask_l]), np.amax(right[mask_r])
    min_z, max_z = np.argmin(left[mask_l]), np.argmax(right[mask_r])
    return min_x, min_z, max_x, max_z

def edges(left, right, thresh):
    minx, minz, maxx, maxz = extremes(left, right)
    return left < minx + thresh, right > maxx - thresh

def bottoms(left, right, thresh):
    mask_l, mask_r = reals(left), reals(right)
    edge_l, edge_r = edges(left, right, thresh)
    x_l, z_l = left[edge_l][-1], np.where(edge_l)[0][-1]
    x_r, z_r = right[edge_r][-1],np.where(edge_r)[0][-1]
    return x_l, z_l, x_r, z_r

def top_depth(x, left, right, thresh):
    x_l, z_l, x_r, z_r = bottoms(left, right, thresh)
    slope = (z_r - z_l)/(x_r - x_l)
    return slope*(x - x_l) + z_l

def rotation_matrix(left, right, thresh):
    x_l, z_l, x_r, z_r = bottoms(left, right, thresh)
    m = (z_r - z_l)/(x_r - x_l)
    x_m, z_m = (x_l + x_r)/2, (z_l + z_r)/2
    θ = np.arctan(m)
    return cv2.getRotationMatrix2D((x_m, z_m), θ*180/np.pi, 1)
    
def rotated(thresh_im, thresh):
    left = search(thresh_im, 'left')
    right = search(thresh_im, 'right')
    bottom = search(thresh_im, 'bottom')
    x, z = np.arange(thresh_im.shape[1]), np.arange(thresh_im.shape[0])
    M = rotation_matrix(left, right, thresh)
    return cv2.warpAffine(thresh_im, M, (thresh_im.shape[1],thresh_im.shape[0]))

def analyse(im0, lower, thresh_x, thresh_z, kernelsize=0, ratio=0, plot=False, arrshape=False, savename='pic.png', rotate=False, close=False):
    im1 = 255 - im0[:,:,0]
    ret1, thresh_im = cv2.threshold(im1, lower, 255, cv2.THRESH_BINARY)
    ims = {'0-orig': im0, '1-gray':im1, '2-thresh':thresh_im}
    if kernelsize != 0:
        kernel1, kernel2 = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) for k in kernelsize)
        morph1 = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel1)
        morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel2)
        ims['3-morph1'] = morph1
#         ims['4-morph2'] = morph2
    if rotate:
        im_rotated = rotated(ims[sorted(ims)[-1]], thresh_x)
        ims['5-rotated'] = im_rotated
    process = ims[sorted(ims)[-1]]
    left, right, bottom = search(process, 'l'), search(process, 'r'), search(process, 'b')
    x, z = np.arange(process.shape[1]), np.arange(process.shape[0])
    xl, zl, xr, zr = bottoms(left, right, thresh_x)
    dimensions = calc_dims(x, z, left, right, bottom, thresh_x, thresh_z, ratio=ratio)
    if plot:
        fig, ax = plotims( [ ims[name] for name in sorted(ims)] , dp=400, savename=savename, arrshape=arrshape)
        ax.flatten()[-1].plot(dimensions['x'],dimensions['t'],'r',linewidth=0.5)
        plt.tight_layout()
        plt.savefig(savename,dpi=400)
    if close==True:
        plt.close()
    for ref, val in zip(['xl','zl','xr','zr'],[xl,zl,xr,zr]):
        dimensions[ref] = val
    return dimensions, ims



def search(im, direction):
    m, n = im.shape
    if direction[0] == 'l':
        return np.array([np.where(im[i,:]!=0)[0][0]  if np.any(im[i,:]) else np.nan for i in range(m)])
    if direction[0] == 'r':
        return np.array([np.where(im[i,:]!=0)[0][-1] if np.any(im[i,:]) else np.nan for i in range(m)])
    if direction[0] == 't':
        return np.array([np.where(im[:,i]!=0)[0][0]  if np.any(im[:,i]) else np.nan for i in range(n)])
    if direction[0] == 'b':
        return np.array([np.where(im[:,i]!=0)[0][-1] if np.any(im[:,i]) else np.nan for i in range(n)])


def calc_dims(x, z, left, right, depth, thresh_x, thresh_z, ref=15, ratio=0, frac=0.7):
    maskl, maskr, maskz = reals(left), reals(right), reals(depth)
    min_x, min_z, max_x, max_z = extremes(left, right)
    x_l, z_l, x_r, z_r = bottoms(left, right, thresh_x)
    max_width = x_r - x_l
    width = right - left
    
    max_depth = np.amax(depth[maskz])
    top_edge = top_depth(x, left, right, thresh_x)    
    if ratio == 0:
        ratio = ref/max_width
    maskx_below = z > min(z_l,z_r)                    # below middle of top edge
    maskl = (z > z_l + 0.1*max_depth) & (z < z_l + .8*(max_depth-z_l))
    maskr = (z > z_r + 0.1*max_depth) & (z < z_r + .8*(max_depth-z_r))
    maskx_above = width > 0.1*max_width             # above part with zero width
    maskx  = np.logical_and(width > 0.1*max_width, maskx_below) # between top and bottom
    maskx2 = np.logical_and(width >  .4*max_width, maskx_below) # below top, and greater than frac thickness
    maskz    = depth > top_edge                      # below the top edge
    maskz_ratio = depth > top_edge + (max_depth - (z_l + z_r)/2)*frac       # below top edge PLUS half of max melted depth
    maskz_minus = depth > max_depth - thresh_z   # above bottom minus thresh_z
    
    return {'left': left,
            'right': right,
            'bottom': depth,
            'x': x,
            'z': z,
            'wtop':ratio*max_width,             # widest point
            'wbot':ratio*np.sum(maskz_minus),     # width where depth > fracz * max depth
            'wavg':ratio*np.mean(width[maskx]), # average width between top and bottom edge
            'lavg':ratio*np.mean(left[maskl] - x_l),
            'ravg':ratio*np.mean(x_r - right[maskr]),
            'wavg2':ratio*np.mean(width[maskx2]),
            'lc': (x_l, z_l),
            'rc': (x_r, z_r),
            't': top_edge, # The upper line
            'r': 1/ratio,  # px/cm ratio
            'dmax':ratio*np.amax(depth[maskz_ratio] - top_edge[maskz_ratio]), # top edge to lowest point
            'davg':ratio*np.mean(depth[maskz_ratio] - top_edge[maskz_ratio]), # average depth where deeper than top + thresh
            'davg2':ratio*np.mean(depth[maskz_minus] - top_edge[maskz_minus]), # average depth where deeper than top + thresh
            'ax':ratio**2*np.sum(width[maskx]), # area from summing widths
            'az':ratio**2*np.sum(depth[maskz] - top_edge[maskz]), # area from summing depths
            'mx':maskx,
            'ml':maskl,
            'mr':maskr,
            'mz':maskz_ratio} 

def drawcontour(im, contours, index, colour=(255,0,0), thickness=1, savename='pic.png', dp=400):
    imf = im.copy()
    cv2.drawContours(imf, contours, index, colour, thickness)
    fig, ax = plt.subplots()
    ax.imshow(imf, cmap='gray')
    ax.set(xticks=[],yticks=[])
    plt.savefig(savename, dpi=dp)        
        
def plotims(arr, savename='pic',dp=400, arrshape=False, close=False):
    l = len(arr)
    if l > 1:
        if arrshape != False:
            fig, ax = plt.subplots(arrshape[0],arrshape[1])
        else:
            r = arr[0].shape[1]/arr[0].shape[0]
            if r > 2:
                fig, ax = plt.subplots(l,1,gridspec_kw={'hspace':0,'wspace':0})
            else:
                fig, ax = plt.subplots(1,l,gridspec_kw={'hspace':0,'wspace':0})
        for axi, im in zip(ax.flatten(), arr):
            axi.imshow(im,cmap='gray')
            axi.set(yticks=[],xticks=[],aspect=1)
    else:
        fig, ax = plt.subplots()
        ax.imshow(arr[0],cmap='gray')
        ax.set(yticks=[],xticks=[],aspect=1)
    plt.tight_layout()
    if close:
        plt.close()
    return fig, ax

def plotwidth(dims, arr, savename='pic', r=False, close=False, arrshape=False, save=False):
    x, z, maskx, maskz = dims[0], dims[1], dims[2], dims[3]
    
    left, right, bottom, top = arr[0], arr[1], arr[2], arr[3]
    if len(x)/len(z) > 2:
        fig, ax = plt.subplots(3,1)
    else:
        fig, ax = plt.subplots(1,3)
    ax[0].scatter(left, z-top[0], s=1,color='gray')
    ax[0].scatter(right, z-top[0],s=1,color='gray')
    ax[0].scatter(x, bottom-top[0],s=1,color='gray')
    ax[0].plot(x, top-top[0], 'r', linewidth=0.5)
    ax[0].set(aspect=1,
              #xlim=[0, np.amax(right[maskx])],ylim=[np.amax(bottom[maskz]),0],
               ylabel='edges [cm]')
    ax[0].invert_yaxis()
    
    ax[1].plot(right-left, z)
    ax[1].plot(x, top, 'r', linewidth=0.5)
    ax[1].set(aspect=1,xlim=[0, np.amax(right[maskx])], ylim=[np.amax(bottom[maskz]),0],ylabel='width [cm]')
    if r != False:
        ax1.set(title='Ratio px/cm = {:.1f}'.format(r))
        
    ax[2].plot(x, bottom-top)
    ax[2].set(aspect=1,xlim=[0, np.amax(right[maskx])], ylim=[np.amax(bottom[maskz]),0],ylabel='depth [cm]')
    plt.tight_layout()
    if save:
        plt.savefig(savename)
    if close:
        plt.close()
    return fig, ax