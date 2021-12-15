import numpy as np

def eye_crop(img,mask,padding_ratio=0.75,min_maskratio=0.001):
    
    mask_ratio=np.sum(mask)/np.sum(np.ones_like(mask))    
    if mask_ratio<min_maskratio:
        raise Exception("minimum mask ratio not met")
    
    maskcoords=np.argwhere(mask)
    [y1,x1]=np.min(maskcoords,0)
    [y2,x2]=np.max(maskcoords,0)

    h,w,_=img.shape

    padding=int((x2-x1)*padding_ratio)
    
    y1=max(y1-padding,0)
    x1=max(x1-padding,0)
    
    y2=min(y2+padding,h)
    x2=min(x2+padding,w)
    
    cropped=img[y1:y2,x1:x2,:]
    
    return cropped