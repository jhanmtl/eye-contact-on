from src.data_tools import eye_crop
import os
import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm
import uuid
import argparse

def load_img_mask(imgpath,labelpath):
    img=np.array(Image.open(imgpath))
    label=np.array(Image.open(labelpath))

    lefteyemask=label==4
    righteyemask=label==5   

    return img, lefteyemask, righteyemask 

def load_paths(workdir):
    splits=["train","test","val"]

    imgpaths=[]
    labelpaths=[]
    for s in splits:
        imgdir=os.path.join(workdir,"{}/images".format(s))
        labeldir=os.path.join(workdir,"{}/labels".format(s))

        imgpaths+=[os.path.join(imgdir,i) for i in os.listdir(imgdir)]
        labelpaths+=[os.path.join(labeldir,i) for i in os.listdir(labeldir)]
    
    imgpaths.sort()
    labelpaths.sort()

    assert len(imgpaths)==len(set(imgpaths))
    assert len(labelpaths)==len(set(labelpaths))

    for ip,lp in zip(imgpaths,labelpaths):
        img_id=ip.split("/")[-1].split(".")[0]
        label_id=lp.split("/")[-1].split(".")[0]

        assert img_id==label_id
    
    return imgpaths,labelpaths

def crop_serialize(img_path,cropkind,img,mask,padding,threshold,hf):
    try:
        crop=eye_crop(img,mask,padding,threshold)
        uid=uuid.uuid4().hex
        hf.create_dataset(uid,data=crop)

        return None
    
    except Exception as e:
        s="{} caused {} to fail crop at {}".format(str(e),img_path,cropkind)
        return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workdir",required=True)
    parser.add_argument("-f",dest="overwrite",action='store_true')
    parser.add_argument("-o",required=True)
    parser.add_argument("--padding",required=True)
    parser.add_argument("--threshold",required=True)

    parser.set_defaults(overwrite=False)

    args=parser.parse_args()

    workdir=args.workdir
    out_hfpath=args.o
    overwrite=args.overwrite
    padding_ratio=float(args.padding)
    threshold=float(args.threshold)

    if overwrite:
        if os.path.exists(out_hfpath):
            os.remove(out_hfpath)
    
    out_hf=h5py.File(out_hfpath,"a")

    imgpaths,labelpaths=load_paths(workdir)
    print("{} total images to crop".format(len(imgpaths)))

    
    crop_results=[]

    for ip, lp in tqdm(zip(imgpaths, labelpaths)):
        
        img,lefteyemask,righteyemask=load_img_mask(ip,lp)

        left_res=crop_serialize(ip,"left",img,lefteyemask,padding_ratio,threshold,out_hf)
        right_res=crop_serialize(ip,"right",img,righteyemask,padding_ratio,threshold,out_hf)

        if left_res:
            crop_results.append(left_res)
        if right_res:
            crop_results.append(right_res)


    print("{} failed crops".format(len(crop_results)))
    for cr in crop_results:
        print(cr)

        


if __name__=="__main__":
    main()