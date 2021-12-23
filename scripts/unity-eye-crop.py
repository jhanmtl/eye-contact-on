import argparse
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import uuid
import h5py

def collect_paths(workdir):
    imgpaths=[os.path.join(workdir,i) for i in os.listdir(workdir) if ".jpg" in i]
    jsonpaths=[os.path.join(workdir,i) for i in os.listdir(workdir) if ".json" in i]

    imgpaths.sort()
    jsonpaths.sort()

    for ip,jp in zip(imgpaths,jsonpaths):
        assert ip.split(".")[0] == jp.split(".")[0]

    return imgpaths,jsonpaths

def parse_strcoords(strlist,img_height,return_int=True):
    x=[]
    y=[]
    
    for s in strlist:
        s=s.replace("(","")
        s=s.replace(")","")
        s=s.split(",")
        x.append(float(s[0]))
        y.append(img_height-float(s[1])) # y-coord in the Unity generator space is positive in the upward direction, needs to be flipped in the image space
    
    x=np.array(x)
    y=np.array(y)
    
    if return_int:
        x=x.astype(int)
        y=y.astype(int)
    
    xy=np.concatenate((x[:,None],y[:,None]),axis=1)
    
    return xy    

def extract_gazevec(metadata):
    vec_str=metadata['eye_details']['look_vec']
    vec_str=vec_str.replace("(","")
    vec_str=vec_str.replace(")","").split(",")
    
    x=float(vec_str[0])
    y=-1*float(vec_str[1])
    
    return np.array([x,y])


def parse_metadata(datapath,h):
    with open(datapath,"r") as jf:
        metadata=json.load(jf)  

    interior_landmarks=parse_strcoords(metadata['interior_margin_2d'],h)
    iris_landmarks=parse_strcoords(metadata['iris_2d'],h)
    gaze_vec=extract_gazevec(metadata)    

    return interior_landmarks, iris_landmarks, gaze_vec    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",dest="input_dir",required=True)
    parser.add_argument("--out-dir",dest="output_dir",required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    out_hfpath = os.path.join(output_dir,"data.hdf5")
    out_labelpath = os.path.join(output_dir, "label.json")

    if os.path.exists(out_hfpath):
        os.remove(out_hfpath)

    hf=h5py.File(out_hfpath,"a")

    imgpaths,jsonpaths = collect_paths(input_dir)

    landmarks={}
    for ipath,jpath in tqdm(zip(imgpaths,jsonpaths)):
        uid=uuid.uuid4().hex
        img=Image.open(ipath)
        hf.create_dataset(uid,data=np.asarray(img))

        interior_lmk, iris_lmk, gaze_vec = parse_metadata(jpath, img.height)
        landmarks[uid]={"interior":interior_lmk.tolist(),"iris":iris_lmk.tolist(),"gaze_vector":gaze_vec.tolist()}
    
    with open(out_labelpath,"w") as jf:
        json.dump(landmarks,jf)

if __name__=="__main__":
    main()
    



      
