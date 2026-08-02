#!/usr/bin/env python3
"""Real-background composites: articulated boxes pasted onto REAL frames.
The pasted box is on top by construction -> perfect labels, real domain."""
import numpy as np, cv2, json, os, sys, glob
import synth3 as S3

def gen(out, N, seed):
    rng = np.random.RandomState(seed)
    for sub in ('images','labels'): os.makedirs(f'{out}/{sub}', exist_ok=True)
    cuts = S3.load_cutouts()
    bgs = sorted(glob.glob('dataset/two_lights/*_rgb.png'))
    BH,BW = 480,640
    FLOOR = np.array([[95,130],[560,130],[600,420],[55,420]],np.int32)
    floor_mask = np.zeros((BH,BW),np.uint8); cv2.fillPoly(floor_mask,[FLOOR],255)
    base_s = 300.0/310.0/(S3.PXM/1000.0)
    recs={}
    for idx in range(N):
        bg = cv2.imread(bgs[rng.randint(len(bgs))]).astype(np.float32)
        n_boxes = rng.randint(1,4)
        top=None
        for b in range(n_boxes):
            c = cuts[rng.randint(len(cuts))]
            rgbc, ac, verts = S3.articulate(c, rng)
            ch,cw = ac.shape
            src4 = np.array([[0,0],[cw,0],[cw,ch],[0,ch]],np.float32)
            tj = S3.TILT_JIT*max(cw,ch)
            dst4 = src4 + rng.uniform(-tj,tj,(4,2)).astype(np.float32)
            Hp = cv2.getPerspectiveTransform(src4,dst4)
            ang = rng.uniform(0,360); sc = base_s*rng.uniform(0.93,1.07)
            th = np.deg2rad(ang)
            R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])*sc
            ok=False
            for _ in range(60):
                corners = np.array([[0,0,1],[cw,0,1],[cw,ch,1],[0,ch,1]]).T
                pc = Hp@corners; pc = (pc[:2]/pc[2]).T @ R.T
                mn,mx = pc.min(0), pc.max(0)
                nw,nh = int(mx[0]-mn[0])+2, int(mx[1]-mn[1])+2
                if nw>=BW or nh>=BH:
                    sc*=0.95; R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])*sc
                    continue
                px,py = rng.randint(0,BW-nw), rng.randint(0,BH-nh)
                if floor_mask[py+nh//2,px+nw//2]: ok=True; break
            if not ok: continue
            Aff = np.zeros((3,3)); Aff[:2,:2]=R; Aff[:2,2]=[px-mn[0],py-mn[1]]; Aff[2,2]=1
            Hfull = Aff@Hp
            wrgb = cv2.warpPerspective(rgbc,Hfull,(BW,BH))
            wa = cv2.warpPerspective(ac,Hfull,(BW,BH)).astype(np.float32)/255.
            sh = cv2.GaussianBlur(np.roll(np.roll(wa,8,0),6,1),(21,21),0)*0.45
            bg *= (1.-sh[...,None]*0.6)
            g = rng.uniform(0.9,1.08)
            bg = bg*(1-wa[...,None]) + (wrgb.astype(np.float32)*g).clip(0,255)*wa[...,None]
            vh = np.hstack([verts,np.ones((len(verts),1))]).T
            pv = Hfull@vh
            top = dict(poly=(pv[:2]/pv[2]).T, mask=(wa*255).astype(np.uint8), key=c['key'])
        if top is None: continue
        bg = (bg*rng.uniform(0.94,1.05)).clip(0,255).astype(np.uint8)
        name=f'rbg_{idx:04d}'
        cv2.imwrite(f'{out}/images/{name}.png', bg)
        cv2.imwrite(f'{out}/labels/{name}_topmask.png', top['mask'])
        recs[name]={'top_polygon_px':top['poly'].round(1).tolist()}
    json.dump(recs,open(f'{out}/labels/labels.json','w'),indent=1)
    print(out,'generated',len(recs))

if __name__=='__main__':
    gen(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
