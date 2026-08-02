import torch, numpy as np, cv2, json, glob, os, time
import torch.nn.functional as F
from train import TinyUNet, dice_loss, W, H
torch.set_num_threads(4)
HOLDOUT = ['003','009','013','020','025','031','034','015']
lref = json.load(open('overlays_lfit/lfit_results.json'))
xs,ys,names=[],[],[]
for f in sorted(os.path.basename(x)[:3] for x in glob.glob('dataset/two_lights/*_rgb.png')):
    if not lref.get(f,{}).get('found'): continue
    img = cv2.resize(cv2.imread(f'dataset/two_lights/{f}_rgb.png'),(W,H))
    m = np.zeros((480,640),np.uint8)
    cv2.fillPoly(m,[np.array(lref[f]['polygon_px'],np.int32)],255)
    m = cv2.resize(m,(W,H))
    xs.append(img); ys.append((m>127).astype(np.float32)); names.append(f)
xs=np.stack(xs); ys=np.stack(ys)
tr = [i for i,n in enumerate(names) if n not in HOLDOUT]
va = [i for i,n in enumerate(names) if n in HOLDOUT]
print('finetune train',len(tr),'holdout',len(va))
net = TinyUNet(); net.load_state_dict(torch.load('topbox_unet.pt',map_location='cpu'))
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
rng = np.random.RandomState(2); bs=6
for ep in range(40):
    net.train(); idx=rng.permutation(tr); tot=0.
    for i in range(0,len(idx),bs):
        b=idx[i:i+bs]
        xb=xs[b].astype(np.float32); yb=ys[b].copy()
        for j in range(len(b)):
            xb[j]=xb[j]*rng.uniform(0.8,1.2)+rng.uniform(-15,15)
            if rng.rand()<0.5: xb[j]=xb[j][:,::-1].copy(); yb[j]=yb[j][:,::-1].copy()
        x=torch.from_numpy(xb.clip(0,255).transpose(0,3,1,2)/255.).float()
        y=torch.from_numpy(yb).unsqueeze(1)
        logit=net(x); loss=F.binary_cross_entropy_with_logits(logit,y)+dice_loss(logit,y)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(b)
    sched.step()
    if ep%5==4 or ep==39:
        net.eval(); ious=[]
        with torch.no_grad():
            x=torch.from_numpy(xs[va].astype(np.float32).transpose(0,3,1,2)/255.)
            p=torch.sigmoid(net(x)).numpy()[:,0]>0.5
            for j in range(len(va)):
                gt=ys[va[j]]>0.5
                ious.append((p[j]&gt).sum()/max((p[j]|gt).sum(),1))
        print(f'ep {ep}: loss {tot/len(tr):.4f} HOLDOUT IoU {np.mean(ious):.3f}', flush=True)
torch.save(net.state_dict(),'topbox_unet_ft.pt')
print('saved topbox_unet_ft.pt')
