"""Task C: each model runs in a separate process, after other GPU experiments finish."""
from engine import *
import argparse,resource,threading,psutil,csv,platform,subprocess,os

def main(kind):
 d=O/'resources';d.mkdir(exist_ok=True);proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
 def sample():
  while not stop.is_set():peak[0]=max(peak[0],proc.memory_info().rss);stop.wait(.005)
 thread=threading.Thread(target=sample,daemon=True);thread.start();start=time.perf_counter();pipe=Pipeline(kind,8,False);torch.cuda.synchronize();load_sec=time.perf_counter()-start
 base_alloc=torch.cuda.memory_allocated();base_reserved=torch.cuda.memory_reserved();loaded_rss=proc.memory_info().rss
 torch.cuda.reset_peak_memory_stats();ids=SUB.validation_row.to_numpy()[:32];rows=[]
 @torch.inference_mode()
 def one(indices,measure=True):
  torch.cuda.synchronize();t0=time.perf_counter();imgs=[load_image(W/V.iloc[i].image_path) for i in indices];td=time.perf_counter()
  px=pipe.proc(images=imgs,return_tensors='pt')['pixel_values'].cuda();vo=pipe.enc.vision_model(pixel_values=px);g=pipe.enc.visual_projection(vo.pooler_output);g=g/g.norm(dim=-1,keepdim=True)
  if kind=='R1':
   attrs,descs=zip(*(compute_attributes_and_text(im) for im in imgs));a=torch.from_numpy(normalize(np.stack(attrs),pipe.norm)).cuda();tk=pipe.proc(text=list(descs),padding=True,truncation=True,return_tensors='pt').to('cuda');t=pipe.enc.get_text_features(**tk);t=t/t.norm(dim=-1,keepdim=True)
   patch=F.adaptive_avg_pool2d(vo.last_hidden_state[:,1:,:].transpose(1,2).reshape(-1,1024,24,24),7).flatten(2).transpose(1,2).half().float()
  else:a=t=patch=None
  torch.cuda.synchronize();tf=time.perf_counter();z=pipe.net(g,a,t,patch);torch.cuda.synchronize();th=time.perf_counter();p=z.softmax(-1).cpu().numpy();torch.cuda.synchronize();end=time.perf_counter()
  ref=np.load(W/pipe.run/'val_logits.npy')[indices];err=float(np.abs(z.cpu().numpy()-ref).max());assert err<.001,err
  return dict(n=len(indices),decode_s=td-t0,features_s=tf-td,head_s=th-tf,complete_s=end-t0,output_s=end-th,max_saved_logit_difference=err)
 for b in range(2):one(ids[b*8:(b+1)*8],False)
 for rep in range(5):
  for b in range(0,32,8):rows.append(dict(repeat=rep,**one(ids[b:b+8])))
 stop.set();thread.join();peak[0]=max(peak[0],proc.memory_info().rss);peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
 with (d/f'{kind}_batches.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 count=sum(r['n'] for r in rows);sums={k:sum(r[k] for r in rows) for k in ['decode_s','features_s','head_s','complete_s','output_s']}
 out={'model':kind,'scope':'EXPLORATORY validation; same first 32 predetermined images, batch8 FP32, two warmup batches and five passes; fresh worker; complete CLIP object loaded for both, not a stripped minimum-memory probe','checkpoint':pipe.run+'/best.pt','checkpoint_sha256':P['head_checkpoints'][kind][0]['checkpoint_sha256'],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),'batch':8,'precision':'FP32, no autocast, TF32 off','text_cache':False,'images':32,'measured_images_with_repetition':count,'load_seconds':load_sec,'model_baseline_allocated_bytes':base_alloc,'model_baseline_reserved_bytes':base_reserved,'peak_allocated_bytes':peak_alloc,'peak_reserved_bytes':peak_reserved,'loaded_RSS_bytes':loaded_rss,'peak_sampled_process_RSS_bytes':peak[0],'peak_ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'amortized_ms_per_image':{k:1000*x/count for k,x in sums.items()},'complete_throughput_images_s':count/sums['complete_s'],'warm_batch_median_ms':1000*float(np.median([r['complete_s'] for r in rows])),'warm_batch_p95_ms':1000*float(np.quantile([r['complete_s'] for r in rows],.95)),'caveats':['GPU synchronized wall-clock includes CPU work, decoding and output transfer; excludes model load and receipt checks','OS file caches may be warm; hardware-specific batch8 observation, not interactive single-image latency','RAM peak includes model import/load and allocator retention; CUDA reserved is not device-wide usage','Feature latency includes actual feature branches used by each model; same encoder parameters, images, precision and hardware','No training or holdout inference'],'protocol_sha256':sha(O/'protocol.json'),'script_sha256':sha(__file__)}
 (d/f'{kind}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('model',choices=['R1','PROBE']);main(parser.parse_args().model)
