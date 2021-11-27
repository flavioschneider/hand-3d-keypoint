## Local  

Run experiment locally in debug mode example: 
```console
python run.py experiment=exp_f004 trainer.gpus=0 debug=True  
```

Local test from trained models example:
```console 
python3 run.py experiment=test_f004 +model.checkpoint_path=/Users/flavioschneider/Documents/NextMachina/trained_models/'f004_epoch\=07-valid_loss\=1.57.ckpt' trainer.gpus=0 test=True debug=True
```

## Leonhard 

```console 
source venv/bin/activate
module load eth_proxy gcc/6.3.0 python_gpu/3.8.5
```

Run example:
```console
bsub -W 24:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py experiment=exp_f004
```

Test from checkpoint example:
```console
bsub -W 01:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py experiment=test_f033 +model.checkpoint_path=/cluster/scratch/scflavio/mp_project/logs/ckpts/2021-06-20/00-31-30/'last.ckpt' test=True model.num_rotations=10 
```

Resume running from checkpoint example:
```console
bsub -W 24:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python3 ./run.py experiment=test_f033 +model.checkpoint_path=/cluster/scratch/scflavio/mp_project/logs/ckpts/2021-06-15/08-35-27/'last.ckpt' model.lr=0.00001 model.use_scheduler=False datamodule.batch_size=1
```


