# BEVGM: A Visual Place Recognition Method with Bird's Eye View Graph Matching

Public code of "BEVGM: A Visual Place Recognition Method with Bird's Eye View Graph Matching", which is accepted by RAL 2024.

![image-20240411092542211](../../proj/BEVGM/README.assets/image-20240411092542211.png)



## Quick Start

### Dependencies

+ python==3.8
+ pytorch==1.10.1
+ networkx=2.8.6
+ pygmtools=0.3.1
+ linsatnet=0.0.6
+ faiss
+ pyquaternion
+ matplotlib
+ opencv
+ scikit-learn
+ pyyaml
+ shapely
+ tqdm

### Preprocess

+ Depth estimation: [Lite-Mono](https://github.com/noahzn/Lite-Mono)
  + model: `lite-mono-8m (1024x320)`
+ Semantic segmentation: [OneFormer](https://github.com/SHI-Labs/OneFormer)
  + config: `oneformer_convnext_large_bs16_90k.yaml` 

+ Dataset: [The SYNTHIA dataset](https://synthia-dataset.net/)
  + springB: `/your/dataset/path/SYNTHIA/SYNTHIA_VIDEO_SEQUENCES/SYNTHIA-SEQS-02-DAWN/RGB/Stereo_Left/ /Omin_B/{000000.png~000699.png}`
  + dawnF:  `/your/dataset/path/SYNTHIA/SYNTHIA_VIDEO_SEQUENCES/SYNTHIA-SEQS-02-DAWN/RGB/Stereo_Left/Omni_F/{000000.png~000699.png}) `

### Usage

1. Generate query-dataset pair

```shell
python script/gen_pair_unity.py
```

2. Generate BEV graphs

```shell
./script/synthia_graph.sh
```

3. Match BEV graphs

```shell
./script/synthia_match.sh
```

4. Eval

```shell
python script/eval_all.py -e SYNTHIA_springB_dawnF -p your/match/result/path
```

## Files

TODO

## Citation

