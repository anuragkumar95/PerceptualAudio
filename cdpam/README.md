# Contrastive learning-based Deep Perceptual Audio Metric [[CDPAM]](https://github.com/pranaymanocha/PerceptualAudio/tree/master/cdpam) [[Paper]](https://arxiv.org/abs/2102.05109) [[Website]](https://pixl.cs.princeton.edu/pubs/Manocha_2021_CCL/index.php)

[[Real-time Speech Enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_se.php)

[[MelGAN Single Speaker enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_mel_single.php)

[[MelGAN Cross Speaker enhanced Examples]](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/clips/clips_mel_cross.php)

[Download Pretrained SE + Vocoder trained models](https://percepaudio.cs.princeton.edu/Manocha20_CDPAM/mel_se.zip)

**Contrastive Learning based Perceptual Audio Similarity**

[Pranay Manocha](https://www.cs.princeton.edu/~pmanocha/), [Zeyu Jin](https://research.adobe.com/person/zeyu-jin/), [Richard Zhang](http://richzhang.github.io/), [Adam Finkelstein](https://www.cs.princeton.edu/~af/)   

<img src='https://richzhang.github.io/index_files/audio_teaser.jpg' width=500>

This is a Pytorch implementation of our new and improved audio perceptual metric. It contains (0) minimal code to run our perceptual metric (CDPAM).

## (0) Usage as a loss function

### Minimal basic usage as a distance metric

Running the command below takes two audio files as input and gives the perceptual distance between the files. It should return (approx)**distance = 0.1696**. Some GPU's are non-deterministic, and so the distance could vary in the lsb.

Installing the metric (CDPAM - perceptual audio similarity metric)
```bash
pip install cdpam
```
Please also run ``` pip install -r requirements.txt``` to install all the requirements required to run the metric. Please make sure to install the latest version of cdpam.

Using the metric is as simple as: 
```bash
import cdpam
loss_fn = cdpam.CDPAM()
wav_ref = cdpam.load_audio('sample_audio/ref.wav')
wav_out = cdpam.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_ref,wav_out)
```
You can pass the device you want to run the metric on (e.g., 'cpu' or 'cuda:0') while calling metric like this: 
```bash 
loss_fn = cdpam.CDPAM(dev='cuda:0')
```

### Citation

If you use our metric for research, please use the following to cite.

```
@inproceedings{Manocha:2021:CCL,
   author = "Pranay Manocha and Zeyu Jin and Richard Zhang and Adam Finkelstein",
   title = "{CDPAM}: Contrastive learning for perceptual audio similarity",
   booktitle = "ICASSP 2021, To Appear",
   year = "2021",
   month = jun
}
```

### License
The source code is published under the [MIT license](https://choosealicense.com/licenses/mit/). See LICENSE for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us. The primary contact is Pranay Manocha.<br/>
