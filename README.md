### AAAI2018-[Modeling Attention and Memory for Auditory Selection in a Cocktail Party Environment](https://github.com/jacoxu/ASAM/blob/master/AAAI2018-Modeling%20Attention%20and%20Memory%20for%20Auditory%20Selection%20in%20a%20Cocktail%20Party%20Environment.pdf)     

=======================================================================    

Our demo code is implemented in Keras (writtern in Python, and the backend is theano).    

Usage:    
$python main_run.py    
or execute it in terminal background:     
$bash run.sh    


Notice:    
(1). In order to aviod the version mismatch of [Keras](https://keras.io/), we fork the verison_1.2.2 of Keras into this project.    
(2). We use Matlab version of BSS_eval to evaluate NSDR.    


![Figure 1: Auditory Attention](http://wx3.sinaimg.cn/mw690/697b070fly1fo99vmp5njj20v10cw0vt.jpg)

Figure 1: Two specific attention tasks for auditory selection in a three speech mixture environment. One is top-down task-specific attention, and the other is bottom-up stimulus-driven attention.     
       
![Figure 2: Framework](http://wx3.sinaimg.cn/mw690/697b070fly1fo99vpoevkj21970hnwiq.jpg)

Figure 2: An illustration of our Auditory Selection with Attention and Memory (ASAM). (a): The overall architecture of the proposed ASAM. (b): Life-long memory module to memory the prior knowledge. In top-down attention scene, the dashed boxes and arrow are only conducted in the training phase and removed in the evaluation time.             

![Figure 3: Attention Heat Map](http://wx3.sinaimg.cn/mw690/697b070fly1fo99vsoeg1j21kw0xh4qq.jpg)

Figure 3: Effects of attention with different amounts of stimulus on one male and female mixture sample from WSJ0. (a) shows the SIR (Signal-to-Interference Ratio), SAR (Signal-to-Artifacts Ratio) and NSDR results, (b)-(d) are the auditory stimuli whose magnitudes are divided by the maximum magnitude, (e) is the mixture input spectrogram, (i) is the target spectrogram, (f)-(h) are attention maps based on the corresponding auditory stimuli and (j)-(l) are the corresponding predictions with their NSDR performances.                


![](https://camo.githubusercontent.com/0e32abe541a386cbaf8370777b4b55c35d31657d/68747470733a2f2f692e6372656174697665636f6d6d6f6e732e6f72672f6c2f62792d6e632f342e302f38387833312e706e67)    

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).    
