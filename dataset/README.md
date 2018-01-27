
These shell scripts generate the WSJ0 and THCHS-30 datasets used in    
         Jiaming Xu, Jing Shi, Guangcan Liu, Xiuyi Chen, Bo Xu.    
         "Modeling attention and memory for auditory selection in a cocktail party environment"    
         AAAI, 2018.    
          
WSJ0: https://catalog.ldc.upenn.edu/LDC93S6A    

    Garofalo, J.; Graff, D.; Paul, D.; and Pallett, D. 2007. Csr-i (wsj0) complete. Linguistic Data Consortium, Philadelphia.    

THCHS-30: https://github.com/kaldi-asr/kaldi/tree/master/egs/thchs30    
 
    Wang, D., and Zhang, X. 2015. Thchs-30: A free chinese speech corpus. arXiv preprint arXiv:1512.01882.    

WSJ0 should be converted into wav files by sph2pipe tool as follows:    

    sph2pipe -f wav filename.wv1 filename.wav    

If you have installed Kaldi, you can directly find the tool at the path: $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe, if not, you can direct download from the website: http://sourceforge.net/projects/kaldi/files/sph2pipe_v2.5.tar.gz    

The GCC command for sph2pipe in Linux is that:    

gcc -o sph2pipe  *.c -lm    
