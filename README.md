# Network Comparison

## About
A embedding based network comparison method, named D_NE, which considers the global structural information. In detail, we calculate the distance between nodes through the vector extracted by DeepWalk and quantify the network dissimilarity by spectral entropy based Jensen-Shannon divergences of the distribution of the node distances. 

For more details, please refer the following paper:
> Zhipeng Wang, Xiu-Xiu Zhan, Chuang Liu, Zi-Ke Zhang: Quantifification of network structural dissimilarities based on graph embedding. arXiv preprint: 2111.13114




## About file 

We have two folder, where the folder 'code' contains the concrete implementations about the method D_NE, D_SP, D_C, respectively and  the folder 'data' contains the 12 real networks using in this paper.

## Usages

We should input two networks and the parameters used in this method.  To compare the differences between two networks, you can use the data in our "data" folder or you can use your own network, but you will need to add the new network to the "data" folder.

**Method** **D_NE**
```
python D_NE.py --data_name1  network_name1 --data_name2  network_name2 
```

**Method** **D_SP**
```
python D_SP.py --data_name1  network_name1 --data_name2  network_name2 
```

**Method** **D_C**
```
python D_C.py --data_name1  network_name1 --data_name2  network_name2 
```

## Contact
If you have any question about the paper or the code, 
please contact us.
**Zhipeng Wang**, **19906810976@163.com**

Please cite that paper if you use this code. Thanks!