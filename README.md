# Network Comparison

## About
A embedding based network comparison method, named D_NE, which considers the global structural information. In detail, we calculate the distance between nodes through the vector extracted by DeepWalk and quantify the network dissimilarity by spectral entropy based Jensen-Shannon divergences of the distribution of the node distances. 

## About file 

We have two folder, where the folder 'code' contains the concrete implementations about the method D_NE and  the folder 'data' contains the 12 real networks using in this paper.

## Usages

We should input two networks and the parameters used in this method.  To compare the differences between the two networks, you can use the data in our "data" folder or you can use your own network, but you will need to add the new network to the "data" folder.
```
python D_NE.py --data_name1 Chesapeake --data_name2 Contiguous 
```
## Contact
If you have any question about the paper or the code, 
please contact us.
**Zhipeng Wang**, **19906810976@163.com**