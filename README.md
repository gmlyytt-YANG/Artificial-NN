# Artificial-NN

This is a self-define NN predicting model. It will be a good reference to new learner of NN model, especially for those who are confused of back propagation.

- **NN model structure**

    ![nn plot](https://github.com/gmlyytt-YANG/img-repo/blob/master/csdn/%E9%97%A8%E5%A4%96%E6%B1%89%E5%85%A5%E9%97%A8DL%20--%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%8E%A8%E5%AF%BC%E5%8F%8Apython%E5%AE%9E%E7%8E%B0/nn_plot.png)
   
- **loss function**

    mean square loss 
    

- **core function of the code** 

    - function `_forward`
    
        Pass input `X` and compute the output step by step.
    
    - function `_backward`
    
        Compute the `loss` and then back inference the gradient of components the nn structure.
        
        Computation of the gradient of weight matrix is shown as follows.
        
        ![h1](https://github.com/gmlyytt-YANG/img-repo/blob/master/csdn/%E9%97%A8%E5%A4%96%E6%B1%89%E5%85%A5%E9%97%A8DL%20--%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%8E%A8%E5%AF%BC%E5%8F%8Apython%E5%AE%9E%E7%8E%B0/nn_plot_bp.png)
        
        Computation of the gradient of sigmoid output is shown as follows.
        
        ![h2](https://github.com/gmlyytt-YANG/img-repo/blob/master/csdn/%E9%97%A8%E5%A4%96%E6%B1%89%E5%85%A5%E9%97%A8DL%20--%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%8E%A8%E5%AF%BC%E5%8F%8Apython%E5%AE%9E%E7%8E%B0/nn_plot_bp_2.png)
    
- **bonus**
    
    Add l1 and l2 regularization to back propagation. And the comparision result of no regularization, l1 regularization and l2 regularization is shown as follows.
    
    ![h3](https://github.com/gmlyytt-YANG/img-repo/blob/master/csdn/%E9%97%A8%E5%A4%96%E6%B1%89%E5%85%A5%E9%97%A8DL%20--%20%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%8E%A8%E5%AF%BC%E5%8F%8Apython%E5%AE%9E%E7%8E%B0/nn_plot_regularization.png)