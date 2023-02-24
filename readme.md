## Problem Statement

Creating a U-Shaped Mecial Image segmentation that uses a both Federated and Split learning to increase the model privacy with decentralized machine learning and reducing the computation cost on the agents. The problem also is defined to tackle the drift between local and global model and their parameters that is caused due to data homogenity that is mostly encountered in Medical data ML cases.

> Note:- data homogenity cause drift as a at local level there can exists a large amount of data with homogeneous composition (similar traits and charactertics) which would set a specific parameters at global model, but other introducing other local level data with varying level of homogenity from the pre-establised dataset can cause a large drift on the local and global model in general.



## Problem Solution

The problem solution here to the above problem can be divided into 3 parts

1. Creating a FCT(Fully Convolution Transformer) to use as a Medical Image segmentation model.
2. Dividing the model into 3 parts as `head`, `body` and `tail` where the data, the `head` (initial layers) and `tail`(end layer) would be at a agent(s) sand the main model `body` with large layers would be at the computational server.
3. Implementation of  Dynamic Weight Correction Strategy (DWCS) to stabilize the training process and avoid model drift.

---

**For Part 1**

The first component of the FCT layer is Convolutional Attention. The output is applied with Depthwise-Convolutions in the projection layer to retain the spatial structure  and To leverage spatial context from images of data without the need for positional encoding, leading to a simpler model, this is done by replacing linear projections with Depthwise-Convolutions, on the MHSA(multi-head self attention) block. The Wide-Focus module then applies dilated convolutions at linearly increasing receptive fields to the MHSA output, the residual connections here help enhance feature propagation throughout the layer. The output from Wide-Focus in then proagated through the transformer encoder and decoder layer and this complete the architechtire of FCT implementation.

```python
def WideFocus(data):
    x1 = Conv2d(data)
    x1 = dropout(x1)
    x2 = Conv2d(data)
    x2 = dropout(x2)
    add_data = add(x1, x2)
    x3 = Conv2d(add_data)
    return x3

def transformer(data):
    x = Attention(data)  # Depthwise-Convolutions added on MHSA
    x = Conv2d(data)
    x = Normalize(data)
    x = WideFocus(data)
    return x

def encoder(data):
    x = Upsampler(data)
    x = Normalizer(x)
    x = Conv2d(x)
    x = MaxPool(x)
    x = transformer(x)
    return x

def decoder(data):
    x = Normalizer(data)
    x = Upsampler(x)
    x = Conv2d(x)
    x = transformer(x)
    return x

def output_layer(data):
    x = Normalizer(data)
    x = Upsampler(x)
    x = Sequential(x)
    return x

def overall_model(data):
    first = encoder(data)
    second = encoder(first)
    third = decoder(second)
    fourth = decoder(third)
    fifth = output_layer(fourth)
    return fifth
```

----

**For Part 2**
`SL`(Split Learning) has the advantage of model privacy protection, but it cannot achieve parallel training. `FL`(Federated Learning) can parallelize training, but it requires huge client computational sources, and no model privacy is committed. So to tackle the problem, the paper proposes a split method without sharing the input, model parameters, output and label of different parties. The full network is split into three parts, including `head`, `body`, and `tail` networks. The lightweight head and tail networks are hosted in clients to reduce the local computational costs, and the computational resource-required body networkhosted in the server with high-performance computational resources.

At the beginning of training, client-side and server-side models are initialized in the aggregation and computation servers, respectively. All clients perform forward
propagation of head models locally `M-h` and then deliver the encoded results to the computation server. With the high computational server, the learning, the forward propagation of body models can be executed in parallel `M-b`. At the end of the forward path, parameters is delivered to the clients again to generate the final prediction
of tail networks as `M-t`. 
After forward propagation, each client calculates the loss and starts backpropagation. Concretely, the gradients about `M-t` body parameters are calculated at first. Then, the gradients are transmitted to the computation server, and the server executes the backpropagation on `M-b` and deliver the gradients of head parametrs to clients. Finally, with the received gradients, the client executes the backpropagation of `M-h` .

This in turn completes one backpropagation pass between clients and the server. To make full use of all local data and get optimal global models, we aggregate the client-side and server-side with the N number of clients.

```python
def HeadForward(data):  # Subset of a overall larger dataset for split learning
    x = Model_Head(data)
    return x

def TailMain(body_gradient, label_data):  # Label of the subset of data passed in `HeadForward`
    pred_label = Model(gradient=body_gradient, parameters)
    Loss = Loss_Function(pred_label, label_data)
    body_gradient_correction = f(Backpropagation on tail parameters, Loss)
    return Diff(Loss)/ Diff(body_gradient_correction)

def HeadBack(tail_backprop):

    head_gradient_correction = Backprop(network parameter of head, tail_backprop)
    return head_gradient_correction


def aggregation_server(number_execution):
    if number_execution == 0:
        head_grad, tail_grad = Initialize()
    else:
        head_grad, tail_grad = Overall_grad_Dataset()
        head_grad = DWCS_Correct(head_grad, head_grad_prev_iteration)
        tail_grad =  DWCS_Correct(tail_grad, tail_grad_prev_iteration)

def DWCS_Correct(current_gradient, previous_iteration_gradient):
    Loss_weight = WeightLossCorrection(current_gradient, previous_iteration_gradient)

    return previous_iteration_gradient + learning_rate * Loss_weight

def overall_training_model():
    body_grad = FCT_Model()  # Dataingestion and output sigmoid layers would be removed
    for iter in range(0, rounds):  # Number if over all iteration needed
        head_grad, tail_grad = aggregation_server()
        with parallel_processing client(1 , N):
            body_grad_current = body_grad_prev_iteration

        for epoch in range(o, MAX_EPOCH):
            with parallel_processing client(1 , N):
                head_param = HeadForward(data)
                body_param = Model_Body(head_params, body_grad_current)
                loss_with_backprop_tail = TailMain(body_param)
                loss_with_backprop_body = HeadBack(loss_with_backprop)
                body_grad_prev_iteration = body_grad_current - learning_rate * loss_with_backprop_body
    
    body_grad = Overall_body_grad_Dataset()
    body_grad = DWCS_Correct(body_grad, body_grad_prev_iteration)

```

---

**For Part 3**

Since there is always a distribution gap between Di and D in practice, local training will lead the local model to work badly in other data domains. As a result, it may generate a poor optimization solution to the global model and cause model to collapse after aggregation. This problem happens more commonly in healthcare/medical tasks, in which the collected data inevitably suffers from serious data homogenity.

To recover from this situation, we propose DWCS to avoid the model drift problem. Specifically, we treat the model of the last communication round as the anchor model and propose a weight correction loss to quantify the drift between the anchor model and its adjacent communication round model. Then we get the correction model by minimizing the weight correction loss, and the weighted sum of the correction and last round models is treated as the final result. 

Skeletal implementation is given above in the function `DWCS_Correct`